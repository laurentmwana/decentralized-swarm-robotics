"""Reprise d’imitation pondérée ; sélection sur validation, sans lire test."""
import argparse
import json
from pathlib import Path
import shutil

import torch

from .local_target_model import load_model,save_model
from .local_target_train import prepare_data,batches
from .multiform_corrections import load_corrections,risk_weights
from .multiform_data import sha256


def weighted_mse(prediction,target,weight):
    if prediction.shape!=target.shape or weight.shape!=prediction.shape[:-1]:raise ValueError('Dimensions de perte incompatibles')
    if not torch.isfinite(weight).all() or (weight<=0).any():raise ValueError('Poids finis positifs requis')
    return (((prediction-target)**2).mean(dim=-1)*weight).sum()/weight.sum()


def train(dataset,corrections,source_run,output):
    torch.set_num_threads(1);dataset,corrections,source_run,output=map(Path,(dataset,corrections,source_run,output))
    correction_manifest=json.loads((corrections/'manifest.json').read_text(encoding='utf-8'))
    cfg=correction_manifest['training'];torch.manual_seed(cfg['seed'])
    source=json.loads((source_run/'run.json').read_text(encoding='utf-8'))
    checkpoint=(source_run/source['checkpoint']).resolve()
    if not checkpoint.is_relative_to(source_run.resolve()) or sha256(checkpoint)!=correction_manifest['source_model_sha256']:
        raise ValueError('Modèle source différent des corrections')
    manifest,base,selection=prepare_data(dataset);arrays={}
    for split in ('train','validation'):
        corrected=load_corrections(corrections,dataset,split)
        base_weights=torch.as_tensor(risk_weights(base[split][1].numpy(),base[split][2].numpy()))
        extra=tuple(torch.as_tensor(value,dtype=torch.bool if i==2 else torch.float32) for i,value in enumerate(corrected))
        arrays[split]=tuple(torch.cat((base[split][i],extra[i])) for i in range(4))+(torch.cat((base_weights,extra[4]*cfg['correction_multiplier'])),)
        selection[split]['correction_samples']=len(extra[0])
    model=load_model(checkpoint);model.train()
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,kind='multiform_local_target_training',status='training',
        experiment='risk_weighted_on_policy_corrections_v1',dataset=str(dataset.resolve()),
        dataset_manifest_sha256=sha256(dataset/'manifest.json'),corrections_manifest_sha256=sha256(corrections/'manifest.json'),
        source_model_sha256=sha256(checkpoint),model=model.metadata(),training=cfg,selection=selection,
        selection_criterion='weighted_train_validation_teacher_imitation_only',history=[])
    def save():(output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    generator=torch.Generator().manual_seed(cfg['seed']);best=float('inf');stale=0;best_path=None;save()
    for epoch in range(1,cfg['epochs']+1):
        totals={}
        for split in ('train','validation'):
            model.train(split=='train');loss_sum=weight_sum=0.
            with torch.set_grad_enabled(split=='train'):
                for own,neighbors,mask,target,weight in batches(arrays[split],cfg['batch_size'],generator=generator,shuffle=split=='train'):
                    if split=='train':optimizer.zero_grad(set_to_none=True)
                    loss=weighted_mse(model(own,neighbors,mask),target,weight)
                    if not torch.isfinite(loss):raise ValueError('Perte corrective non finie')
                    if split=='train':loss.backward();optimizer.step()
                    mass=float(weight.sum());loss_sum+=float(loss.detach())*mass;weight_sum+=mass
            totals[split]=loss_sum/weight_sum
        score=totals['validation'];report['history'].append(dict(epoch=epoch,train_weighted_mse=totals['train'],validation_weighted_mse=score))
        if score<best:
            best=score;stale=0;best_path=output/f'epoch-{epoch:04d}.pt';save_model(best_path,model)
            report.update(best_epoch=epoch,best_validation_weighted_mse=score)
        else:stale+=1
        save();print(f'Epoch {epoch}: train={totals["train"]:.6f}, validation={score:.6f}',flush=True)
        if stale>=cfg['patience']:break
    shutil.copyfile(best_path,output/'best.pt')
    report.update(status='trained',checkpoint='best.pt',epochs_completed=len(report['history']));save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','corrections','source-run','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();train(args.dataset,args.corrections,args.source_run,args.output)


if __name__=='__main__':main()
