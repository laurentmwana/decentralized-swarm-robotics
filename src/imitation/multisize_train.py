"""Un même GAT pour 12/24/64, lots CSR équilibrés par effectif."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

from .local_target_model import LocalTargetGAT,LocalTargetConfig,save_model
from .multiform_data import sha256


class LocalSamples(Dataset):
    def __init__(self,folder,split,seed,cap):
        if split not in ('train','validation'):raise ValueError('Train/validation uniquement')
        if type(cap)is not int or cap<1:raise ValueError('Budget positif requis')
        folder=Path(folder).resolve();manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('kind')!='multi_size_local_target_imitation' or manifest.get('status')!='complete' or manifest.get('quality_status')!='passed':
            raise ValueError('Collecte multi-effectifs complète et réussie requise')
        self.blocks=[];candidates={12:[],24:[],64:[]}
        for entry in manifest['episodes']:
            if entry['split']!=split:continue
            path=(folder/entry['samples']).resolve()
            if not path.is_relative_to(folder) or sha256(path)!=entry['sha256']:raise ValueError('Archive CSR corrompue ou extérieure')
            with np.load(path,allow_pickle=False) as archive:data={k:archive[k] for k in ('own','labels','edges','offsets')}
            own,labels,edges,offsets=(data[k] for k in ('own','labels','edges','offsets'))
            n=len(own)
            if own.shape!=(n,8) or labels.shape!=(n,2) or edges.ndim!=2 or edges.shape[1]!=4 or offsets.shape!=(n+1,):
                raise ValueError('Dimensions CSR incompatibles')
            if offsets[0]!=0 or offsets[-1]!=len(edges) or (np.diff(offsets)<0).any() or offsets.dtype.kind not in 'iu':
                raise ValueError('Offsets CSR incompatibles')
            if not all(np.isfinite(a).all() for a in (own,labels,edges)) or (np.linalg.norm(labels,axis=-1)>2.+1e-6).any():
                raise ValueError('Exemples CSR non finis ou hors borne')
            if not (own[:,7]==entry['count']).all() or entry['shape'] not in manifest['shape_splits'][split]:
                raise ValueError('Effectif ou partition CSR invalide')
            block=len(self.blocks);self.blocks.append(data)
            candidates[entry['count']].extend((block,i) for i in range(n))
        if any(not values for values in candidates.values()):raise ValueError('Un effectif est absent de la partition')
        target=min(cap,min(len(v) for v in candidates.values()));rng=np.random.default_rng(seed)
        self.indices=[];self.selection={}
        for count,values in candidates.items():
            selected=rng.choice(len(values),target,replace=False)
            self.indices.extend(values[i] for i in selected);self.selection[str(count)]=dict(available=len(values),selected=target)

    def __len__(self):return len(self.indices)

    def __getitem__(self,index):
        block,row=self.indices[index];data=self.blocks[block];a,b=data['offsets'][row:row+2]
        return data['own'][row],data['edges'][a:b],data['labels'][row]


def collate_local(samples):
    size=len(samples);degree=max(1,max(len(s[1]) for s in samples))
    edges=np.zeros((size,degree,4),np.float32);mask=np.zeros((size,degree),bool)
    for i,(_,neighbors,_) in enumerate(samples):edges[i,:len(neighbors)]=neighbors;mask[i,:len(neighbors)]=True
    return tuple(torch.as_tensor(a) for a in (np.stack([s[0] for s in samples]),edges,mask,np.stack([s[2] for s in samples])))


def train(dataset,output):
    torch.set_num_threads(1);dataset,output=map(Path,(dataset,output))
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'));cfg=manifest['training']
    torch.manual_seed(cfg['seed']);generator=torch.Generator().manual_seed(cfg['seed'])
    sets={s:LocalSamples(dataset,s,cfg['seed'],cfg['max_samples_per_count']) for s in ('train','validation')}
    loaders={s:DataLoader(sets[s],batch_size=cfg['batch_size'],shuffle=s=='train',generator=generator,
                          collate_fn=collate_local,num_workers=0) for s in sets}
    model=LocalTargetGAT(LocalTargetConfig(**manifest['model']),2.)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,kind='multi_size_local_target_training',status='training',
        dataset_manifest_sha256=sha256(dataset/'manifest.json'),storage_schema_version=3,model=model.metadata(),
        training=cfg,selection={s:sets[s].selection for s in sets},history=[])
    def save():(output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);best=float('inf');stale=0;best_path=None
    for epoch in range(1,cfg['epochs']+1):
        scores={}
        for split,loader in loaders.items():
            model.train(split=='train');total=components=0
            with torch.set_grad_enabled(split=='train'):
                for own,edges,mask,target in loader:
                    if split=='train':optimizer.zero_grad(set_to_none=True)
                    loss=((model(own,edges,mask)-target)**2).mean()
                    if not torch.isfinite(loss):raise ValueError('Perte multi-effectifs non finie')
                    if split=='train':loss.backward();optimizer.step()
                    total+=float(loss.detach())*target.numel();components+=target.numel()
            scores[split]=total/components
        report['history'].append(dict(epoch=epoch,train_mse=scores['train'],validation_mse=scores['validation']))
        if scores['validation']<best:
            best=scores['validation'];stale=0;best_path=output/f'epoch-{epoch:04d}.pt';save_model(best_path,model)
            report.update(best_epoch=epoch,best_validation_mse=best)
        else:stale+=1
        save();print(f'Epoch {epoch}: train={scores["train"]:.6f}, validation={scores["validation"]:.6f}',flush=True)
        if stale>=cfg['patience']:break
    shutil.copyfile(best_path,output/'best.pt');report.update(status='trained',checkpoint='best.pt',epochs_completed=len(report['history']))
    save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();train(args.dataset,args.output)


if __name__=='__main__':main()
