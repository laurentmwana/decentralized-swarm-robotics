"""Commande d'entraînement destinée à l'utilisateur. --dry-run n'optimise rien."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from .checkpoint import save_model, load_model
from .data import FEATURE_NAMES, load_episode, load_manifest
from .model import AccelerationGAT, GATConfig
from .objective import acceleration_mse


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    seed: int

    def __post_init__(self):
        for v in (self.epochs,self.batch_size,self.patience):
            if isinstance(v,bool) or not isinstance(v,int) or v<1:
                raise ValueError('Époques, lots et patience : entiers positifs requis')
        if not math.isfinite(self.learning_rate) or self.learning_rate<=0:
            raise ValueError('Taux d’apprentissage fini et positif requis')
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed<0:
            raise ValueError('Graine entière positive ou nulle requise')

    @classmethod
    def from_yaml(cls,path):
        with Path(path).open(encoding='utf-8') as stream:
            return cls(**yaml.safe_load(stream))


def prepare_data(dataset, *, corrections=None):
    manifest=load_manifest(dataset)
    if tuple(manifest['feature_names']) != FEATURE_NAMES or manifest['target']!='bounded_teacher_acceleration_m_s2':
        raise ValueError('Schéma de caractéristiques ou cibles incompatible')
    arrays={}
    # Le test n'est jamais chargé pour sélectionner les poids.
    for split in ('train','validation'):
        episodes=[load_episode(dataset,e,split) for e in manifest['episodes'] if e['split']==split]
        if split == 'train' and corrections is not None:
            from .corrections import load_corrections
            episodes.extend(load_corrections(corrections, dataset, manifest))
        features=np.concatenate([e['node_features'] for e in episodes])
        adjacency=np.concatenate([e['adjacency'] for e in episodes])
        targets=np.concatenate([e['targets'] for e in episodes])
        if features.ndim!=3 or features.shape[-1]!=8 or targets.shape!=features.shape[:2]+(2,) or adjacency.shape!=features.shape[:2]+(features.shape[1],):
            raise ValueError('Dimensions de graphes incompatibles')
        if not np.isfinite(features).all() or not np.isfinite(targets).all() or adjacency.dtype!=bool:
            raise ValueError('Données numériques invalides')
        limit=manifest['simulator']['physics']['max_acceleration']
        if (np.linalg.norm(targets,axis=-1)>limit+1e-9).any():
            raise ValueError('Cible dépassant la limite physique')
        arrays[split]=(torch.tensor(features,dtype=torch.float32),torch.tensor(adjacency),
                       torch.tensor(targets,dtype=torch.float32))
    return manifest,arrays


def batches(arrays,batch_size,*,generator=None,shuffle=False):
    count=len(arrays[0])
    order=torch.randperm(count,generator=generator) if shuffle else torch.arange(count)
    for start in range(0,count,batch_size):
        indices=order[start:start+batch_size]
        yield tuple(array[indices] for array in arrays)


def validation_mse(model,arrays,batch_size,device):
    model.eval()
    total,count=0.,0
    with torch.inference_mode():
        for features,graph,targets in batches(arrays,batch_size):
            prediction=model(features.to(device),graph.to(device))
            loss=acceleration_mse(prediction,targets.to(device))
            total+=float(loss)*targets.numel(); count+=targets.numel()
    return total/count


def run_training(dataset,output,model_config,training_config,device='cpu',*,dry_run=False,corrections=None,initial_checkpoint=None):
    manifest,arrays=prepare_data(dataset,corrections=corrections)
    if device!='cpu' and not (device=='cuda' and torch.cuda.is_available()):
        raise ValueError('Appareil attendu : cpu ou cuda disponible')
    model=AccelerationGAT(model_config,manifest['simulator']['physics']['max_acceleration']).to(device)
    initial_digest = None
    if initial_checkpoint is not None:
        initial = load_model(initial_checkpoint, device=device)
        if initial.metadata() != model.metadata():
            raise ValueError('Modèle initial incompatible avec la configuration')
        model.load_state_dict(initial.state_dict())
        initial_digest = hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest()
    description=dict(dataset=str(Path(dataset).resolve()),
                     dataset_manifest_sha256=hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),
                     model=model.metadata(),training=asdict(training_config),device=device,
                     torch_version=str(torch.__version__),numpy_version=str(np.__version__),
                     graphs={s:len(a[0]) for s,a in arrays.items()},status='dry_run' if dry_run else 'training')
    if initial_digest is not None:
        description['initial_checkpoint_sha256'] = initial_digest
    if corrections is not None:
        description['corrections'] = dict(path=str(Path(corrections).resolve()),
            manifest_sha256=hashlib.sha256((Path(corrections)/'manifest.json').read_bytes()).hexdigest())
    if dry_run:
        return description
    # Ce bloc n'est exécuté que par la commande d'entraînement de l'utilisateur.
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    def save_run():
        (output/'run.json').write_text(json.dumps(description,indent=2,allow_nan=False),encoding='utf-8')
    save_run()
    optimizer=torch.optim.Adam(model.parameters(),lr=training_config.learning_rate)
    generator=torch.Generator().manual_seed(training_config.seed)
    history=[]; best_loss=float('inf'); stale=0; best_path=None
    if initial_checkpoint is not None:
        best_loss = validation_mse(model, arrays['validation'], training_config.batch_size, device)
        best_path = output/'epoch-0000.pt'
        save_model(best_path, model)
        description.update(initial_validation_mse=best_loss, best_epoch=0, best_validation_mse=best_loss)
        save_run()
    for epoch in range(1,training_config.epochs+1):
        model.train(); total,count=0.,0
        for features,graph,targets in batches(arrays['train'],training_config.batch_size,generator=generator,shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            predictions=model(features.to(device),graph.to(device))
            loss=acceleration_mse(predictions,targets.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Perte non finie : entraînement interrompu')
            loss.backward(); optimizer.step()
            total+=float(loss.detach())*targets.numel(); count+=targets.numel()
        score=validation_mse(model,arrays['validation'],training_config.batch_size,device)
        if not math.isfinite(score):
            raise ValueError('Perte validation non finie')
        history.append(dict(epoch=epoch,train_mse=total/count,validation_mse=score))
        if score<best_loss:
            best_loss=score; stale=0
            best_path=output/f'epoch-{epoch:04d}.pt'
            save_model(best_path,model)
            description.update(best_epoch=epoch,best_validation_mse=score)
        else:
            stale+=1
        description['history']=history; save_run()
        print(f'Époque {epoch}: MSE train={total/count:.6f}, validation={score:.6f}',flush=True)
        if stale>=training_config.patience: break
    shutil.copyfile(best_path,output/'best.pt')
    description.update(status='trained',checkpoint='best.pt',epochs_completed=len(history))
    save_run()
    return description


def main(argv=None):
    parser=argparse.ArgumentParser(description='Entraîner le GAT par imitation ; à lancer par l’utilisateur')
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',default='outputs/imitation/training-circle-v1')
    parser.add_argument('--model',default='config/imitation-model.yaml')
    parser.add_argument('--training',default='config/imitation-training.yaml')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dry-run',action='store_true',help='Vérifier les données et paramètres sans entraînement ni écriture')
    parser.add_argument('--corrections', help='Corrections du professeur sur les états GAT train uniquement')
    parser.add_argument('--initial-checkpoint', help='Initialiser les poids depuis une sauvegarde compatible ; nouvel optimiseur')
    args=parser.parse_args(argv)
    result=run_training(args.dataset,args.output,GATConfig.from_yaml(args.model),
                        TrainingConfig.from_yaml(args.training),args.device,dry_run=args.dry_run,
                        corrections=args.corrections,initial_checkpoint=args.initial_checkpoint)
    if args.dry_run: print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    main()
