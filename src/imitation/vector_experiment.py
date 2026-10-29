"""Expérience structurée : professeur local, augmentation, contraintes à l’apprentissage."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import torch
from torch.utils.data import Dataset,ConcatDataset,DataLoader

from .vector_gat import VectorGAT,VectorConfig,save_vector,load_vector,barrier_penalty
from .multisize_train import LocalSamples,collate_local
from .multisize import PROFILES,controlled_start
from .multiform_data import sha256
from .local_target_model import LocalTargetController
from src.reference.safety import filter_command
from src.reference.shape_control import ShapeGains,assigned_command,execute_shape,evaluate_shape,ShapeMetricConfig
from src.reference.shapes import named_shape
from src.reference.metrics import summarize
from src.reference.obstacles import wall_events
from src.simulator.acceleration_episode import LocalObservation


class Augmented(Dataset):
    def __init__(self,folder,split):
        with np.load(Path(folder)/f'{split}.npz',allow_pickle=False) as data:
            self.own=data['own'];self.edges=data['edges'];self.offsets=data['offsets'];self.labels=data['labels']
    def __len__(self):return len(self.own)
    def __getitem__(self,index):
        a,b=self.offsets[index:index+2]
        return self.own[index],self.edges[a:b],self.labels[index]


def augment(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);gains=ShapeGains(detour_only_on_constraint=True)
    report=dict(kind='vector_gat_local_augmentation',status='collecting',created_at_utc=datetime.now(timezone.utc).isoformat(),
        source='local_teacher_projection_training_labels_only',train_seed=8400,validation_seed=8401,
        nominal_fitting=dict(seed=8402,steps=1200,learning_rate=.03),
        model=dict(hidden_dim=48,init_seed=42),training=dict(epochs=80,patience=12,batch_size=512,learning_rate=.001,seed=42,
            max_samples_per_count=6000,barrier_weight=20.,risk_weight=4.),
        final_reserve=dict(seeds=[8500,8501],counts=[12,24,64],shapes=['geometries_rectangle','letters_C','numbers_3'],
            thresholds=dict(error=.1,speed=.05,hold_seconds=1.),domain='controlled_near_rendezvous',modes=['teacher','raw']),splits={})
    def save():(output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for split,size,seed in [('train',12000,8400),('validation',3000,8401)]:
        rng=np.random.default_rng(seed);own=[];edges=[];offsets=[0];labels=[];attempts=0;infeasible=0
        while len(own)<size:
            attempts+=1;count=int(rng.choice([12,24,64]));k=int(rng.choice([0,1,2,4,8],p=[.1,.45,.25,.15,.05]))
            theta=rng.uniform(-np.pi,np.pi,size=k);d=rng.uniform(.45,2.9,size=k)
            close=rng.random(k)<.65;d[close]=rng.uniform(.45,1.,size=close.sum())
            relative=np.column_stack((d*np.cos(theta),d*np.sin(theta)))
            if k>1:
                distances=np.linalg.norm(relative[:,None]-relative[None,:],axis=-1);np.fill_diagonal(distances,np.inf)
                if distances.min()<.3:continue
            velocity=rng.uniform(-1.,1.,size=2);velocity/=max(1.,np.linalg.norm(velocity))
            other=rng.uniform(-1.,1.,size=(k,2));other/=np.maximum(1.,np.linalg.norm(other,axis=-1,keepdims=True))
            w=other-velocity;goal=rng.normal(size=2);goal/=max(np.linalg.norm(goal),1e-6)
            goal*=rng.uniform(0.,1.) if rng.random()<.5 else rng.uniform(1.,28.)
            observation=LocalObservation(np.zeros(2),velocity,relative,w,0.)
            label,feasible=filter_command(observation,assigned_command(observation,goal,gains),gains,2.)
            if not feasible:infeasible+=1;continue
            own.append(np.r_[0.,0.,velocity,goal,int(rng.choice([0,3])),count]);labels.append(label)
            edges.extend(np.column_stack((relative,w)));offsets.append(len(edges))
        path=output/f'{split}.npz'
        np.savez_compressed(path,own=np.asarray(own,np.float32),edges=np.asarray(edges,np.float32).reshape(-1,4),
                            offsets=np.asarray(offsets,np.int64),labels=np.asarray(labels,np.float32))
        report['splits'][split]=dict(samples=size,attempts=attempts,infeasible_labels=infeasible,sha256=sha256(path))
        save();print(f'{split}: {size} labels locaux, {infeasible} états infaisables exclus',flush=True)
    report['status']='complete';save();return report


def train(dataset,augmentation,output,*,conservative=False,source_run=None):
    torch.set_num_threads(1);dataset,augmentation,output=map(Path,(dataset,augmentation,output))
    manifest=json.loads((augmentation/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='complete':raise ValueError('Augmentation incomplète')
    cfg=dict(manifest['training']);cfg['safety_distance']=.4
    if conservative:cfg.update(epochs=60,learning_rate=.0003,barrier_weight=200.,safety_distance=.5)
    torch.manual_seed(cfg['seed']);model=VectorGAT(VectorConfig(**manifest['model'])) if source_run is None else load_vector(Path(source_run)/'best.pt')
    output.mkdir(parents=True,exist_ok=False)
    # Régression sur exemples isolés : les gains sont appris et archivés, pas codés à 6 et 4.
    rng=np.random.default_rng(manifest['nominal_fitting']['seed']);own=np.zeros((8192,8),np.float32)
    own[:,4:6]=rng.uniform(-.6,.6,(8192,2));own[:,2:4]=rng.uniform(-.6,.6,(8192,2))
    targets=6.*own[:,4:6]-4.*own[:,2:4];targets*=np.minimum(1.,2./np.maximum(np.linalg.norm(targets,axis=-1,keepdims=True),1e-9))
    x=torch.from_numpy(own);y=torch.from_numpy(targets)
    optimizer=torch.optim.Adam([model.log_kp,model.log_kd],lr=.03)
    for _ in range(manifest['nominal_fitting']['steps'] if source_run is None else 0):
        optimizer.zero_grad(set_to_none=True);raw=model.nominal(x)
        prediction=raw*(2./torch.linalg.vector_norm(raw,dim=-1,keepdim=True).clamp_min(2.))
        loss=(prediction-y).square().mean();loss.backward();optimizer.step()
    model.log_kp.requires_grad_(False);model.log_kd.requires_grad_(False)
    generator=torch.Generator().manual_seed(cfg['seed']);loaders={};selection={}
    for split in ('train','validation'):
        if sha256(augmentation/f'{split}.npz')!=manifest['splits'][split]['sha256']:raise ValueError('Augmentation modifiée')
        original=LocalSamples(dataset,split,cfg['seed'],cfg['max_samples_per_count'])
        data=ConcatDataset([original,Augmented(augmentation,split)])
        loaders[split]=DataLoader(data,batch_size=cfg['batch_size'],shuffle=split=='train',generator=generator,num_workers=0,collate_fn=collate_local)
        selection[split]=dict(source=original.selection,total=len(data))
    report=dict(kind='vector_gat_training',status='training',model=model.metadata(),dataset_manifest_sha256=sha256(dataset/'manifest.json'),
        augmentation_manifest_sha256=sha256(augmentation/'manifest.json'),training=cfg,selection=selection,
        learned_nominal_gains=dict(kp=float(model.log_kp.exp()),kd=float(model.log_kd.exp())),history=[])
    if source_run is not None:report['source_model_sha256']=sha256(Path(source_run)/'best.pt')
    def save():(output/'run.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save();optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'])
    best=float('inf');stale=0;best_path=None
    for epoch in range(1,cfg['epochs']+1):
        scores={}
        for split,loader in loaders.items():
            model.train(split=='train');loss_sum=mass=0.
            with torch.set_grad_enabled(split=='train'):
                for own,neighbors,mask,target in loader:
                    if split=='train':optimizer.zero_grad(set_to_none=True)
                    prediction=model(own,neighbors,mask)
                    close=(mask&(torch.linalg.vector_norm(neighbors[...,:2],dim=-1)<1.)).any(-1)
                    weights=torch.where(close,cfg['risk_weight'],1.)
                    mse=((prediction-target).square().mean(-1)*weights).sum()/weights.sum()
                    loss=mse+cfg['barrier_weight']*barrier_penalty(prediction,neighbors,mask,cfg['safety_distance'])
                    if not torch.isfinite(loss):raise ValueError('Perte non finie')
                    if split=='train':loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    loss_sum+=float(loss.detach())*len(own);mass+=len(own)
            scores[split]=loss_sum/mass
        report['history'].append(dict(epoch=epoch,train_loss=scores['train'],validation_loss=scores['validation']))
        if scores['validation']<best:
            best=scores['validation'];stale=0;best_path=output/f'epoch-{epoch:04d}.pt';save_vector(best_path,model)
            report.update(best_epoch=epoch,best_validation_loss=best)
        else:stale+=1
        save();print(f'Epoch {epoch}: train={scores["train"]:.5f}, validation={scores["validation"]:.5f}',flush=True)
        if stale>=cfg['patience']:break
    shutil.copyfile(best_path,output/'best.pt');report.update(status='trained',checkpoint='best.pt',epochs_completed=len(report['history']));save();return report


def rollout(run,output,*,validation=False,canary=False,remaining=False):
    if (canary or remaining) and not validation:raise ValueError('Le test final ne peut pas être réduit à un sous-ensemble')
    torch.set_num_threads(1);run,output=map(Path,(run,output));model=load_vector(run/'best.pt');policy=LocalTargetController(model)
    shapes=['geometries_triangle','letters_B','numbers_7'] if validation else ['geometries_rectangle','letters_C','numbers_3']
    seeds=[8200] if validation else [8500,8501]
    if canary:shapes=shapes[:1]
    if remaining:shapes=shapes[1:]
    output.mkdir(parents=True,exist_ok=False)
    report=dict(kind='vector_gat_closed_loop',status='incomplete',split='validation' if validation else 'fresh_test',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),model_sha256=sha256(run/'best.pt'),seeds=seeds,shapes=shapes,reports=[])
    def save():(output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');save()
    for count,profile in PROFILES.items():
        config=profile.config;gains=ShapeGains(detour_only_on_constraint=True)
        for seed in seeds:
            for name in shapes:
                result=execute_shape(named_shape(name,count,width=profile.width),config,controlled_start(profile,seed),
                    gains=gains,reconnection=profile.reconnection,safety_filter=True,motion_policy=policy,guard_motion=False,seed=seed)
                result.episode.save(output/f'raw-{count}-{name}-{seed}.npz')
                np.savez_compressed(output/f'raw-{count}-{name}-{seed}-control.npz',phases=result.phases,
                    local_targets=result.local_targets,assignments=result.assignments,
                    allocation_consensus=result.allocation_consensus,safety_interventions=result.safety_interventions,
                    safety_infeasible=result.safety_infeasible,messages_per_step=result.messages_per_step)
                metrics=evaluate_shape(result,ShapeMetricConfig(.1,.05,1.,config.horizon*.1))
                walls=wall_events(result.episode.positions,config.physics.arena_size,config.physics.robot_radius)
                metrics.update(robot_collisions=metrics['collisions'],wall_collisions=walls)
                metrics['collisions']+=walls
                if walls:metrics.update(success=False,stabilization_time_s=None)
                entry=dict(count=count,shape=name,seed=seed,metrics=metrics)
                if not validation:
                    teacher=execute_shape(named_shape(name,count,width=profile.width),config,controlled_start(profile,seed),
                        gains=gains,reconnection=profile.reconnection,safety_filter=True,guard_motion=True,seed=seed)
                    teacher.episode.save(output/f'teacher-{count}-{name}-{seed}.npz')
                    tm=evaluate_shape(teacher,ShapeMetricConfig(.1,.05,1.,config.horizon*.1))
                    tw=wall_events(teacher.episode.positions,config.physics.arena_size,config.physics.robot_radius)
                    tm.update(robot_collisions=tm['collisions'],wall_collisions=tw);tm['collisions']+=tw
                    if tw:tm.update(success=False,stabilization_time_s=None)
                    entry['teacher_metrics']=tm
                report['reports'].append(entry);save()
                print(f'N={count} {name} {seed}: success={metrics["success"]} collisions={metrics["collisions"]} error={metrics["formation_error_m"]:.4f}',flush=True)
    report.update(status='complete',summary={str(n):summarize(r['metrics'] for r in report['reports'] if r['count']==n) for n in PROFILES})
    if not validation:report['teacher_summary']={str(n):summarize(r['teacher_metrics'] for r in report['reports'] if r['count']==n) for n in PROFILES}
    if sha256(run/'best.pt')!=report['model_sha256']:raise ValueError('Poids modifiés pendant la campagne')
    save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['augment','train','rollout'])
    parser.add_argument('--output',required=True);parser.add_argument('--dataset');parser.add_argument('--augmentation');parser.add_argument('--run')
    parser.add_argument('--validation',action='store_true');parser.add_argument('--canary',action='store_true');parser.add_argument('--remaining',action='store_true')
    parser.add_argument('--conservative',action='store_true');parser.add_argument('--source-run');args=parser.parse_args()
    if args.action=='augment':augment(args.output)
    elif args.action=='train':train(args.dataset,args.augmentation,args.output,conservative=args.conservative,source_run=args.source_run)
    else:rollout(args.run,args.output,validation=args.validation,canary=args.canary,remaining=args.remaining)


if __name__=='__main__':main()
