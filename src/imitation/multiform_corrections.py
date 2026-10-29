"""Corrections locales sur les états du GAT brut, séparées de l’ancien jeu."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .local_target_model import load_model,LocalTargetController
from .multiform_data import load_manifest,load_episode,episode_arrays,sha256
from src.reference.reconnection import ReconnectionConfig
from src.reference.safety import filter_command
from src.reference.shape_control import ShapeGains,ShapeMetricConfig,assigned_command,execute_shape,evaluate_shape
from src.reference.shapes import ShapeMission
from src.simulator.acceleration_episode import AccelerationEpisodeConfig,LocalObservation
from src.simulator.dynamics import DynamicsConfig


def risk_weights(relative,adjacency):
    """Poids 8 si voisin <1 m ou rapprochement prédit <0,6 m sous une seconde."""
    relative=np.asarray(relative);adjacency=np.asarray(adjacency,dtype=bool)
    r,w=relative[...,:2],relative[...,2:]
    distance=np.linalg.norm(r,axis=-1)
    time=np.clip(-(r*w).sum(axis=-1)/np.maximum((w*w).sum(axis=-1),1e-12),0,1)
    future=np.linalg.norm(r+time[...,None]*w,axis=-1)
    risk=(adjacency&((distance<1.)|(future<.6))).any(axis=-1)
    return np.where(risk,8.,1.).astype(np.float32)


def teacher_labels(arrays,gains,maximum):
    features=arrays['node_features'];labels=np.zeros(features.shape[:2]+(2,));feasible=np.zeros(features.shape[:2],dtype=bool)
    for step,robot in zip(*np.where(arrays['learn_mask'])):
        own=features[step,robot];visible=arrays['adjacency'][step,robot]
        edges=arrays['relative_neighbors'][step,robot,visible]
        obs=LocalObservation(own[:2].copy(),own[2:4].copy(),edges[:,:2].copy(),edges[:,2:].copy(),float(arrays['times'][step]))
        labels[step,robot],feasible[step,robot]=filter_command(obs,assigned_command(obs,own[4:6],gains),gains,maximum)
    return labels,feasible


def collect(run_folder,dataset,output):
    torch.set_num_threads(1)
    run_folder,dataset,output=map(Path,(run_folder,dataset,output))
    manifest=load_manifest(dataset);run=json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('status')!='trained' or run['dataset_manifest_sha256']!=sha256(dataset/'manifest.json'):
        raise ValueError('Modèle et jeu source incompatibles')
    checkpoint=(run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):raise ValueError('Checkpoint extérieur')
    model=load_model(checkpoint);controller=LocalTargetController(model)
    raw=manifest['simulator'];config=AccelerationEpisodeConfig(**{**raw,'physics':DynamicsConfig(**raw['physics'])})
    if model.metadata()!=run['model'] or model.max_acceleration!=config.physics.max_acceleration:
        raise ValueError('Contrat du modèle incompatible')
    gains=ShapeGains(**manifest['teacher']['gains']);reconnection=ReconnectionConfig(**manifest['teacher']['reconnection'])
    protocol=ShapeMetricConfig(**manifest['protocol'])
    # First seed in each partition, all unique geometries. No test archive loaded.
    seeds={s:min(manifest['collection']['episode_seeds'][s]) for s in ('train','validation')}
    rows=[e for e in manifest['episodes'] if e['split'] in seeds and e['seed']==seeds[e['split']] and e['duplicate_within_split_of'] is None]
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,kind='multiform_on_policy_corrections',status='incomplete',
        dataset_manifest_sha256=sha256(dataset/'manifest.json'),source_model_sha256=sha256(checkpoint),
        source_model=model.metadata(),selection='first_seed_per_train_validation_unique_geometry',
        risk=dict(close_distance_m=1.,predicted_distance_m=.6,prediction_seconds=1.,weight=8.),
        training=dict(epochs=30,patience=8,learning_rate=.0003,batch_size=1024,seed=42,correction_multiplier=3.),
        fresh_test=dict(seeds=[7800,7801],shapes=manifest['collection']['shape_splits']['test'],
            modes=['teacher','raw','guarded'],rule='one_final_evaluation_after_weights_frozen'),episodes=[])
    def save():(output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    save()
    for entry in rows:
        arrays=load_episode(dataset,entry,entry['split'])
        result=execute_shape(ShapeMission(**manifest['missions'][entry['shape']]),config,arrays['positions'][0],
            gains=gains,seed=entry['seed'],reconnection=reconnection,safety_filter=True,motion_policy=controller,guard_motion=False)
        corrected=episode_arrays(result);labels,feasible=teacher_labels(corrected,gains,config.physics.max_acceleration)
        keep=corrected['learn_mask']&feasible
        weights=risk_weights(corrected['relative_neighbors'],corrected['adjacency'])
        filename=f'{entry["split"]}-{entry["shape"]}-{entry["seed"]}.npz'
        with (output/filename).open('xb') as stream:
            np.savez_compressed(stream,**corrected,teacher_labels=labels,label_feasible=feasible,
                                training_mask=keep,risk_weights=weights)
        report['episodes'].append(dict(split=entry['split'],shape=entry['shape'],seed=entry['seed'],path=filename,
            sha256=sha256(output/filename),parent_sha256=entry['sha256'],samples=int(keep.sum()),
            infeasible_labels=int((corrected['learn_mask']&~feasible).sum()),
            critical_samples=int(((weights>1)&keep).sum()),collector_metrics=evaluate_shape(result,protocol)))
        save();print(f'{entry["split"]} {entry["shape"]}: corrections={keep.sum()}',flush=True)
    report['status']='complete';save();return report


def load_corrections(folder,dataset,split):
    if split not in ('train','validation'):raise ValueError('Corrections train/validation uniquement')
    folder=Path(folder).resolve();report=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if report.get('kind')!='multiform_on_policy_corrections' or report.get('status')!='complete':
        raise ValueError('Corrections complètes requises')
    if report['dataset_manifest_sha256']!=sha256(Path(dataset)/'manifest.json'):
        raise ValueError('Jeu source des corrections différent')
    source=load_manifest(dataset);pieces=[]
    for entry in report['episodes']:
        if entry['split']!=split:continue
        if entry['shape'] not in source['collection']['shape_splits'][split]:raise ValueError('Forme dans une mauvaise partition')
        path=(folder/entry['path']).resolve()
        if not path.is_relative_to(folder) or sha256(path)!=entry['sha256']:raise ValueError('Archive corrective corrompue ou extérieure')
        with np.load(path,allow_pickle=False) as data:
            keep=data['training_mask']
            if not np.array_equal(keep,data['learn_mask']&data['label_feasible']):raise ValueError('Masque correctif invalide')
            if not np.isfinite(data['teacher_labels']).all():raise ValueError('Labels correctifs non finis')
            pieces.append(tuple(data[name][keep] for name in ('node_features','relative_neighbors','adjacency','teacher_labels','risk_weights')))
    if not pieces:raise ValueError('Partition corrective vide')
    return tuple(np.concatenate([piece[i] for piece in pieces]) for i in range(5))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();collect(args.run,args.dataset,args.output)


if __name__=='__main__':main()
