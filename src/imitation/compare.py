"""Comparer GAT appris et professeur sur les mêmes épisodes de test réservés."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.reference.circle import CircleTeacher, CircleMission, CircleGains
from src.reference.metrics import CircleMetricConfig, evaluate_circle, summarize
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig
from .checkpoint import load_model
from .controller import LocalGATController
from .data import load_episode, load_manifest
from .objective import acceleration_mse


def compare_run(run_folder,output,*,dataset=None):
    run_folder=Path(run_folder)
    run=json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('status')!='trained':
        raise ValueError('Un entraînement terminé par l’utilisateur est requis')
    dataset=Path(run['dataset'] if dataset is None else dataset)
    if hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()!=run['dataset_manifest_sha256']:
        raise ValueError('Le manifeste du jeu a changé depuis l’entraînement')
    manifest=load_manifest(dataset)
    raw=manifest['simulator']
    config=AccelerationEpisodeConfig(**{**raw,'physics':DynamicsConfig(**raw['physics'])})
    teacher=CircleTeacher(CircleMission(**manifest['teacher']['mission']),CircleGains(**manifest['teacher']['gains']))
    teacher.validate_environment(config)
    protocol=CircleMetricConfig(**manifest['protocol'])
    checkpoint=(run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):
        raise ValueError('Chemin de sauvegarde extérieur au dossier d’entraînement')
    model=load_model(checkpoint).eval()
    if model.metadata()!=run['model'] or model.max_acceleration!=config.physics.max_acceleration:
        raise ValueError('Modèle incompatible avec les paramètres de la campagne')
    controller=LocalGATController(model,teacher.mission)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    reports=[]; squared_error=0.; components=0
    for entry in manifest['episodes']:
        if entry['split']!='test': continue
        arrays=load_episode(dataset,entry,'test')
        with torch.inference_mode():
            prediction=model(torch.tensor(arrays['node_features'],dtype=torch.float32),torch.tensor(arrays['adjacency']))
            targets=torch.tensor(arrays['targets'],dtype=torch.float32)
            squared_error+=float(acceleration_mse(prediction,targets))*targets.numel()
            components+=targets.numel()
        initial,velocity=arrays['positions'][0],arrays['velocities'][0]
        pair={}
        for name,policy in (('teacher',teacher),('gat',controller)):
            result=AccelerationEpisode(config).execute(policy,teacher.mission.robot_count,entry['seed'],initial,velocity)
            result.save(output/f'{name}-{entry["seed"]}.npz')
            pair[name]=evaluate_circle(result,teacher.mission,protocol)
        reports.append(dict(seed=entry['seed'],**pair))
    payload=dict(run=str(run_folder.resolve()),dataset=str(dataset.resolve()),model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 dataset_manifest_sha256=run['dataset_manifest_sha256'],protocol=manifest['protocol'],
                 test_teacher_state_mse=squared_error/components,reports=reports,
                 summary={name:summarize([r[name] for r in reports]) for name in ('teacher','gat')})
    (output/'comparison.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description='Comparer le GAT entraîné et le professeur sur le test réservé')
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',default='outputs/imitation/comparison-circle-v1')
    parser.add_argument('--dataset',help='Emplacement actuel du jeu après déplacement ; empreinte du manifeste vérifiée')
    args=parser.parse_args(argv)
    result=compare_run(args.run,args.output,dataset=args.dataset)
    print(json.dumps(result['summary'],indent=2))
    return result


if __name__=='__main__':
    main()
