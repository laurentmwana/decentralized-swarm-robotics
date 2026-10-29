"""Campagnes appariées de navigation, sans modification des poids."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import json
from pathlib import Path
import re

import yaml

from .evaluate_transitions import duration_steps,_sha
from .navigation import MovingReference,NavigationProtocol,initial_positions,execute_navigation,evaluate_navigation
from .obstacles import DiskWorld,DiskObstacle
from .shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def evaluate_campaign(config_path,output,*,model_run=None):
    raw=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    config=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),
                   communication_radius=raw.get('communication_radius',3.))
    protocol=NavigationProtocol(**raw.get('metrics',{}))
    controller=None;model_hash=None
    if model_run is not None:
        from src.imitation.local_target_model import load_model,LocalTargetController
        folder=Path(model_run);run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if run.get('kind')!='multiform_local_target_training' or run.get('status')!='trained':
            raise ValueError('Entraînement multi-formes terminé requis')
        checkpoint=(folder/run['checkpoint']).resolve()
        if not checkpoint.is_relative_to(folder.resolve()):raise ValueError('Checkpoint extérieur au run')
        model=load_model(checkpoint)
        if model.metadata()!=run['model'] or model.max_acceleration!=config.physics.max_acceleration:
            raise ValueError('Modèle incompatible')
        controller=LocalTargetController(model);model_hash=_sha(checkpoint)
    seeds=raw['seeds']
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Graines uniques non vides requises')
    scenarios=[]
    for name,item in raw['scenarios'].items():
        if re.fullmatch('[A-Za-z0-9_-]+',name) is None:raise ValueError('Nom sûr requis')
        shape=named_shape(item['shape'],raw.get('robot_count',12),width=raw.get('width',6.))
        reference=MovingReference(item['points'],item['durations'])
        world=DiskWorld(DiskObstacle(tuple(o['center']),o['radius']) for o in item.get('obstacles',[]))
        steps=duration_steps(item['navigation_seconds'],config.physics.dt)
        scenarios.append((name,item.get('group','nominal'),shape,reference,world,steps))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    payload=dict(schema_version=1,kind='formation_navigation_campaign',status='incomplete',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),configuration=raw,
        configuration_sha256=_sha(config_path),physics=asdict(config.physics),model_sha256=model_hash,
        metrics=asdict(protocol),policy_frame='common_translating_frame_plus_classical_reference_acceleration',
        obstacle_policy='local_disk_sensor_to_classical_joint_safety_filter_only',reports=[])
    def save():
        (output/'evaluation.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8');save()
    preparation_steps=duration_steps(raw.get('preparation_seconds',40.),config.physics.dt)
    for name,group,shape,reference,world,steps in scenarios:
        for seed in seeds:
            initial=initial_positions(config,shape.robot_count,seed,world,clearance=protocol.obstacle_clearance)
            report=dict(scenario=name,group=group,seed=seed,initial_positions=initial.tolist(),modes={},trajectories={})
            for mode,policy in [('teacher',None)]+([] if controller is None else [('guarded',controller)]):
                result=execute_navigation(shape,reference,world,config,initial,navigation_steps=steps,
                    preparation_steps=preparation_steps,motion_policy=policy,protocol=protocol,seed=seed)
                filename=f'{mode}-{name}-{seed}.npz';result.save(output/filename)
                report['modes'][mode]=evaluate_navigation(result);report['trajectories'][mode]=filename
            payload['reports'].append(report);save()
            print(f'{name} seed={seed}: '+str({m:r['success'] for m,r in report['modes'].items()}),flush=True)
    summary={}
    for mode in payload['reports'][0]['modes']:
        summary[mode]={}
        for group in ['all']+sorted({r['group'] for r in payload['reports']}):
            reports=[r['modes'][mode] for r in payload['reports'] if group=='all' or r['group']==group]
            summary[mode][group]=dict(episodes=len(reports),successes=sum(r['success'] for r in reports),
                collisions=sum(r['collisions'] for r in reports),
                max_formation_error_m=max(r['max_navigation_formation_error_m'] for r in reports),
                max_destination_error_m=max(r['destination_error_m'] for r in reports))
    payload.update(status='complete',summary=summary);save();return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/formation-navigation.yaml')
    parser.add_argument('--output',required=True);parser.add_argument('--model-run')
    args=parser.parse_args()
    result=evaluate_campaign(args.config,args.output,model_run=args.model_run)
    print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
