"""Campagne continue professeur / GAT protégé, sans entraînement ni remise à zéro."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import numpy as np
import yaml

from .allocation import _integer
from .metrics import summarize
from .reconnection import ReconnectionConfig
from .shape_control import ShapeGains
from .shapes import named_shape
from .transitions import FormationStage, TransitionProtocol, execute_sequence, evaluate_sequence
from src.imitation.data import nonoverlapping_positions
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def duration_steps(seconds, dt):
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0:
        raise ValueError('Durée de mission finie et positive requise')
    steps=round(seconds/dt)
    if steps<1 or abs(steps*dt-seconds)>1e-8:
        raise ValueError('La durée de mission doit être un multiple du pas physique')
    return steps


def evaluate_transitions(config_path, output, *, model_run=None, modes=None):
    raw=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    count=raw.get('robot_count',12); _integer(count,'robot_count',2)
    seeds=list(raw['seeds'])
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError('Graines non vides et uniques requises')
    for seed in seeds: _integer(seed,'seed')
    config=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),
                   communication_radius=raw.get('communication_radius',3.))
    gains=ShapeGains(detour_only_on_constraint=True)
    reconnection=ReconnectionConfig()
    protocol=TransitionProtocol(**raw.get('metrics',{}))
    sequences={}
    if not isinstance(raw['sequences'],dict) or not raw['sequences']:
        raise ValueError('Séquences nommées non vides requises')
    for name,definitions in raw['sequences'].items():
        if not isinstance(name,str) or re.fullmatch(r'[A-Za-z0-9_-]+',name) is None or not definitions:
            raise ValueError('Nom de séquence sûr et liste non vide requis')
        stages=[]
        for definition in definitions:
            item=dict(shape=definition,seconds=raw.get('stage_seconds',40.)) if isinstance(definition,str) else definition
            mission=named_shape(item['shape'],count,width=raw.get('width',6.))
            mission.validate_environment(config,gains.safety_distance)
            stages.append(FormationStage(mission,duration_steps(item['seconds'],config.physics.dt)))
        sequences[name]=tuple(stages)
    modes=tuple(('teacher',) if model_run is None else ('teacher','guarded')) if modes is None else tuple(modes)
    if not modes or len(set(modes))!=len(modes) or any(m not in ('teacher','guarded','raw') for m in modes):
        raise ValueError('Variantes uniques teacher/guarded/raw requises')
    controller=None; model_hash=None
    if any(m!='teacher' for m in modes):
        if model_run is None: raise ValueError('Run entraîné requis pour le GAT')
        from src.imitation.local_target_model import load_model,LocalTargetController
        folder=Path(model_run)
        run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if run.get('kind')!='multiform_local_target_training' or run.get('status')!='trained':
            raise ValueError('Entraînement multi-formes terminé requis')
        checkpoint=(folder/run['checkpoint']).resolve()
        if not checkpoint.is_relative_to(folder.resolve()): raise ValueError('Checkpoint extérieur au run')
        model=load_model(checkpoint)
        if model.metadata()!=run['model'] or model.max_acceleration!=config.physics.max_acceleration:
            raise ValueError('Modèle incompatible avec la campagne de transitions')
        model_hash=_sha(checkpoint); controller=LocalTargetController(model)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    payload=dict(schema_version=1,kind='continuous_transition_campaign',status='incomplete',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),configuration=raw,configuration_sha256=_sha(config_path),
        physics=asdict(config.physics),communication_radius=config.communication_radius,gains=asdict(gains),
        reconnection=asdict(reconnection),metrics=asdict(protocol),model_sha256=model_hash,modes=list(modes),
        stale_replay='one_previous_snapshot_per_visible_sender_at_first_current_allocation_round',
        sequences={name:[dict(shape=s.mission.identifier,steps=s.steps,epoch=f'stage:{i}') for i,s in enumerate(stages)]
                   for name,stages in sequences.items()},reports=[])
    def save():
        (output/'evaluation.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8');save()
    for seed in seeds:
        initial=nonoverlapping_positions(config,count,seed,.45,20000)
        payload.setdefault('initial_states',{})[str(seed)]=initial.tolist();save()
        for name,stages in sequences.items():
            report=dict(sequence=name,seed=seed,modes={},trajectories={})
            for mode in modes:
                result=execute_sequence(stages,config,initial,gains=gains,reconnection=reconnection,safety_filter=True,
                    motion_policy=None if mode=='teacher' else controller,guard_motion=mode!='raw',seed=seed)
                filename=f'{mode}-{name}-{seed}.npz';result.save(output/filename)
                report['modes'][mode]=evaluate_sequence(result,protocol);report['trajectories'][mode]=filename
            payload['reports'].append(report);save()
    summary={}
    for mode in modes:
        reports=[r['modes'][mode] for r in payload['reports']]
        stages=[stage for report in reports for stage in report['stages']]
        summary[mode]=dict(sequences=len(reports),successful_sequences=sum(r['success'] for r in reports),
            sequence_success_rate_percent=100*sum(r['success'] for r in reports)/len(reports),
            stages=summarize(stages),collisions_total=sum(r['collisions_total'] for r in reports),
            boundary_states_preserved=all(r['boundary_states_preserved'] for r in reports),
            stale_packets_discarded=sum(r['stale_packets_discarded'] for r in reports),
            safety_infeasible_robot_steps=sum(s['safety_infeasible_robot_steps'] for s in stages),
            safety_intervention_robot_steps=sum(s['safety_intervention_robot_steps'] for s in stages))
    payload.update(status='complete',summary=summary);save()
    return payload


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/formation-transitions.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--model-run')
    parser.add_argument('--modes',choices=['teacher','guarded','raw'],nargs='+')
    args=parser.parse_args(argv)
    result=evaluate_transitions(args.config,args.output,model_run=args.model_run,modes=args.modes)
    print(json.dumps(result['summary'],indent=2));return result


if __name__=='__main__':main()
