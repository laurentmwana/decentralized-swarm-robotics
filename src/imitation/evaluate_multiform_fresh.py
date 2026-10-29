"""Test frais figé : professeur, ancien GAT brut et candidat brut, mêmes départs."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import torch

from .data import nonoverlapping_positions
from .local_target_model import load_model,LocalTargetController
from .multiform_data import load_manifest,sha256
from src.reference.metrics import summarize
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import ShapeGains,ShapeMetricConfig,execute_shape,evaluate_shape
from src.reference.shapes import ShapeMission
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig


def evaluate(dataset,corrections,baseline,candidate,output):
    torch.set_num_threads(1)
    dataset,corrections,baseline,candidate,output=map(Path,(dataset,corrections,baseline,candidate,output))
    manifest=load_manifest(dataset);plan=json.loads((corrections/'manifest.json').read_text(encoding='utf-8'))
    run=json.loads((candidate/'run.json').read_text(encoding='utf-8'))
    if run.get('status')!='trained' or run.get('corrections_manifest_sha256')!=sha256(corrections/'manifest.json'):
        raise ValueError('Candidat figé incompatible avec les corrections')
    if sha256(dataset/'manifest.json')!=plan['dataset_manifest_sha256'] or run['dataset_manifest_sha256']!=plan['dataset_manifest_sha256']:
        raise ValueError('Jeu source incompatible')
    reserved=plan['fresh_test'];seeds=reserved['seeds'];shapes=reserved['shapes']
    used={s for values in manifest['collection']['episode_seeds'].values() for s in values}
    if set(seeds)&used or len(set(seeds))!=len(seeds):raise ValueError('Départs réservés non indépendants des données source')
    if set(shapes)!=set(manifest['collection']['shape_splits']['test']):raise ValueError('Formes réservées modifiées')
    base_checkpoint=baseline/'best.pt';checkpoint=(candidate/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(candidate.resolve()) or sha256(base_checkpoint)!=plan['source_model_sha256']:
        raise ValueError('Poids source incompatibles')
    old,new=load_model(base_checkpoint),load_model(checkpoint)
    if old.metadata()!=new.metadata():raise ValueError('Contrats de modèle différents')
    policies=dict(teacher=None,baseline_raw=LocalTargetController(old),candidate_raw=LocalTargetController(new),
                  candidate_guarded=LocalTargetController(new))
    raw=manifest['simulator'];config=AccelerationEpisodeConfig(**{**raw,'physics':DynamicsConfig(**raw['physics'])})
    gains=ShapeGains(**manifest['teacher']['gains']);reconnection=ReconnectionConfig(**manifest['teacher']['reconnection'])
    protocol=ShapeMetricConfig(**manifest['protocol'])
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,kind='fresh_multiform_correction_evaluation',status='incomplete',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),reserved=reserved,
        corrections_manifest_sha256=sha256(corrections/'manifest.json'),baseline_sha256=sha256(base_checkpoint),
        candidate_sha256=sha256(checkpoint),protocol=manifest['protocol'],simulator=raw,
        motion_phases=[0,3],shared_classical_phases=[1,2],reports=[])
    def save():(output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');save()
    for seed in seeds:
        initial=nonoverlapping_positions(config,manifest['collection']['robot_count'],seed,.45,20000)
        for name in shapes:
            mission=ShapeMission(**manifest['missions'][name]);entry=dict(seed=seed,shape=name,metrics={})
            for mode,policy in policies.items():
                result=execute_shape(mission,config,initial,gains=gains,seed=seed,reconnection=reconnection,
                    safety_filter=True,motion_policy=policy,guard_motion=mode in ('teacher','candidate_guarded'))
                filename=f'{mode}-{name}-{seed}.npz';result.episode.save(output/filename)
                entry['metrics'][mode]=evaluate_shape(result,protocol)
            report['reports'].append(entry);save()
            print(f'{name} seed={seed}: '+str({m:v['success'] for m,v in entry['metrics'].items()}),flush=True)
    report.update(status='complete',summary={mode:summarize(r['metrics'][mode] for r in report['reports']) for mode in policies})
    if sha256(checkpoint)!=report['candidate_sha256']:raise ValueError('Poids modifiés pendant le test')
    save();print(json.dumps(report['summary'],indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','corrections','baseline','candidate','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();evaluate(args.dataset,args.corrections,args.baseline,args.candidate,args.output)


if __name__=='__main__':main()
