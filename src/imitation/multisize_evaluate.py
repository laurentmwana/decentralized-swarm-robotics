"""Réserve multi-effectifs : mêmes départs pour professeur, GAT brut et protégé."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np
import torch

from .multisize import PROFILES,controlled_start
from .multiform_data import sha256
from .local_target_model import load_model,LocalTargetController
from src.reference.metrics import summarize
from src.reference.shape_control import execute_shape,evaluate_shape,ShapeGains,ShapeMetricConfig
from src.reference.shapes import named_shape


def evaluate(dataset,run_folder,output):
    torch.set_num_threads(1);dataset,run_folder,output=map(Path,(dataset,run_folder,output))
    manifest=json.loads((dataset/'manifest.json').read_text(encoding='utf-8'))
    run=json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('kind')!='multi_size_local_target_training' or run.get('status')!='trained' or run['dataset_manifest_sha256']!=sha256(dataset/'manifest.json'):
        raise ValueError('Modèle multi-effectifs terminé et jeu source identique requis')
    if manifest['profiles']!={str(n):asdict(p) for n,p in PROFILES.items()}:
        raise ValueError('Les profils physiques ont changé depuis la collecte')
    used=set(manifest['episode_seeds']['train'])|set(manifest['episode_seeds']['validation'])
    if used&set(manifest['episode_seeds']['test']):raise ValueError('Départs test partagés avec train/validation')
    checkpoint=(run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):raise ValueError('Checkpoint extérieur')
    model=load_model(checkpoint)
    if model.metadata()!=run['model']:raise ValueError('Contrat de modèle incompatible')
    controller=LocalTargetController(model)
    output.mkdir(parents=True,exist_ok=False)
    report=dict(schema_version=1,kind='multi_size_paired_evaluation',status='incomplete',
        registered_at_utc=datetime.now(timezone.utc).isoformat(),dataset_manifest_sha256=sha256(dataset/'manifest.json'),
        model_sha256=sha256(checkpoint),domain=manifest['domain'],profiles=manifest['profiles'],
        seeds=manifest['episode_seeds']['test'],shapes=manifest['shape_splits']['test'],modes=['teacher','raw','guarded'],reports=[])
    def save():(output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');save()
    for count,profile in PROFILES.items():
        config=profile.config;gains=ShapeGains(detour_only_on_constraint=True)
        protocol=ShapeMetricConfig(.1,.05,1.,config.horizon*.1)
        for seed in report['seeds']:
            initial=controlled_start(profile,seed)
            for name in report['shapes']:
                shape=named_shape(name,count,width=profile.width);entry=dict(count=count,shape=name,seed=seed,metrics={})
                for mode in report['modes']:
                    result=execute_shape(shape,config,initial,gains=gains,reconnection=profile.reconnection,
                        safety_filter=True,motion_policy=None if mode=='teacher' else controller,guard_motion=mode!='raw',seed=seed)
                    prefix=f'{mode}-{count}-{name}-{seed}'
                    result.episode.save(output/f'{prefix}.npz')
                    np.savez_compressed(output/f'{prefix}-control.npz',phases=result.phases,local_targets=result.local_targets,
                        assignments=result.assignments,safety_infeasible=result.safety_infeasible,safety_interventions=result.safety_interventions,
                        allocation_consensus=result.allocation_consensus,messages_per_step=result.messages_per_step)
                    entry['metrics'][mode]=evaluate_shape(result,protocol)
                report['reports'].append(entry);save()
                print(f'N={count} {name}: '+str({m:r['success'] for m,r in entry['metrics'].items()}),flush=True)
    report.update(status='complete',summary={str(count):{mode:summarize(e['metrics'][mode] for e in report['reports'] if e['count']==count)
        for mode in report['modes']} for count in PROFILES})
    if sha256(checkpoint)!=report['model_sha256']:raise ValueError('Poids modifiés pendant le test')
    save();print(json.dumps(report['summary'],indent=2));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('dataset','run','output'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();evaluate(args.dataset,args.run,args.output)


if __name__=='__main__':main()
