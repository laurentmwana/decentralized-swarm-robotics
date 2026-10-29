"""Comparaison appariée professeur / GAT brut / GAT filtré sur un ensemble réservé."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from .local_target_model import load_model, LocalTargetController
from .local_target_train import batches
from .multiform_data import load_manifest, load_episode, sha256
from .objective import acceleration_mse
from src.reference.metrics import summarize
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import ShapeGains, ShapeMetricConfig, execute_shape, evaluate_shape
from src.reference.shapes import ShapeMission
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig


def evaluate_run(run_folder, dataset, output, *, split='validation', modes=('raw', 'guarded')):
    modes = tuple(modes)
    if split not in ('validation', 'test') or not modes or len(set(modes)) != len(modes) or any(m not in ('raw','guarded') for m in modes):
        raise ValueError('Ensemble validation/test et variantes raw/guarded uniques requis')
    run_folder, dataset, output = map(Path, (run_folder, dataset, output))
    run = json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('kind') != 'multiform_local_target_training' or run.get('status') != 'trained':
        raise ValueError('Entraînement multi-formes terminé requis')
    if sha256(dataset/'manifest.json') != run['dataset_manifest_sha256']:
        raise ValueError('Manifeste différent du jeu entraîné')
    manifest = load_manifest(dataset)
    checkpoint = (run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):
        raise ValueError('Checkpoint extérieur au run')
    model = load_model(checkpoint).eval()
    raw = manifest['simulator']
    config = AccelerationEpisodeConfig(**{**raw, 'physics':DynamicsConfig(**raw['physics'])})
    if model.metadata() != run['model'] or model.max_acceleration != config.physics.max_acceleration:
        raise ValueError('Modèle incompatible avec la campagne')
    gains = ShapeGains(**manifest['teacher']['gains'])
    reconnection = ReconnectionConfig(**manifest['teacher']['reconnection'])
    protocol = ShapeMetricConfig(**manifest['protocol'])
    rows = [e for e in manifest['episodes'] if e['split'] == split and e['duplicate_within_split_of'] is None]
    if not rows: raise ValueError('Aucun épisode unique réservé')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='local_target_paired_evaluation', status='incomplete',
        registered_at_utc=datetime.now(timezone.utc).isoformat(), split=split, modes=['teacher', *modes],
        learned_phases=[0,3], shared_classical_phases=[1,2],
        model_sha256=sha256(checkpoint), dataset_manifest_sha256=run['dataset_manifest_sha256'],
        protocol=manifest['protocol'], teacher=manifest['teacher'], simulator=raw,
        episodes=[dict(shape=e['shape'], seed=e['seed'], path=e['path']) for e in rows], reports=[])
    def save():
        (output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (output/'protocol.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    controller = LocalTargetController(model)
    total_error, components = 0., 0
    for entry in rows:
        arrays = load_episode(dataset, entry, split)
        if np.any(arrays['velocities'][0] != 0):
            raise ValueError('Vitesse initiale non nulle non prise en charge dans ce protocole')
        keep = arrays['learn_mask']
        tensors = tuple(torch.as_tensor(arrays[name][keep], dtype=torch.bool if name == 'adjacency' else torch.float32)
                        for name in ('node_features','relative_neighbors','adjacency','targets'))
        with torch.inference_mode():
            for own, neighbors, mask, targets in batches(tensors, 512):
                loss = acceleration_mse(model(own,neighbors,mask), targets)
                total_error += float(loss)*targets.numel(); components += targets.numel()
        mission = ShapeMission(**manifest['missions'][entry['shape']])
        pair = dict(shape=entry['shape'], seed=entry['seed'], metrics={})
        for mode in ('teacher', *modes):
            result = execute_shape(mission, config, arrays['positions'][0], gains=gains, seed=entry['seed'],
                safety_filter=True, reconnection=reconnection,
                motion_policy=None if mode == 'teacher' else controller, guard_motion=mode != 'raw')
            prefix = f'{mode}-{entry["shape"]}-{entry["seed"]}'
            result.episode.save(output/f'{prefix}.npz')
            np.savez_compressed(output/f'{prefix}-control.npz', phases=result.phases, targets=result.targets,
                local_targets=result.local_targets, assignments=result.assignments, allocation_consensus=result.allocation_consensus,
                safety_infeasible=result.safety_infeasible, safety_interventions=result.safety_interventions)
            pair['metrics'][mode] = evaluate_shape(result, protocol)
        report['reports'].append(pair); save()
    report.update(status='complete', teacher_state_mse=total_error/components,
        summary={mode:{**summarize(r['metrics'][mode] for r in report['reports']),
            'safety_intervention_robot_steps':sum(r['metrics'][mode]['safety_intervention_robot_steps'] for r in report['reports']),
            'motion_safety_intervention_robot_steps':sum(r['metrics'][mode]['motion_safety_intervention_robot_steps'] for r in report['reports']),
            'safety_infeasible_robot_steps':sum(r['metrics'][mode]['safety_infeasible_robot_steps'] for r in report['reports'])}
            for mode in ('teacher',*modes)})
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--modes', choices=['raw','guarded'], nargs='+', default=['raw','guarded'])
    args = parser.parse_args(argv)
    report = evaluate_run(args.run,args.dataset,args.output,split=args.split,modes=args.modes)
    print(json.dumps(report['summary'],indent=2))
    return report


if __name__ == '__main__':
    main()
