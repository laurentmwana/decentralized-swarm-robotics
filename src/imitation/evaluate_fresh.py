"""Campagne indépendante appariée, sans lecture des démonstrations ni optimisation."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from .checkpoint import load_model
from .controller import LocalGATController
from .data import load_manifest, nonoverlapping_positions
from src.reference.circle import CircleTeacher, CircleMission, CircleGains
from src.reference.metrics import CircleMetricConfig, evaluate_circle, summarize
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig


def evaluate_fresh(run_folder, dataset, output, seeds):
    run_folder, dataset, output = map(Path, (run_folder, dataset, output))
    seeds = list(seeds)
    if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Graines entières positives ou nulles, uniques et non vides requises')
    run = json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'trained':
        raise ValueError('Entraînement terminé requis')
    digest = hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    if digest != run['dataset_manifest_sha256']:
        raise ValueError('Empreinte du manifeste différente du run')
    manifest = load_manifest(dataset)
    used = {e['seed'] for e in manifest['episodes']}
    # Inclure les graines exclues de la collecte revue dans le contrôle de fuite.
    used.update(e['seed'] for e in manifest.get('review', {}).get('excluded', []))
    if used.intersection(seeds):
        raise ValueError('Les graines de la campagne doivent être inédites dans le jeu source')
    checkpoint = (run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):
        raise ValueError('Sauvegarde extérieure au run')
    model = load_model(checkpoint).eval()
    raw = manifest['simulator']
    config = AccelerationEpisodeConfig(**{**raw, 'physics': DynamicsConfig(**raw['physics'])})
    if model.metadata() != run['model'] or model.max_acceleration != config.physics.max_acceleration:
        raise ValueError('Modèle incompatible avec la campagne')
    teacher = CircleTeacher(CircleMission(**manifest['teacher']['mission']), CircleGains(**manifest['teacher']['gains']))
    teacher.validate_environment(config)
    protocol = CircleMetricConfig(**manifest['protocol'])
    controller = LocalGATController(model, teacher.mission)
    output.mkdir(parents=True, exist_ok=False)
    # Écrit AVANT le premier scénario : aucun réglage à partir des résultats.
    registration = dict(status='registered', registered_at_utc=datetime.now(timezone.utc).isoformat(),
                        seeds=seeds, run=str(run_folder.resolve()), dataset_manifest_sha256=digest,
                        model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        protocol=asdict(protocol), simulator=asdict(config), teacher=asdict(teacher),
                        initialization=manifest['collection'], torch_version=str(torch.__version__),
                        policy='paired_initial_states_all_failures_retained_no_model_selection')
    (output/'protocol.json').write_text(json.dumps(registration, indent=2, allow_nan=False), encoding='utf-8')
    reports = []
    for seed in seeds:
        initial = nonoverlapping_positions(config, teacher.mission.robot_count, seed,
                    manifest['collection']['minimum_distance'], manifest['collection']['max_spawn_attempts'])
        pair = {}
        for name, policy in (('teacher', teacher), ('gat', controller)):
            result = AccelerationEpisode(config).execute(policy, teacher.mission.robot_count, seed, initial)
            result.save(output/f'{name}-{seed}.npz')
            pair[name] = evaluate_circle(result, teacher.mission, protocol)
        reports.append(dict(seed=seed, **pair))
    report = dict(**{**registration, 'status': 'complete'}, reports=reports,
                  summary={name: summarize(r[name] for r in reports) for name in ('teacher', 'gat')})
    (output/'evaluation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args(argv)
    result = evaluate_fresh(args.run, args.dataset, args.output, args.seeds)
    print(json.dumps(result['summary'], indent=2))
    return result


if __name__ == '__main__':
    main()
