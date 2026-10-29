"""Corrections supervisées sur états GAT train ; aucune sélection des réussites."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .checkpoint import load_model
from .controller import LocalGATController
from .data import load_manifest, load_episode, episode_arrays
from .diagnose import teacher_labels
from src.reference.circle import CircleTeacher, CircleMission, CircleGains
from src.reference.metrics import CircleMetricConfig, evaluate_circle
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_corrections(run_folder, dataset, output):
    run_folder, dataset, output = map(Path, (run_folder, dataset, output))
    run = json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'trained':
        raise ValueError('Entraînement terminé requis')
    digest = sha256(dataset/'manifest.json')
    if digest != run['dataset_manifest_sha256']:
        raise ValueError('Manifeste différent du run')
    manifest = load_manifest(dataset)
    checkpoint = (run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):
        raise ValueError('Sauvegarde extérieure au run')
    model = load_model(checkpoint).eval()
    raw = manifest['simulator']
    config = AccelerationEpisodeConfig(**{**raw, 'physics': DynamicsConfig(**raw['physics'])})
    if model.metadata() != run['model'] or model.max_acceleration != config.physics.max_acceleration:
        raise ValueError('Modèle incompatible')
    teacher = CircleTeacher(CircleMission(**manifest['teacher']['mission']), CircleGains(**manifest['teacher']['gains']))
    teacher.validate_environment(config)
    protocol = CircleMetricConfig(**manifest['protocol'])
    controller = LocalGATController(model, teacher.mission)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, kind='train_on_policy_teacher_labels', status='incomplete',
                  dataset_manifest_sha256=digest, source_model_sha256=sha256(checkpoint),
                  source_model=model.metadata(), simulator=raw, teacher=manifest['teacher'],
                  target=manifest['target'], policy='all_train_rollouts_kept_including_collisions', episodes=[])
    def save():
        (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    save()
    for entry in manifest['episodes']:
        if entry['split'] != 'train':
            continue
        source = load_episode(dataset, entry, 'train')
        result = AccelerationEpisode(config).execute(controller, teacher.mission.robot_count, entry['seed'],
                   source['positions'][0], source['velocities'][0])
        arrays = episode_arrays(result, teacher.mission)
        # Les commandes exécutées restent distinctes des labels du professeur.
        arrays['collector_commands'] = arrays['targets']
        arrays['collector_requests'] = arrays.pop('requested_targets')
        arrays['targets'] = teacher_labels(result, teacher)
        metadata = dict(split='train', seed=entry['seed'], parent_archive_sha256=entry['sha256'],
                        source_model_sha256=report['source_model_sha256'])
        filename = f'train-{entry["seed"]}.npz'
        with (output/filename).open('xb') as stream:
            np.savez_compressed(stream, **arrays, metadata=json.dumps(metadata))
        report['episodes'].append(dict(seed=entry['seed'], split='train', path=filename,
                    sha256=sha256(output/filename), parent_archive_sha256=entry['sha256'],
                    collector_quality=evaluate_circle(result, teacher.mission, protocol),
                    graphs=len(arrays['targets'])))
        save()
    report['status'] = 'complete'
    save()
    return report


def load_corrections(folder, dataset, manifest):
    folder = Path(folder).resolve()
    report = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if report.get('schema_version') != 1 or report.get('kind') != 'train_on_policy_teacher_labels' or report.get('status') != 'complete':
        raise ValueError('Corrections incomplètes ou incompatibles')
    if report['dataset_manifest_sha256'] != sha256(Path(dataset)/'manifest.json'):
        raise ValueError('Corrections issues d’un autre jeu')
    if any(report[k] != manifest[k] for k in ('simulator', 'teacher', 'target')):
        raise ValueError('Contrat physique ou professeur des corrections incompatible')
    parents = {e['seed']: e['sha256'] for e in manifest['episodes'] if e['split'] == 'train'}
    entries = report['episodes']
    if {e['seed'] for e in entries} != set(parents) or len(entries) != len(parents):
        raise ValueError('Une correction par épisode train est requise')
    arrays = []
    for entry in entries:
        if entry.get('split') != 'train' or entry['parent_archive_sha256'] != parents[entry['seed']]:
            raise ValueError('Une correction doit provenir de train')
        path = (folder/entry['path']).resolve()
        if not path.is_relative_to(folder) or sha256(path) != entry['sha256']:
            raise ValueError('Archive de correction corrompue ou extérieure')
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            expected = dict(split='train', seed=entry['seed'], parent_archive_sha256=entry['parent_archive_sha256'],
                            source_model_sha256=report['source_model_sha256'])
            if metadata != expected:
                raise ValueError('Provenance de correction incompatible')
            arrays.append({k: archive[k].copy() for k in ('node_features', 'adjacency', 'targets')})
    return arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = collect_corrections(args.run, args.dataset, args.output)
    print(json.dumps(dict(episodes=len(report['episodes']), graphs=sum(e['graphs'] for e in report['episodes']))))


if __name__ == '__main__':
    main()
