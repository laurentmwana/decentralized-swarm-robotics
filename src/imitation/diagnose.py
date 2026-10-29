"""Diagnostic sans optimisation, limité aux ensembles train et validation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model
from .controller import LocalGATController
from .data import load_episode, load_manifest, episode_arrays
from src.reference.circle import CircleTeacher, CircleMission, CircleGains
from src.reference.metrics import CircleMetricConfig, evaluate_circle
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig, LocalObservation
from src.simulator.dynamics import DynamicsConfig, _limit


def predict(model, arrays):
    with torch.inference_mode():
        return model(torch.tensor(arrays['node_features'], dtype=torch.float32),
                     torch.tensor(arrays['adjacency'])).numpy()


def sample_statistics(arrays, prediction, safety_distance):
    features = arrays['node_features']
    offset = features[..., :2] - features[..., 4:6]
    radius = np.linalg.norm(offset, axis=-1)
    radial = offset / np.maximum(radius[..., None], 1e-12)
    # Même convention au centre exact que le professeur.
    radial[radius <= 1e-12] = (1., 0.)
    error = prediction - arrays['targets']
    positions = features[..., :2]
    distances = np.linalg.norm(positions[:, :, None] - positions[:, None, :], axis=-1)
    indices = np.arange(positions.shape[1])
    distances[:, indices, indices] = np.inf
    nearest = distances.min(axis=-1)
    return dict(mse=(error**2).mean(axis=-1),
                radial_squared=(error*radial).sum(axis=-1)**2,
                tangential_squared=(error*np.stack((-radial[..., 1], radial[..., 0]), axis=-1)).sum(axis=-1)**2,
                target_norm=np.linalg.norm(arrays['targets'], axis=-1),
                degree=arrays['adjacency'].sum(axis=-1), near=nearest < safety_distance,
                radius_error=np.abs(radius-features[..., 6]),
                speed=np.linalg.norm(features[..., 2:4], axis=-1))


def aggregate_statistics(records, dt):
    keys = records[0].keys()
    merged = {k: np.concatenate([r[k].reshape(-1) for r in records]) for k in keys}
    time = np.concatenate([np.broadcast_to(np.arange(r['mse'].shape[0])[:, None]*dt,
                                           r['mse'].shape).reshape(-1) for r in records])
    def summary(mask):
        count = int(mask.sum())
        return dict(samples=count, **{k: float(merged[k][mask].mean()) if count else None
                    for k in ('mse', 'radial_squared', 'tangential_squared', 'target_norm', 'radius_error', 'speed')})
    return dict(overall=summary(np.ones(len(time), dtype=bool)),
                by_time={label: summary(mask) for label, mask in
                         [('0_to_2_s', time < 2), ('2_to_5_s', (time >= 2)&(time < 5)), ('5_s_plus', time >= 5)]},
                by_degree={str(int(d)): summary(merged['degree'] == d) for d in np.unique(merged['degree'])},
                by_safety={label: summary(mask) for label, mask in
                           [('inside_safety', merged['near']), ('outside_safety', ~merged['near'])]})


def teacher_labels(result, teacher):
    requested = []
    for p, v, graph, time in zip(result.positions[:-1], result.velocities[:-1], result.adjacency[:-1], result.times[:-1]):
        commands = [teacher(LocalObservation(p[i], v[i], p[graph[i]]-p[i], v[graph[i]]-v[i], time))
                    for i in range(len(p))]
        requested.append(_limit(np.asarray(commands), result.config.physics.max_acceleration))
    return np.asarray(requested)


def diagnose_run(run_folder, dataset, output):
    run_folder, dataset, output = map(Path, (run_folder, dataset, output))
    run = json.loads((run_folder/'run.json').read_text(encoding='utf-8'))
    if run.get('status') != 'trained':
        raise ValueError('Entraînement terminé requis')
    digest = hashlib.sha256((dataset/'manifest.json').read_bytes()).hexdigest()
    if digest != run['dataset_manifest_sha256']:
        raise ValueError('Empreinte du manifeste différente du run')
    manifest = load_manifest(dataset)
    checkpoint = (run_folder/run['checkpoint']).resolve()
    if not checkpoint.is_relative_to(run_folder.resolve()):
        raise ValueError('Sauvegarde extérieure au run')
    model = load_model(checkpoint).eval()
    if model.metadata() != run['model']:
        raise ValueError('Métadonnées du modèle incompatibles')
    raw = manifest['simulator']
    config = AccelerationEpisodeConfig(**{**raw, 'physics': DynamicsConfig(**raw['physics'])})
    teacher = CircleTeacher(CircleMission(**manifest['teacher']['mission']), CircleGains(**manifest['teacher']['gains']))
    teacher.validate_environment(config)
    protocol = CircleMetricConfig(**manifest['protocol'])
    output.mkdir(parents=True, exist_ok=False)
    records = {s: [] for s in ('train', 'validation')}
    rollouts, shifted = [], []
    for entry in manifest['episodes']:
        split = entry['split']
        if split not in records:
            continue
        arrays = load_episode(dataset, entry, split)
        records[split].append(sample_statistics(arrays, predict(model, arrays), teacher.gains.safety_distance))
        if split != 'validation':
            continue
        result = AccelerationEpisode(config).execute(LocalGATController(model, teacher.mission),
                 teacher.mission.robot_count, entry['seed'], arrays['positions'][0], arrays['velocities'][0])
        result.save(output/f'validation-gat-{entry["seed"]}.npz')
        visited = episode_arrays(result, teacher.mission)
        visited['targets'] = teacher_labels(result, teacher)
        shifted.append(sample_statistics(visited, predict(model, visited), teacher.gains.safety_distance))
        drift = np.sqrt(np.mean(np.sum((result.positions-arrays['positions'])**2, axis=-1), axis=1))
        rollouts.append(dict(seed=entry['seed'], metrics=evaluate_circle(result, teacher.mission, protocol),
                             teacher_metrics=entry['quality'], position_rmse_final_m=float(drift[-1]),
                             position_rmse_by_second_m={str(t): float(drift[min(round(t/config.physics.dt), len(drift)-1)])
                                                       for t in range(int(result.times[-1])+1)}))
    payload = dict(schema_version=1, dataset_manifest_sha256=digest,
                   model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                   splits_loaded=['train', 'validation'], protocol=manifest['protocol'],
                   teacher_states={s: aggregate_statistics(r, config.physics.dt) for s, r in records.items()},
                   validation_gat_states=aggregate_statistics(shifted, config.physics.dt), validation_rollouts=rollouts)
    (output/'diagnostic.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = diagnose_run(args.run, args.dataset, args.output)
    print(json.dumps({s: r['overall'] for s, r in report['teacher_states'].items()}, indent=2))


if __name__ == '__main__':
    main()
