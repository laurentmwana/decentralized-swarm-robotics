"""Imitation multi-formes v2 : cible propre, voisins physiques et partitions figées."""
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import yaml

from .data import CollectionConfig, nonoverlapping_positions
from src.reference.allocation import _integer
from src.reference.catalog import catalog_entries
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import ShapeGains, ShapeMetricConfig, execute_shape, evaluate_shape
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


FEATURE_NAMES = ('position_x', 'position_y', 'velocity_x', 'velocity_y',
                 'local_target_x', 'local_target_y', 'phase', 'robot_count')
MESSAGE_NAMES = ('relative_position_x', 'relative_position_y', 'relative_velocity_x', 'relative_velocity_y')
SPLITS = ('train', 'validation', 'test')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_fingerprint(mission):
    """Segments non orientés quantifiés ; alias et sens de parcours ignorés."""
    segments = set()
    for stroke, closed in zip(mission.strokes, mission.closed):
        vertices = [tuple(int(v) for v in np.rint(np.asarray(p)*1e9)) for p in stroke]
        if closed: vertices.append(vertices[0])
        segments.update(tuple(sorted((a, b))) for a, b in zip(vertices[:-1], vertices[1:]))
    return hashlib.sha256(json.dumps(sorted(segments)).encode()).hexdigest()


def default_partitions(width=6., seed=20261029):
    """Stratifier les groupes géométriques, sans séparer V/V ni O/cercle."""
    _integer(seed, 'partition_seed')
    groups = {}
    for entry in catalog_entries():
        mission = named_shape(entry['identifier'], 12, width=width)
        groups.setdefault(geometry_fingerprint(mission), []).append(entry)
    strata = {key: [] for key in ('letters', 'numbers', 'geometries')}
    for fingerprint, entries in groups.items():
        category = next(key for key in strata if any(e['category'] == key for e in entries))
        strata[category].append((fingerprint, entries))
    parts = {s: [] for s in SPLITS}
    for rows in strata.values():
        rows.sort(key=lambda row: hashlib.sha256(f'{seed}:{row[0]}'.encode()).hexdigest())
        holdout = max(1, round(len(rows)*.15))
        for index, (_, entries) in enumerate(rows):
            split = 'validation' if index < holdout else 'test' if index < 2*holdout else 'train'
            parts[split].extend(e['identifier'] for e in entries)
    return {s: sorted(values) for s, values in parts.items()}


@dataclass(frozen=True)
class MultiFormCollectionConfig:
    shape_splits: dict
    episode_seeds: dict
    robot_count: int = 12
    width: float = 6.
    minimum_distance: float = .45
    max_spawn_attempts: int = 20000

    def __post_init__(self):
        _integer(self.robot_count, 'robot_count', 2)
        if not np.isfinite(self.width) or self.width <= 0:
            raise ValueError('Largeur finie et positive requise')
        seeds = CollectionConfig(self.episode_seeds, self.minimum_distance, self.max_spawn_attempts)
        if set(self.shape_splits) != set(SPLITS):
            raise ValueError('Les trois partitions de formes sont requises')
        normalized, identifiers, fingerprints = {}, set(), {}
        for split in SPLITS:
            if not isinstance(self.shape_splits[split], (list, tuple)) or not self.shape_splits[split]:
                raise ValueError('Chaque partition doit contenir des formes')
            normalized[split] = []
            for name in self.shape_splits[split]:
                mission = named_shape(name, self.robot_count, width=self.width)
                if mission.identifier in identifiers:
                    raise ValueError('Identifiant de forme dupliqué entre épisodes ou partitions')
                identifiers.add(mission.identifier)
                fingerprint = geometry_fingerprint(mission)
                if fingerprint in fingerprints and fingerprints[fingerprint] != split:
                    raise ValueError('Formes géométriquement équivalentes dans plusieurs partitions')
                fingerprints[fingerprint] = split
                normalized[split].append(mission.identifier)
        object.__setattr__(self, 'shape_splits', {s: tuple(v) for s, v in normalized.items()})
        object.__setattr__(self, 'episode_seeds', seeds.splits)

    @classmethod
    def from_yaml(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def episode_arrays(result):
    episode = result.episode
    p, v = episode.positions[:-1], episode.velocities[:-1]
    if result.local_targets is None or result.local_targets.shape != p.shape:
        raise ValueError('Cibles locales alignées avant commande requises')
    phases = np.broadcast_to(result.phases[:, None, None], p.shape[:2]+(1,))
    count = np.full(p.shape[:2]+(1,), p.shape[1])
    features = np.concatenate((p, v, result.local_targets, phases, count), axis=-1)
    graph = episode.adjacency[:-1]
    relative = np.concatenate((p[:, None]-p[:, :, None], v[:, None]-v[:, :, None]), axis=-1)
    relative = np.where(graph[..., None], relative, 0.)
    mask = np.broadcast_to(np.isin(result.phases, (0, 3))[:, None], p.shape[:2]).copy()
    return dict(node_features=features, relative_neighbors=relative, adjacency=graph, learn_mask=mask,
                targets=episode.accelerations, requested_targets=episode.requested_accelerations,
                positions=episode.positions, velocities=episode.velocities, times=episode.times,
                phases=result.phases, local_targets=result.local_targets, assignments=result.assignments,
                safety_infeasible=result.safety_infeasible, knowledge_sizes=result.knowledge_sizes,
                allocation_consensus=result.allocation_consensus, messages_per_step=result.messages_per_step,
                shape_targets=result.targets)


def local_example(arrays, step, robot):
    _integer(step, 'step'); _integer(robot, 'robot')
    if step >= len(arrays['node_features']) or robot >= arrays['node_features'].shape[1]:
        raise ValueError('Exemple hors épisode')
    neighbors = arrays['adjacency'][step, robot]
    # Seul accès destiné au futur modèle : aucune cible privée des voisins.
    return dict(own=arrays['node_features'][step, robot].copy(),
                neighbors=arrays['relative_neighbors'][step, robot, neighbors].copy(),
                target=arrays['targets'][step, robot].copy(), learn=bool(arrays['learn_mask'][step, robot]))


def collect_dataset(config, output, *, source=None):
    simulator = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=400, communication_radius=3.)
    gains = ShapeGains(detour_only_on_constraint=True)
    reconnection = ReconnectionConfig()
    protocol = ShapeMetricConfig(.1, .05, 1., 40.)
    if simulator.horizon*simulator.physics.dt+1e-12 < protocol.deadline_seconds:
        raise ValueError('Horizon insuffisant pour le protocole de collecte')
    if config.minimum_distance < gains.safety_distance:
        raise ValueError('Le domaine initial doit respecter la distance de sécurité')
    if sum(reconnection.steps(simulator.physics.dt))+config.robot_count**2 >= simulator.horizon:
        raise ValueError('Budget de phases incompatible avec l’horizon de collecte')
    missions = {name: named_shape(name, config.robot_count, width=config.width)
                for names in config.shape_splits.values() for name in names}
    for mission in missions.values(): mission.validate_environment(simulator, gains.safety_distance)
    source_manifest, source_entries = None, {}
    if source is not None:
        source = Path(source).resolve()
        source_manifest = load_manifest(source)
        previous = MultiFormCollectionConfig(**source_manifest['collection'])
        if (previous.shape_splits != config.shape_splits or previous.robot_count != config.robot_count or
            previous.width != config.width or previous.minimum_distance != config.minimum_distance or
            previous.max_spawn_attempts != config.max_spawn_attempts):
            raise ValueError('Extension : partitions de formes et domaine initial doivent être conservés')
        if config.episode_seeds['test'] != previous.episode_seeds['test'] or any(
            not set(previous.episode_seeds[s]).issubset(config.episode_seeds[s]) for s in SPLITS):
            raise ValueError('Extension : conserver test et les graines existantes')
        if (source_manifest['simulator'] != asdict(simulator) or source_manifest['protocol'] != asdict(protocol) or
            json.dumps(source_manifest['teacher'], sort_keys=True) != json.dumps(dict(gains=asdict(gains),reconnection=asdict(reconnection)),sort_keys=True)):
            raise ValueError('Extension : professeur ou protocole modifié')
        source_entries = {(e['split'],e['shape'],e['seed']):e for e in source_manifest['episodes']}
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=2, kind='multiform_local_target_imitation', status='incomplete',
        collection=asdict(config), simulator=asdict(simulator), teacher=dict(gains=asdict(gains), reconnection=asdict(reconnection)),
        protocol=asdict(protocol), feature_names=FEATURE_NAMES, message_feature_names=MESSAGE_NAMES,
        target='bounded_teacher_acceleration_m_s2', learned_phases=[0, 3], classical_phases=[1, 2],
        geometry_groups={name:geometry_fingerprint(mission) for name, mission in missions.items()},
        missions={name:asdict(mission) for name, mission in missions.items()},
        units=dict(position='m', velocity='m/s', acceleration='m/s^2', time='s'), episodes=[])
    if source is not None:
        manifest['derived_from'] = dict(path=str(source),manifest_sha256=sha256(source/'manifest.json'),
                                      policy='append_train_validation_preserve_test_archives')
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    save()
    fingerprints = {}
    for split in SPLITS:
        (output/split).mkdir()
        for seed in config.episode_seeds[split]:
            initial = nonoverlapping_positions(simulator, config.robot_count, seed, config.minimum_distance, config.max_spawn_attempts)
            for name in config.shape_splits[split]:
                prior = source_entries.get((split,name,seed))
                if prior is not None:
                    archive = (source/prior['path']).resolve()
                    if not archive.is_relative_to(source) or sha256(archive) != prior['sha256']:
                        raise ValueError('Archive source corrompue ou extérieure')
                    shutil.copyfile(archive,output/prior['path'])
                    entry = dict(prior)
                    previous = fingerprints.get(entry['simulation_sha256'])
                    if previous is not None and previous['split'] != split:
                        raise ValueError('Simulation identique détectée entre partitions')
                    entry['duplicate_within_split_of'] = previous['path'] if previous is not None else None
                    fingerprints.setdefault(entry['simulation_sha256'],entry)
                    manifest['episodes'].append(entry); save()
                    continue
                result = execute_shape(missions[name], simulator, initial, gains=gains, seed=seed,
                                       safety_filter=True, reconnection=reconnection)
                arrays = episode_arrays(result)
                fingerprint = hashlib.sha256(arrays['node_features'].tobytes()+arrays['targets'].tobytes()).hexdigest()
                previous = fingerprints.get(fingerprint)
                if previous is not None and previous['split'] != split:
                    raise ValueError('Simulation identique détectée entre partitions')
                relative_path = f'{split}/{name}-{seed}.npz'
                metadata = dict(schema_version=2, split=split, seed=seed, shape=name,
                                geometry_fingerprint=manifest['geometry_groups'][name])
                with (output/relative_path).open('xb') as stream:
                    np.savez_compressed(stream, **arrays, metadata=json.dumps(metadata))
                entry = dict(**metadata, path=relative_path, sha256=sha256(output/relative_path),
                             simulation_sha256=fingerprint, graphs=len(arrays['targets']),
                             learned_robot_samples=int(arrays['learn_mask'].sum()),
                             quality=evaluate_shape(result, protocol),
                             duplicate_within_split_of=previous['path'] if previous is not None else None)
                fingerprints.setdefault(fingerprint, entry)
                manifest['episodes'].append(entry); save()
    manifest.update(status='complete', quality_status='passed' if all(e['quality']['success'] for e in manifest['episodes']) else 'review_required',
        summary={s:dict(episodes=sum(e['split']==s for e in manifest['episodes']),
            successes=sum(e['quality']['success'] for e in manifest['episodes'] if e['split']==s),
            learned_robot_samples=sum(e['learned_robot_samples'] for e in manifest['episodes'] if e['split']==s),
            unique_episodes=sum(e['split']==s and e['duplicate_within_split_of'] is None for e in manifest['episodes'])) for s in SPLITS})
    save()
    return manifest


def load_manifest(dataset, *, allow_quality_failures=False):
    manifest = json.loads((Path(dataset)/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2 or manifest.get('kind') != 'multiform_local_target_imitation' or manifest.get('status') != 'complete':
        raise ValueError('Jeu multi-formes incomplet ou schéma incompatible')
    config = MultiFormCollectionConfig(**manifest['collection'])
    expected = {(s, name, seed) for s in SPLITS for name in config.shape_splits[s] for seed in config.episode_seeds[s]}
    entries = manifest['episodes']
    if len(entries) != len(expected) or {(e['split'], e['shape'], e['seed']) for e in entries} != expected:
        raise ValueError('Partition multi-formes incohérente')
    if len({e['path'] for e in entries}) != len(entries):
        raise ValueError('Archives multi-formes dupliquées')
    names = {name for shapes in config.shape_splits.values() for name in shapes}
    if set(manifest['geometry_groups']) != names or set(manifest['missions']) != names:
        raise ValueError('Consignes multi-formes manquantes ou supplémentaires')
    for name, fingerprint in manifest['geometry_groups'].items():
        mission = named_shape(name, config.robot_count, width=config.width)
        if fingerprint != geometry_fingerprint(mission):
            raise ValueError('La géométrie a changé depuis la collecte')
        if json.dumps(manifest['missions'][name], sort_keys=True) != json.dumps(asdict(mission), sort_keys=True):
            raise ValueError('La consigne sauvegardée diffère du catalogue')
    if tuple(manifest['feature_names']) != FEATURE_NAMES or tuple(manifest['message_feature_names']) != MESSAGE_NAMES:
        raise ValueError('Caractéristiques multi-formes incompatibles')
    hashes, previous_paths = {}, {}
    for entry in entries:
        if entry.get('schema_version') != 2 or entry['geometry_fingerprint'] != manifest['geometry_groups'][entry['shape']]:
            raise ValueError('Géométrie d’épisode incompatible')
        digest = entry['simulation_sha256']
        if digest in hashes and hashes[digest] != entry['split']:
            raise ValueError('Simulation identique entre partitions')
        hashes[digest] = entry['split']
        duplicate = entry['duplicate_within_split_of']
        if duplicate is not None:
            parent = previous_paths.get(duplicate)
            if parent is None or parent['split'] != entry['split'] or parent['simulation_sha256'] != digest:
                raise ValueError('Provenance de duplication incompatible')
        previous_paths[entry['path']] = entry
    all_successful = all(e['quality']['success'] for e in entries)
    if manifest.get('quality_status') == 'passed' and not all_successful:
        raise ValueError('Qualité déclarée incohérente avec les échecs conservés')
    if manifest.get('quality_status') != 'passed' and not allow_quality_failures:
        raise ValueError('Qualité du professeur multi-formes à examiner')
    return manifest


def load_episode(dataset, entry, expected_split):
    if expected_split not in SPLITS or entry['split'] != expected_split:
        raise ValueError('Épisode demandé dans une autre partition')
    root = Path(dataset).resolve(); path = (root/entry['path']).resolve()
    if not path.is_relative_to(root) or sha256(path) != entry['sha256']:
        raise ValueError('Archive multi-formes corrompue ou extérieure')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if any(metadata[k] != entry[k] for k in ('schema_version', 'split', 'seed', 'shape', 'geometry_fingerprint')):
            raise ValueError('Métadonnées multi-formes incompatibles')
        arrays = {k: archive[k].copy() for k in archive.files if k != 'metadata'}
    features, graph, labels = arrays['node_features'], arrays['adjacency'], arrays['targets']
    if features.ndim != 3 or features.shape[-1] != 8 or labels.shape != features.shape[:2]+(2,) or graph.shape != features.shape[:2]+(features.shape[1],):
        raise ValueError('Dimensions multi-formes incompatibles')
    if not np.isfinite(features).all() or not np.isfinite(labels).all() or graph.dtype != bool:
        raise ValueError('Données multi-formes invalides')
    if arrays['relative_neighbors'].shape != graph.shape+(4,) or arrays['learn_mask'].shape != features.shape[:2] or arrays['learn_mask'].dtype != bool:
        raise ValueError('Voisinage ou masque multi-formes incompatible')
    if not np.isfinite(arrays['relative_neighbors']).all() or graph.diagonal(axis1=-2, axis2=-1).any():
        raise ValueError('Voisinage multi-formes invalide')
    if np.any(arrays['relative_neighbors'][~graph] != 0):
        raise ValueError('Les voisins hors portée doivent être masqués')
    expected_mask = np.broadcast_to(np.isin(arrays['phases'], (0, 3))[:, None], features.shape[:2])
    if not np.array_equal(arrays['learn_mask'], expected_mask):
        raise ValueError('Masque de phases appris incompatible')
    t, n = features.shape[:2]
    if arrays['positions'].shape != (t+1, n, 2) or arrays['velocities'].shape != (t+1, n, 2):
        raise ValueError('Trajectoires multi-formes incompatibles')
    if arrays['local_targets'].shape != (t, n, 2) or arrays['phases'].shape != (t,) or not np.isin(arrays['phases'], (0, 1, 2, 3)).all():
        raise ValueError('Cibles ou phases multi-formes incompatibles')
    aligned = (np.array_equal(features[..., :2], arrays['positions'][:-1]) and
               np.array_equal(features[..., 2:4], arrays['velocities'][:-1]) and
               np.array_equal(features[..., 4:6], arrays['local_targets']) and
               np.array_equal(features[..., 6], np.broadcast_to(arrays['phases'][:, None], (t, n))) and
               (features[..., 7] == n).all())
    if not aligned:
        raise ValueError('Désalignement des états, cibles ou phases')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 2 or manifest.get('kind') != 'multiform_local_target_imitation':
        raise ValueError('Schéma multi-formes incompatible')
    if (np.linalg.norm(labels, axis=-1) > manifest['simulator']['physics']['max_acceleration']+1e-9).any():
        raise ValueError('Label multi-formes hors limite physique')
    digest = hashlib.sha256(features.tobytes()+labels.tobytes()).hexdigest()
    if digest != entry['simulation_sha256']:
        raise ValueError('Empreinte des exemples multi-formes incompatible')
    return arrays


def verify_dataset(dataset, *, splits=('train', 'validation')):
    """Audit explicite ; ne lit jamais test par défaut."""
    if not splits or len(set(splits)) != len(splits) or any(s not in SPLITS for s in splits):
        raise ValueError('Partitions explicites uniques requises pour l’audit')
    manifest = load_manifest(dataset, allow_quality_failures=True)
    initial_hashes, audit = {}, {}
    for split in splits:
        count = unique = learned = 0
        for entry in manifest['episodes']:
            if entry['split'] != split: continue
            arrays = load_episode(dataset, entry, split)
            initial = hashlib.sha256(np.asarray(arrays['positions'][0], dtype='<f8').tobytes()).hexdigest()
            if initial in initial_hashes and initial_hashes[initial] != split:
                raise ValueError('Même état initial dans plusieurs partitions')
            initial_hashes[initial] = split
            count += 1; unique += entry['duplicate_within_split_of'] is None
            learned += int(arrays['learn_mask'].sum())
        audit[split] = dict(episodes=count, unique_episodes=unique, learned_robot_samples=learned)
    return dict(schema_version=2, manifest_sha256=sha256(Path(dataset)/'manifest.json'),
                quality_status=manifest['quality_status'], splits_checked=list(splits), summary=audit)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/multiform-data.yaml')
    parser.add_argument('--output')
    parser.add_argument('--write-default-config', action='store_true')
    parser.add_argument('--verify', help='Auditer un jeu existant ; aucun entraînement')
    parser.add_argument('--include-test', action='store_true', help='Inclure explicitement test dans l’audit')
    parser.add_argument('--extend-from', help='Étendre uniquement train/validation en conservant test et le protocole')
    args = parser.parse_args(argv)
    if args.verify is not None:
        result = verify_dataset(args.verify, splits=SPLITS if args.include_test else ('train', 'validation'))
        print(json.dumps(result, indent=2))
        return result
    if args.write_default_config:
        config = dict(shape_splits=default_partitions(), episode_seeds=dict(train=[7100], validation=[7200], test=[7300]),
                      robot_count=12, width=6., minimum_distance=.45, max_spawn_attempts=20000)
        with Path(args.config).open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
        return config
    if args.output is None: parser.error('--output requis pour collecter')
    result = collect_dataset(MultiFormCollectionConfig.from_yaml(args.config), args.output, source=args.extend_from)
    print(json.dumps(dict(quality_status=result['quality_status'], summary=result['summary']), indent=2))
    return result


if __name__ == '__main__':
    main()
