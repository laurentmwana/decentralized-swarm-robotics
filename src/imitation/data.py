"""Collecte déterministe du professeur ; séparation stricte par épisodes."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.common.seeds import SeedManager
from src.reference.metrics import evaluate_circle, summarize
from src.simulator.acceleration_episode import AccelerationEpisode

FEATURE_NAMES = ('position_x', 'position_y', 'velocity_x', 'velocity_y',
                 'center_x', 'center_y', 'radius', 'robot_count')
RELATIVE_NAMES = ('relative_position_x', 'relative_position_y',
                  'relative_velocity_x', 'relative_velocity_y')


@dataclass(frozen=True)
class CollectionConfig:
    splits: dict[str, list[int]]
    minimum_distance: float
    max_spawn_attempts: int

    def __post_init__(self):
        if set(self.splits) != {'train', 'validation', 'test'}:
            raise ValueError('Les trois ensembles train, validation et test sont requis')
        seeds = []
        for values in self.splits.values():
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError('Chaque ensemble doit contenir des graines')
            if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in values):
                raise ValueError('Les graines doivent être des entiers positifs ou nuls')
            seeds.extend(values)
        if len(seeds) != len(set(seeds)):
            raise ValueError('Une graine ne peut apparaître dans plusieurs épisodes ou ensembles')
        if not np.isfinite(self.minimum_distance) or self.minimum_distance <= 0:
            raise ValueError('Distance minimale finie et positive requise')
        if isinstance(self.max_spawn_attempts, bool) or not isinstance(self.max_spawn_attempts, int) or self.max_spawn_attempts < 1:
            raise ValueError('Budget de tirages entier positif requis')
        object.__setattr__(self, 'splits', {key: tuple(values) for key, values in self.splits.items()})

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError('Objet de configuration requis')
        return cls(**raw)


def nonoverlapping_positions(config, robot_count, seed, minimum_distance, max_attempts):
    """Rejet séquentiel dans un disque ; budget global borné, pas de repli silencieux."""
    if isinstance(robot_count, bool) or not isinstance(robot_count, int) or robot_count < 1:
        raise ValueError('Effectif entier positif requis')
    if not np.isfinite(minimum_distance) or minimum_distance <= 2*config.physics.robot_radius:
        raise ValueError('La distance minimale doit dépasser le diamètre physique')
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError('Budget de tirages entier positif requis')
    rng = SeedManager(seed).rng('imitation_initial_positions')
    positions = []
    for _ in range(max_attempts):
        angle = rng.uniform(0, 2*np.pi)
        radius = config.spawn_radius * np.sqrt(rng.uniform())
        candidate = radius * np.array([np.cos(angle), np.sin(angle)])
        if not positions or np.linalg.norm(np.array(positions) - candidate, axis=1).min() >= minimum_distance:
            positions.append(candidate)
            if len(positions) == robot_count:
                return np.array(positions)
    raise ValueError('Initialisation impossible dans le budget de tirages ; modifier explicitement le domaine')


def episode_arrays(result, mission):
    """Un graphe par pas : états AVANT commande, voisinage masqué et cibles bornées."""
    positions, velocities = result.positions[:-1], result.velocities[:-1]
    if positions.shape != velocities.shape or positions.shape[1] != mission.robot_count:
        raise ValueError('État et consigne incompatibles')
    shape = (*positions.shape[:2], 1)
    center = np.broadcast_to(np.asarray(mission.center), positions.shape)
    features = np.concatenate((positions, velocities, center,
                               np.full(shape, mission.radius), np.full(shape, mission.robot_count)), axis=-1)
    adjacency = result.adjacency[:-1]
    relative_positions = positions[:, None, :, :] - positions[:, :, None, :]
    relative_velocities = velocities[:, None, :, :] - velocities[:, :, None, :]
    neighbors = np.concatenate((relative_positions, relative_velocities), axis=-1)
    neighbors = np.where(adjacency[..., None], neighbors, 0.)
    return dict(node_features=features, adjacency=adjacency, relative_neighbors=neighbors,
                targets=result.accelerations.copy(), requested_targets=result.requested_accelerations.copy(),
                positions=result.positions, velocities=result.velocities, times=result.times)


def _fingerprint(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


def collect_dataset(episode_config, teacher, metric_config, collection_config, output):
    """Conserve tous les épisodes et leurs échecs. Refuse un dossier de sortie existant."""
    teacher.validate_environment(episode_config)
    if episode_config.horizon * episode_config.physics.dt + 1e-12 < metric_config.deadline_seconds:
        raise ValueError('Horizon insuffisant pour le protocole de qualité')
    if collection_config.minimum_distance < teacher.gains.safety_distance:
        raise ValueError('Le domaine de collecte doit respecter la distance de sécurité du professeur')
    if collection_config.minimum_distance <= 2*episode_config.physics.robot_radius:
        raise ValueError('Le domaine de collecte doit exclure les chevauchements')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, status='incomplete',
                    simulator=asdict(episode_config), teacher=asdict(teacher), protocol=asdict(metric_config),
                    collection=asdict(collection_config), feature_names=FEATURE_NAMES,
                    relative_feature_names=RELATIVE_NAMES, target='bounded_teacher_acceleration_m_s2',
                    units=dict(position='m', velocity='m/s', acceleration='m/s^2', time='s'), episodes=[])
    path = output / 'manifest.json'
    def save_manifest():
        path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    save_manifest()
    initial_hashes, trajectory_hashes = set(), set()
    for split, seeds in collection_config.splits.items():
        (output / split).mkdir()
        for seed in seeds:
            initial = nonoverlapping_positions(episode_config, teacher.mission.robot_count, seed,
                                               collection_config.minimum_distance, collection_config.max_spawn_attempts)
            initial_hash = _fingerprint(initial)
            result = AccelerationEpisode(episode_config).execute(teacher, teacher.mission.robot_count, seed, initial)
            trajectory_hash = _fingerprint(result.positions)
            if initial_hash in initial_hashes or trajectory_hash in trajectory_hashes:
                raise ValueError('Épisodes identiques détectés ; collecte interrompue pour éviter une fuite')
            initial_hashes.add(initial_hash)
            trajectory_hashes.add(trajectory_hash)
            quality = evaluate_circle(result, teacher.mission, metric_config)
            relative_path = f'{split}/episode-{seed}.npz'
            metadata = dict(schema_version=1, split=split, seed=seed, mission=asdict(teacher.mission),
                            initial_sha256=initial_hash, trajectory_sha256=trajectory_hash)
            with (output / relative_path).open('wb') as stream:
                np.savez_compressed(stream, **episode_arrays(result, teacher.mission), metadata=json.dumps(metadata))
            file_hash = hashlib.sha256((output / relative_path).read_bytes()).hexdigest()
            manifest['episodes'].append(dict(split=split, seed=seed, path=relative_path, sha256=file_hash,
                                             initial_sha256=initial_hash, trajectory_sha256=trajectory_hash,
                                             samples=episode_config.horizon * teacher.mission.robot_count, quality=quality))
            save_manifest()
    manifest['summary'] = {split: summarize([r['quality'] for r in manifest['episodes'] if r['split']==split])
                           for split in collection_config.splits}
    manifest['status'] = 'complete'
    manifest['quality_status'] = 'passed' if all(r['quality']['success'] for r in manifest['episodes']) else 'review_required'
    save_manifest()
    return manifest


def load_episode(dataset, entry, expected_split):
    """Chargement sans pickle, intégrité et appartenance à l'ensemble vérifiées."""
    if expected_split not in ('train', 'validation', 'test') or entry['split'] != expected_split:
        raise ValueError('Épisode demandé dans un autre ensemble')
    root = Path(dataset).resolve()
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Chemin extérieur au jeu de données')
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Archive modifiée ou corrompue')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        if metadata['split'] != expected_split or metadata['seed'] != entry['seed']:
            raise ValueError('Métadonnées incompatibles avec le manifeste')
        return {name: archive[name].copy() for name in archive.files if name != 'metadata'}


def load_manifest(dataset, *, allow_quality_failures=False):
    """Refuse une collecte inachevée ou une partition incohérente avant consommation."""
    manifest = json.loads((Path(dataset) / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Jeu de données incomplet ou version non prise en charge')
    config = CollectionConfig(**manifest['collection'])
    expected = {(split, seed) for split, seeds in config.splits.items() for seed in seeds}
    entries = manifest['episodes']
    actual = [(e['split'], e['seed']) for e in entries]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Partition incohérente ou épisodes manquants/dupliqués')
    for field in ('path', 'initial_sha256', 'trajectory_sha256'):
        if len({e[field] for e in entries}) != len(entries):
            raise ValueError('Fuite ou duplication entre épisodes détectée')
    if manifest.get('quality_status') != 'passed' and not allow_quality_failures:
        raise ValueError('Qualité du professeur à examiner ; accès explicite requis pour inspection')
    return manifest
