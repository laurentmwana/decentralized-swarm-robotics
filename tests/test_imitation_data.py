from dataclasses import replace
import json

import numpy as np
import pytest

from src.imitation.data import (CollectionConfig, collect_dataset, episode_arrays,
                                load_episode, load_manifest, nonoverlapping_positions)
from src.reference.circle import CircleTeacher
from src.reference.metrics import CircleMetricConfig
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig


def settings():
    cfg = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), horizon=3)
    teacher = CircleTeacher.from_yaml('config/circle-teacher.yaml', cfg)
    protocol = CircleMetricConfig(.1, .05, .1, .3)
    collection = CollectionConfig(dict(train=[100], validation=[200], test=[300]), .4, 20000)
    return cfg, teacher, protocol, collection


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path/'data'
    manifest = collect_dataset(*settings(), folder)
    return folder, manifest


def test_spawn_is_reproducible_bounded_and_without_overlap():
    cfg, _, _, _ = settings()
    args = cfg, 12, 7, .4, 10000
    a = nonoverlapping_positions(*args)
    np.testing.assert_array_equal(a, nonoverlapping_positions(*args))
    assert not np.array_equal(a, nonoverlapping_positions(cfg, 12, 8, .4, 10000))
    distances = np.linalg.norm(a[:,None]-a[None,:], axis=-1)
    assert distances[~np.eye(12,dtype=bool)].min() >= .4
    assert np.linalg.norm(a,axis=1).max() <= cfg.spawn_radius


def test_impossible_spawn_fails_instead_of_overlapping():
    cfg, _, _, _ = settings()
    with pytest.raises(ValueError, match='budget'):
        nonoverlapping_positions(cfg, 12, 0, 20., 20)
    with pytest.raises(ValueError):
        nonoverlapping_positions(cfg, 12, 0, .2, 100)


@pytest.mark.parametrize('splits', [dict(train=[0], validation=[0], test=[2]),
    dict(train=[0,0], validation=[1], test=[2]), dict(train=[], validation=[1], test=[2]),
    dict(train=[0], validation=[1]), dict(train=[True], validation=[1], test=[2]),
    dict(train=[0], validation=[-1], test=[2])])
def test_invalid_partition(splits):
    with pytest.raises(ValueError):
        CollectionConfig(splits, .4, 100)


@pytest.mark.parametrize('distance,attempts', [(0,100), (np.nan,100), (.4,0), (.4,True)])
def test_invalid_collection_parameters(distance, attempts):
    with pytest.raises(ValueError):
        CollectionConfig(dict(train=[0], validation=[1], test=[2]), distance, attempts)


def test_features_and_targets_are_aligned_before_command():
    cfg, teacher, _, _ = settings()
    episode = AccelerationEpisode(cfg).execute(teacher, 12, 4)
    arrays = episode_arrays(episode, teacher.mission)
    assert arrays['node_features'].shape == (3,12,8)
    np.testing.assert_array_equal(arrays['node_features'][...,:2], episode.positions[:-1])
    np.testing.assert_array_equal(arrays['node_features'][...,2:4], episode.velocities[:-1])
    np.testing.assert_array_equal(arrays['targets'], episode.accelerations)
    assert arrays['relative_neighbors'].shape == (3,12,12,4)
    mask = arrays['adjacency']
    assert not arrays['relative_neighbors'][~mask].any()
    t,i,j = np.argwhere(mask)[0]
    np.testing.assert_allclose(arrays['relative_neighbors'][t,i,j,:2], episode.positions[t,j]-episode.positions[t,i])
    assert np.linalg.norm(arrays['targets'],axis=-1).max() <= cfg.physics.max_acceleration + 1e-12


def test_partition_export_and_integrity(dataset):
    folder, original = dataset
    manifest = load_manifest(folder, allow_quality_failures=True)
    assert len(manifest['episodes']) == 3
    assert manifest['quality_status'] == 'review_required'  # horizon volontairement court
    for entry in manifest['episodes']:
        arrays = load_episode(folder, entry, entry['split'])
        assert arrays['targets'].shape == (3,12,2)
        assert entry['samples'] == 36
    assert len({e['initial_sha256'] for e in original['episodes']}) == 3
    assert len({e['trajectory_sha256'] for e in original['episodes']}) == 3


def test_cross_split_loading_refused(dataset):
    folder, manifest = dataset
    with pytest.raises(ValueError, match='autre ensemble'):
        load_episode(folder, manifest['episodes'][0], 'test')


def test_corrupt_archive_refused(dataset):
    folder, manifest = dataset
    entry = manifest['episodes'][0]
    with (folder/entry['path']).open('ab') as stream:
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='corrompue'):
        load_episode(folder, entry, 'train')


def test_outside_path_refused(dataset):
    folder, manifest = dataset
    entry = dict(manifest['episodes'][0], path='../outside.npz')
    with pytest.raises(ValueError, match='extérieur'):
        load_episode(folder, entry, 'train')


@pytest.mark.parametrize('change', ['incomplete','missing','duplicate','shared_initial'])
def test_manifest_rejects_incomplete_or_leaking_data(dataset, change):
    folder, manifest = dataset
    if change == 'incomplete': manifest['status']='incomplete'
    elif change == 'missing': manifest['episodes'].pop()
    elif change == 'duplicate': manifest['episodes'].append(manifest['episodes'][0])
    else: manifest['episodes'][1]['initial_sha256']=manifest['episodes'][0]['initial_sha256']
    (folder/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(folder)


def test_existing_output_not_overwritten(dataset):
    folder, _ = dataset
    before = (folder/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        collect_dataset(*settings(), folder)
    assert before == (folder/'manifest.json').read_bytes()


def test_collection_is_numerically_reproducible(tmp_path):
    a = collect_dataset(*settings(), tmp_path/'a')
    b = collect_dataset(*settings(), tmp_path/'b')
    for ea,eb in zip(a['episodes'],b['episodes']):
        assert ea['trajectory_sha256'] == eb['trajectory_sha256']
        assert ea['quality'] == eb['quality']


def test_duplicate_states_interrupt_collection(tmp_path, monkeypatch):
    import src.imitation.data as data
    cfg, _, _, _ = settings()
    fixed = nonoverlapping_positions(cfg,12,0,.4,10000)
    monkeypatch.setattr(data, 'nonoverlapping_positions', lambda *args: fixed.copy())
    with pytest.raises(ValueError, match='identiques'):
        collect_dataset(*settings(), tmp_path/'data')
    with pytest.raises(ValueError, match='incomplet'):
        load_manifest(tmp_path/'data')


def test_quality_failures_are_retained(dataset):
    _, manifest = dataset
    assert len(manifest['episodes']) == 3
    assert all(not e['quality']['success'] for e in manifest['episodes'])
    assert all(e['quality']['stabilization_time_s'] is None for e in manifest['episodes'])


def test_cli_collects_without_training(tmp_path):
    from src.imitation.collect import main
    cfg, _, protocol, collection = settings()
    import yaml
    from dataclasses import asdict
    for name, data in [('simulator', asdict(cfg)), ('metrics', asdict(protocol)), ('collection', asdict(collection))]:
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(data))
    manifest=main(['--simulator',str(tmp_path/'simulator.yaml'), '--metrics',str(tmp_path/'metrics.yaml'),
                   '--collection',str(tmp_path/'collection.yaml'), '--output',str(tmp_path/'data')])
    assert manifest['status'] == 'complete'


def test_quality_gate_requires_explicit_inspection(dataset):
    folder, _ = dataset
    with pytest.raises(ValueError, match='examiner'):
        load_manifest(folder)
    assert load_manifest(folder, allow_quality_failures=True)['quality_status'] == 'review_required'
