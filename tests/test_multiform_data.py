"""Contrat multi-formes, partitions par géométrie et absence de cibles privées."""
import json
import shutil

import numpy as np
import pytest

from src.imitation.multiform_data import (FEATURE_NAMES, MultiFormCollectionConfig, collect_dataset,
    default_partitions, geometry_fingerprint, load_manifest, load_episode, local_example, verify_dataset)
from src.reference.catalog import catalog_entries
from src.reference.reconnection import ReconnectionConfig, rendezvous_targets
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def config():
    return MultiFormCollectionConfig(dict(train=['square'], validation=['triangle'], test=['A']),
                                     dict(train=[7100], validation=[7200], test=[7300]), robot_count=4, width=2.)


@pytest.fixture(scope='module')
def pilot(tmp_path_factory):
    output = tmp_path_factory.mktemp('multiform')/'pilot'
    collect_dataset(config(), output)
    return output


def test_complete_partition_is_stable_and_groups_aliases():
    parts = default_partitions()
    assert parts == default_partitions()
    flattened = [name for values in parts.values() for name in values]
    assert set(flattened) == {e['identifier'] for e in catalog_entries()}
    assert len(flattened) == 56
    assigned = {name:split for split, values in parts.items() for name in values}
    assert assigned['letters_O'] == assigned['geometries_circle']
    assert assigned['letters_V'] == assigned['geometries_v']
    assert geometry_fingerprint(named_shape('O', 12)) == geometry_fingerprint(named_shape('circle', 12))


@pytest.mark.parametrize('case', ['alias', 'seeds', 'identifier', 'empty'])
def test_partition_leaks_are_rejected(case):
    shapes = dict(train=['O'], validation=['A'], test=['B'])
    seeds = dict(train=[1], validation=[2], test=[3])
    if case == 'alias': shapes['validation'] = ['circle']
    elif case == 'seeds': seeds['test'] = [1]
    elif case == 'identifier': shapes['test'] = ['letters_O']
    else: shapes['test'] = []
    with pytest.raises(ValueError): MultiFormCollectionConfig(shapes, seeds)


def test_episode_states_targets_phases_and_neighbor_masks_align(pilot):
    manifest = load_manifest(pilot, allow_quality_failures=True)
    entry = manifest['episodes'][0]
    arrays = load_episode(pilot, entry, 'train')
    features = arrays['node_features']
    assert tuple(manifest['feature_names']) == FEATURE_NAMES
    assert features.shape == (400, 4, 8)
    np.testing.assert_array_equal(features[..., :2], arrays['positions'][:-1])
    np.testing.assert_array_equal(features[..., 2:4], arrays['velocities'][:-1])
    np.testing.assert_array_equal(features[..., 4:6], arrays['local_targets'])
    assert arrays['positions'].shape == (401, 4, 2)
    expected = rendezvous_targets(4, AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'), ReconnectionConfig(), .4)
    np.testing.assert_allclose(features[0, :, 4:6], expected)
    assert arrays['learn_mask'][:80].all()
    assert not arrays['learn_mask'][80:106].any()
    assert arrays['learn_mask'][106:].all()
    assert not arrays['relative_neighbors'][~arrays['adjacency']].any()
    assert np.linalg.norm(arrays['targets'], axis=-1).max() <= 2+1e-9


def test_local_example_never_exposes_other_robot_targets(pilot):
    manifest = load_manifest(pilot, allow_quality_failures=True)
    arrays = load_episode(pilot, manifest['episodes'][0], 'train')
    before = local_example(arrays, 0, 0)
    arrays['node_features'][0, 1:, 4:6] += 999
    after = local_example(arrays, 0, 0)
    assert set(after) == {'own', 'neighbors', 'target', 'learn'}
    assert after['neighbors'].shape[1] == 4
    for name in ('own', 'neighbors', 'target'): np.testing.assert_array_equal(before[name], after[name])
    after['own'][0] = 999
    assert arrays['node_features'][0, 0, 0] != 999


def test_split_and_integrity_are_enforced(pilot, tmp_path):
    manifest = load_manifest(pilot, allow_quality_failures=True)
    entry = manifest['episodes'][0]
    with pytest.raises(ValueError, match='partition'): load_episode(pilot, entry, 'test')
    folder = tmp_path/'copy'; shutil.copytree(pilot, folder)
    (folder/entry['path']).write_bytes(b'broken')
    with pytest.raises(ValueError, match='corrompue'): load_episode(folder, entry, 'train')


def test_quality_failures_remain_visible_and_require_explicit_inspection(pilot, tmp_path):
    folder = tmp_path/'copy'; shutil.copytree(pilot, folder)
    path = folder/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['quality_status'] = 'review_required'
    manifest['episodes'][0]['quality']['success'] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='examiner'): load_manifest(folder)
    inspected = load_manifest(folder, allow_quality_failures=True)
    assert inspected['episodes'][0]['quality']['success'] is False


def test_circle_loader_rejects_new_schema(pilot):
    from src.imitation.data import load_manifest as load_circle
    with pytest.raises(ValueError, match='version'): load_circle(pilot)


def test_existing_output_is_never_overwritten(pilot):
    with pytest.raises(FileExistsError): collect_dataset(config(), pilot)


def test_default_audit_does_not_load_test(pilot, monkeypatch):
    import src.imitation.multiform_data as data
    original = data.load_episode
    seen = []
    def tracked(dataset, entry, split):
        assert split != 'test'
        seen.append(split)
        return original(dataset, entry, split)
    monkeypatch.setattr(data, 'load_episode', tracked)
    report = verify_dataset(pilot)
    assert report['splits_checked'] == ['train', 'validation']
    assert seen == ['train', 'validation']


def test_extension_preserves_original_archives_and_test(pilot,tmp_path):
    source = load_manifest(pilot)
    original = (pilot/'manifest.json').read_bytes()
    previous = config()
    extended = MultiFormCollectionConfig(previous.shape_splits,
        dict(train=[7100,7101],validation=[7200],test=[7300]),robot_count=4,width=2.)
    output=tmp_path/'extended'
    report=collect_dataset(extended,output,source=pilot)
    assert len(report['episodes'])==4
    assert (pilot/'manifest.json').read_bytes()==original
    for entry in source['episodes']:
        assert (pilot/entry['path']).read_bytes()==(output/entry['path']).read_bytes()
    assert report['derived_from']['policy']=='append_train_validation_preserve_test_archives'


def test_extension_cannot_add_test_seeds(pilot,tmp_path):
    previous=config()
    extended=MultiFormCollectionConfig(previous.shape_splits,
        dict(train=[7100],validation=[7200],test=[7300,7301]),robot_count=4,width=2.)
    with pytest.raises(ValueError,match='conserver test'):
        collect_dataset(extended,tmp_path/'invalid',source=pilot)
    assert not (tmp_path/'invalid').exists()
