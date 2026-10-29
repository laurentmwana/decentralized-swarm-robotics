"""Diagnostic analytique et inférence uniquement, aucun entraînement."""
import hashlib
import json

import numpy as np
import pytest
import torch

from src.imitation.diagnose import sample_statistics, aggregate_statistics, diagnose_run
from src.imitation.checkpoint import save_model
from src.imitation.model import AccelerationGAT, GATConfig
from test_imitation_integration import source, reviewed


def test_radial_tangential_and_safety_statistics():
    features = np.zeros((1, 2, 8))
    features[0, :, :2] = [[1, 0], [1.2, 0]]
    features[..., 6] = 1
    arrays = dict(node_features=features, adjacency=np.array([[[False, True], [True, False]]]),
                  targets=np.zeros((1, 2, 2)))
    records = sample_statistics(arrays, np.array([[[3., 4.], [3., 4.]]]), .4)
    assert records['near'].all()
    report = aggregate_statistics([records], .1)
    assert report['overall']['mse'] == 12.5
    assert report['overall']['radial_squared'] == 9
    assert report['overall']['tangential_squared'] == 16
    assert report['by_safety']['outside_safety']['samples'] == 0
    assert report['by_safety']['outside_safety']['mse'] is None
    assert report['by_degree']['1']['samples'] == 2


def test_time_groups_do_not_mix_episode_boundaries():
    shape = (100, 1)
    record = {k: np.ones(shape) for k in ('mse', 'radial_squared', 'tangential_squared', 'target_norm', 'degree', 'radius_error', 'speed')}
    record['near'] = np.zeros(shape, dtype=bool)
    report = aggregate_statistics([record, record], .1)
    assert [r['samples'] for r in report['by_time'].values()] == [40, 60, 100]


def test_diagnostic_never_loads_test_or_creates_optimizer(reviewed, tmp_path, monkeypatch):
    import src.imitation.diagnose as diagnostic
    net = AccelerationGAT(GATConfig.from_yaml('config/imitation-model.yaml'), 2.)
    run = tmp_path/'run'; run.mkdir(); save_model(run/'best.pt', net)
    (run/'run.json').write_text(json.dumps(dict(status='trained', checkpoint='best.pt', model=net.metadata(),
        dataset_manifest_sha256=hashlib.sha256((reviewed/'manifest.json').read_bytes()).hexdigest())))
    original = diagnostic.load_episode
    seen = []
    def tracked(dataset, entry, split):
        assert split in ('train', 'validation')
        seen.append(split)
        return original(dataset, entry, split)
    monkeypatch.setattr(diagnostic, 'load_episode', tracked)
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('Optimiseur interdit'))
    before = (run/'best.pt').read_bytes()
    report = diagnose_run(run, reviewed, tmp_path/'diagnostic')
    assert seen == ['train', 'validation']
    assert [r['seed'] for r in report['validation_rollouts']] == [3]
    assert (run/'best.pt').read_bytes() == before
    assert report['teacher_states']['train']['overall']['samples'] == 36
    assert np.isfinite(report['validation_gat_states']['overall']['mse'])
