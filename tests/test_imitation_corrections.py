"""Corrections train et provenance ; pas d'optimisation dans ces tests."""
import hashlib
import json

import numpy as np
import pytest
import torch

from src.imitation.checkpoint import save_model, load_model
from src.imitation.corrections import collect_corrections, load_corrections
from src.imitation.data import load_manifest
from src.imitation.model import AccelerationGAT, GATConfig
from src.imitation.train import prepare_data, run_training, TrainingConfig
from test_imitation_integration import source, reviewed


@pytest.fixture
def run(reviewed, tmp_path):
    model = AccelerationGAT(GATConfig.from_yaml('config/imitation-model-relative.yaml'), 2.)
    folder = tmp_path/'run'; folder.mkdir(); save_model(folder/'best.pt', model)
    (folder/'run.json').write_text(json.dumps(dict(status='trained', checkpoint='best.pt', model=model.metadata(),
        dataset_manifest_sha256=hashlib.sha256((reviewed/'manifest.json').read_bytes()).hexdigest())))
    return folder


@pytest.fixture
def corrections(run, reviewed, tmp_path):
    output = tmp_path/'corrections'
    collect_corrections(run, reviewed, output)
    return output


def test_collection_never_loads_validation_or_test_and_keeps_failed_rollouts(run, reviewed, tmp_path, monkeypatch):
    import src.imitation.corrections as correction
    original = correction.load_episode
    seen = []
    def tracked(dataset, entry, split):
        assert split == 'train'
        seen.append(entry['seed'])
        return original(dataset, entry, split)
    monkeypatch.setattr(correction, 'load_episode', tracked)
    output = tmp_path/'corrections'
    report = collect_corrections(run, reviewed, output)
    assert seen == [1]
    assert [e['seed'] for e in report['episodes']] == [1]
    assert report['episodes'][0]['collector_quality']['success'] is False
    with np.load(output/'train-1.npz') as archive:
        assert archive['targets'].shape == archive['collector_commands'].shape
        assert not np.array_equal(archive['targets'], archive['collector_commands'])


def test_prepare_data_only_augments_train(corrections, reviewed):
    _, before = prepare_data(reviewed)
    _, after = prepare_data(reviewed, corrections=corrections)
    assert len(after['train'][0]) == 2*len(before['train'][0])
    for left, right in zip(before['validation'], after['validation']):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


@pytest.mark.parametrize('change', ['split', 'parent', 'fingerprint', 'corrupt', 'incomplete'])
def test_invalid_corrections_refused(corrections, reviewed, change):
    path = corrections/'manifest.json'
    report = json.loads(path.read_text())
    if change == 'split': report['episodes'][0]['split'] = 'validation'
    elif change == 'parent': report['episodes'][0]['parent_archive_sha256'] = 'different'
    elif change == 'fingerprint': report['dataset_manifest_sha256'] = 'different'
    elif change == 'corrupt': (corrections/'train-1.npz').write_bytes(b'invalid')
    else: report['status'] = 'incomplete'
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_corrections(corrections, reviewed, load_manifest(reviewed))


def test_warm_start_dry_run_preserves_source_and_creates_no_optimizer(corrections, reviewed, run, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('Optimisation interdite'))
    checkpoint = run/'best.pt'; before = checkpoint.read_bytes()
    output = tmp_path/'unused'
    report = run_training(reviewed, output, GATConfig.from_yaml('config/imitation-model-relative.yaml'),
        TrainingConfig(2, 2, .0003, 1, 42), dry_run=True, corrections=corrections, initial_checkpoint=checkpoint)
    assert report['graphs'] == dict(train=6, validation=3)
    assert report['initial_checkpoint_sha256'] == hashlib.sha256(before).hexdigest()
    assert checkpoint.read_bytes() == before
    assert not output.exists()


def test_warm_start_rejects_different_architecture(reviewed, run, tmp_path):
    with pytest.raises(ValueError, match='initial incompatible'):
        run_training(reviewed, tmp_path/'unused', GATConfig.from_yaml('config/imitation-model.yaml'),
            TrainingConfig(2, 2, .0003, 1, 42), dry_run=True, initial_checkpoint=run/'best.pt')


def test_initial_checkpoint_remains_best_if_validation_worsens(reviewed, run, tmp_path, monkeypatch):
    import src.imitation.train as training
    class NoUpdate:
        def zero_grad(self, **kwargs): pass
        def step(self): pass
    # Pas de mise à jour : vérifie seulement la sélection et les fichiers.
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: NoUpdate())
    scores = iter([.1, .2])
    monkeypatch.setattr(training, 'validation_mse', lambda *a, **k: next(scores))
    output = tmp_path/'selection'
    report = run_training(reviewed, output, GATConfig.from_yaml('config/imitation-model-relative.yaml'),
        TrainingConfig(2, 32, .0003, 1, 42), initial_checkpoint=run/'best.pt')
    assert report['best_epoch'] == 0
    assert report['best_validation_mse'] == .1
    assert report['status'] == 'trained'
    expected, actual = load_model(run/'best.pt'), load_model(output/'best.pt')
    for left, right in zip(expected.parameters(), actual.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
