"""Localité, calcul et dérivées du nouveau GAT ; aucune optimisation réelle."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.imitation.local_target_model import LocalTargetConfig, LocalTargetGAT, LocalTargetController, save_model, load_model
from src.imitation.local_target_train import prepare_data, run_training
from src.imitation.train import TrainingConfig
from src.simulator.acceleration_episode import LocalObservation
from test_multiform_data import pilot


def scene(k=3, batch=2):
    own = torch.zeros(batch, 8); own[:, 4] = 1; own[:, 6] = 3; own[:, 7] = 12
    neighbors = torch.zeros(batch, k, 4)
    if k: neighbors[..., 0] = torch.arange(1, k+1)*.3
    mask = torch.ones(batch, k, dtype=torch.bool)
    return own, neighbors, mask


@pytest.mark.parametrize('k', [0, 1, 4, 12])
def test_bounds_isolation_batches_and_finite_gradients(k):
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(k)
    own.requires_grad_(); neighbors.requires_grad_()
    output, attention = model(own, neighbors, mask, return_attention=True)
    assert output.shape == (2, 2)
    assert attention.shape == (2, k)
    assert torch.isfinite(output).all()
    assert (torch.linalg.vector_norm(output, dim=-1) <= 2+1e-6).all()
    before = {key:value.clone() for key, value in model.state_dict().items()}
    gradients = torch.autograd.grad(output.square().mean(), tuple(model.parameters()), allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    for key,value in model.state_dict().items(): torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    torch.testing.assert_close(model(own[0], neighbors[0], mask[0]), output[0])


def test_masked_neighbors_are_ignored_even_with_nan_padding():
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(4)
    mask[:, 2:] = False
    before = model(own, neighbors, mask)
    neighbors[:, 2:] = float('nan')
    torch.testing.assert_close(model(own, neighbors, mask), before, atol=0, rtol=0)
    assert not model(own, neighbors, mask, return_attention=True)[1][:, 2:].any()


def test_neighbor_permutation_and_global_translation_invariance():
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(4)
    before = model(own, neighbors, mask)
    torch.testing.assert_close(model(own, neighbors.flip(-2), mask.flip(-1)), before)
    translated = own.clone(); translated[:, :2] += 5; translated[:, 4:6] += 5
    torch.testing.assert_close(model(translated, neighbors, mask), before, atol=1e-7, rtol=1e-6)


def test_derivative_of_masked_neighbor_is_zero():
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(3)
    mask[:, 2] = False; neighbors.requires_grad_()
    gradient = torch.autograd.grad(model(own, neighbors, mask).sum(), neighbors)[0]
    assert not gradient[:, 2].any()
    assert gradient[:, :2].abs().sum() > 0


def test_neighbor_target_features_are_not_accepted():
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(3)
    with pytest.raises(ValueError, match='physiques'):
        model(own, torch.zeros(2, 3, 8), mask)


def test_controller_matches_local_forward_without_private_neighbor_goals():
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    own, neighbors, mask = scene(3, 1)
    observation = LocalObservation(own[0, :2].numpy(), own[0, 2:4].numpy(), neighbors[0, :, :2].numpy(),
                                   neighbors[0, :, 2:].numpy(), 0.)
    controller = LocalTargetController(model)
    np.testing.assert_allclose(controller(observation, [1., 0.], 3, 12), model(own, neighbors, mask)[0].detach().numpy(), atol=1e-6)


def test_checkpoint_roundtrip_and_circle_checkpoint_rejection(tmp_path):
    model = LocalTargetGAT(LocalTargetConfig(), 2.)
    path = tmp_path/'local.pt'; save_model(path, model)
    torch.testing.assert_close(load_model(path)(*scene()), model(*scene()), atol=0, rtol=0)
    with pytest.raises(FileExistsError): save_model(path, model)
    wrong = tmp_path/'wrong.pt'; torch.save(dict(metadata=dict(model_type='AccelerationGAT'), state_dict={}), wrong)
    with pytest.raises(ValueError, match='incompatible'): load_model(wrong)


def test_prepare_data_omits_test_and_duplicate_aliases(monkeypatch):
    import src.imitation.local_target_train as training
    manifest = dict(episodes=[dict(split='train',path='train/a',duplicate_within_split_of=None),
        dict(split='train',path='train/alias',duplicate_within_split_of='train/a'),
        dict(split='validation',path='validation/b',duplicate_within_split_of=None),
        dict(split='test',path='test/c',duplicate_within_split_of=None)])
    monkeypatch.setattr(training, 'load_manifest', lambda _: manifest)
    seen = []
    def episode(dataset, entry, split):
        assert split != 'test' and entry['path'] != 'train/alias'
        seen.append(entry['path'])
        return dict(learn_mask=np.array([[True, False]]),node_features=np.zeros((1,2,8)),
            relative_neighbors=np.zeros((1,2,2,4)),adjacency=np.zeros((1,2,2),dtype=bool),targets=np.zeros((1,2,2)))
    monkeypatch.setattr(training,'load_episode',episode)
    _,arrays,selection=prepare_data('unused')
    assert seen == ['train/a','validation/b']
    assert selection['train']['samples'] == 1 and len(arrays['train'][0]) == 1


def test_dry_run_creates_no_optimizer_or_weights(pilot, tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k: pytest.fail('Optimiseur interdit en dry-run'))
    report=run_training(pilot,tmp_path/'unused',LocalTargetConfig(),TrainingConfig(2,32,.001,1,42),dry_run=True)
    assert report['status']=='dry_run'
    assert set(report['selection']) == {'train','validation'}
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('change',[dict(hidden_dim=0),dict(init_seed=True),dict(spatial_scale=0)])
def test_invalid_configuration(change):
    with pytest.raises(ValueError): replace(LocalTargetConfig(),**change)
