"""Contrats de la variante relative ; dérivées sans mise à jour de poids."""
from dataclasses import replace

import pytest
import torch

from src.imitation.model import AccelerationGAT, GATConfig, OneHopAttention
from src.imitation.checkpoint import save_model, load_model


def config():
    return GATConfig.from_yaml('config/imitation-model-relative.yaml')


def scene(n):
    data = torch.zeros(n, 8)
    data[:, 0] = torch.arange(n)*.4
    data[:, 1] = .3
    data[:, 6] = 2
    data[:, 7] = n
    indices = torch.arange(n)
    graph = (indices[:, None]-indices[None, :]).abs() == 1
    return data, graph


def test_relative_scores_can_rank_shared_neighbors_by_receiver_distance():
    encoded = torch.zeros(4, 8)
    graph = torch.zeros(4, 4, dtype=torch.bool)
    graph[:2, 2:] = True
    edges = torch.zeros(4, 4, 5)
    edges[0, 2:, 4] = torch.tensor([.1, .9])
    edges[1, 2:, 4] = torch.tensor([.9, .1])
    old = OneHopAttention(8, .2)
    new = OneHopAttention(8, .2, True)
    with torch.no_grad():
        new.edge_score.weight.zero_()
        new.edge_score.weight[0, 4] = -1
    _, old_weights = old(encoded, graph)
    _, new_weights = new(encoded, graph, edges)
    torch.testing.assert_close(old_weights[0], old_weights[1])
    assert new_weights[0, 2] > new_weights[0, 3]
    assert new_weights[1, 2] < new_weights[1, 3]


@pytest.mark.parametrize('n', [1, 4, 12])
def test_relative_locality_derivatives_permutation_and_bounds(n):
    net = AccelerationGAT(config(), 2.)
    data, graph = scene(n)
    data.requires_grad_()
    original = {k: v.clone() for k, v in net.state_dict().items()}
    result = net(data, graph)
    derivatives = torch.autograd.grad(result[0].sum(), data, retain_graph=True)[0]
    assert torch.isfinite(derivatives).all()
    if n > 2:
        assert derivatives[2:].count_nonzero() == 0
        changed = data.detach().clone(); changed[2:, :4] += 100
        torch.testing.assert_close(net(changed, graph)[0], result[0], atol=0, rtol=0)
    parameter_gradients = torch.autograd.grad(result.square().mean(), tuple(net.parameters()))
    assert all(torch.isfinite(g).all() for g in parameter_gradients)
    assert (torch.linalg.vector_norm(result, dim=-1) <= 2+1e-6).all()
    permutation = torch.arange(n-1, -1, -1)
    torch.testing.assert_close(net(data[permutation], graph[permutation][:, permutation]), result[permutation])
    shifted = data.detach().clone(); shifted[:, :2] += 3; shifted[:, 4:6] += 3
    torch.testing.assert_close(net(shifted, graph), result, atol=1e-6, rtol=1e-5)
    batch = net(data.unsqueeze(0).expand(2, -1, -1), graph.unsqueeze(0).expand(2, -1, -1))
    torch.testing.assert_close(batch[0], result)
    for k, v in net.state_dict().items():
        torch.testing.assert_close(v, original[k], atol=0, rtol=0)


@pytest.mark.parametrize('relative', [False, True])
def test_both_checkpoint_versions_roundtrip(relative, tmp_path):
    net = AccelerationGAT(replace(config(), relative_edges=relative), 2.)
    path = tmp_path/'net.pt'; save_model(path, net)
    restored = load_model(path)
    data, graph = scene(4)
    assert net.metadata()['schema_version'] == (2 if relative else 1)
    assert ('relative_edges' in net.metadata()['config']) == relative
    torch.testing.assert_close(restored(data, graph), net(data, graph), atol=0, rtol=0)


def test_mismatched_checkpoint_version_rejected(tmp_path):
    net = AccelerationGAT(config(), 2.)
    metadata = net.metadata(); metadata['schema_version'] = 1
    path = tmp_path/'wrong.pt'; torch.save(dict(metadata=metadata, state_dict=net.state_dict()), path)
    with pytest.raises(ValueError, match='Version'):
        load_model(path)
