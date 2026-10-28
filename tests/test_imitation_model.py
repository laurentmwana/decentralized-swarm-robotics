"""Tests de calcul et dérivées uniquement : aucun optimiseur ni entraînement."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from src.imitation.checkpoint import load_model, save_model
from src.imitation.model import AccelerationGAT, GATConfig
from src.imitation.objective import acceleration_mse


def config():
    return GATConfig.from_yaml('config/imitation-model.yaml')


def model():
    return AccelerationGAT(config(), 2.)


def features(n=4):
    data = torch.zeros(n,8)
    data[:,0] = torch.arange(n)*.4
    data[:,1] = .3
    data[:,6] = 2.
    data[:,7] = n
    return data


def chain(n=4):
    indices = torch.arange(n)
    return (indices[:,None]-indices[None,:]).abs() == 1


def test_shapes_bounds_and_attention_mask():
    net=model()
    command, weights=net(features(),chain(),return_attention=True)
    assert command.shape == (4,2)
    assert torch.isfinite(command).all()
    assert (torch.linalg.vector_norm(command,dim=-1) <= 2.+1e-6).all()
    assert (weights[~chain()] == 0).all()
    torch.testing.assert_close(weights.sum(-1),torch.ones(4))
    assert len(net.encoder) == 4 and len(net.decoder) == 3


@pytest.mark.parametrize('n',[1,3,12,40])
def test_isolated_and_variable_size_swarm(n):
    output, attention=model()(features(n),torch.zeros(n,n,dtype=torch.bool),return_attention=True)
    assert output.shape == (n,2) and torch.isfinite(output).all()
    assert not attention.any()


def test_exactly_one_hop_and_no_distant_influence():
    net=model()
    data=features()
    graph=chain()
    initial=net(data,graph)
    changed=data.clone(); changed[2:,:4] += 100
    after=net(changed,graph)
    torch.testing.assert_close(initial[0],after[0],atol=0,rtol=0)
    # Le voisin de 2 reçoit bien son changement : le masque ne coupe pas tout échange.
    assert not torch.allclose(initial[1],after[1])


def test_neighbor_influence_and_permutation_equivariance():
    net=model()
    data=features(); graph=chain()
    command=net(data,graph)
    permutation=torch.tensor([2,0,3,1])
    permuted=net(data[permutation],graph[permutation][:,permutation])
    torch.testing.assert_close(permuted,command[permutation],atol=1e-6,rtol=1e-6)
    changed=data.clone(); changed[1,0]+=2
    assert not torch.allclose(net(changed,graph)[0],command[0])


def test_translation_invariance_with_shared_center():
    net=model(); data=features()
    moved=data.clone(); moved[:,:2] += torch.tensor([5.,-3.]); moved[:,4:6] += torch.tensor([5.,-3.])
    torch.testing.assert_close(net(data,chain()),net(moved,chain()),atol=1e-6,rtol=1e-6)


def test_batched_and_individual_forward_match():
    net=model(); batch=torch.stack([features(),features()+.1]); graphs=torch.stack([chain(),chain()])
    actual=net(batch,graphs)
    expected=torch.stack([net(batch[i],graphs[i]) for i in range(2)])
    torch.testing.assert_close(actual,expected)


def test_initialization_is_reproducible_and_preserves_global_rng():
    before=torch.random.get_rng_state().clone()
    first=model(); after=torch.random.get_rng_state()
    assert torch.equal(before,after)
    second=model()
    for a,b in zip(first.parameters(),second.parameters()):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    other=AccelerationGAT(replace(config(),init_seed=43),2.)
    assert any(not torch.equal(a,b) for a,b in zip(first.parameters(),other.parameters()))


def test_finite_derivatives_for_isolated_nodes_without_updating_weights():
    net=model(); original={k:v.clone() for k,v in net.state_dict().items()}
    data=features().requires_grad_()
    prediction=net(data,torch.zeros(4,4,dtype=torch.bool))
    loss=acceleration_mse(prediction,torch.zeros_like(prediction))
    gradients=torch.autograd.grad(loss,tuple(net.parameters()),allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    for key,value in net.state_dict().items():
        torch.testing.assert_close(value,original[key],atol=0,rtol=0)


def test_derivative_distant_node_is_zero():
    net=model(); data=features().requires_grad_()
    derivatives=torch.autograd.grad(net(data,chain())[0].sum(),data)[0]
    assert torch.count_nonzero(derivatives[2:]) == 0
    assert torch.count_nonzero(derivatives[1]) > 0


def test_mse_matches_known_value():
    prediction=torch.tensor([[1.,2.],[3.,4.]])
    assert acceleration_mse(prediction,torch.zeros_like(prediction)).item() == 7.5
    assert acceleration_mse(prediction,prediction).item() == 0


@pytest.mark.parametrize('case',['shape','integer','nan','empty','dtype'])
def test_invalid_mse_inputs(case):
    prediction=torch.zeros(2,2); target=prediction.clone()
    if case=='shape': target=torch.zeros(1,2)
    elif case=='integer': target=torch.zeros(2,2,dtype=torch.int64)
    elif case=='nan': target[0,0]=float('nan')
    elif case=='empty': prediction=target=torch.empty(0,2)
    else: target=target.double()
    with pytest.raises(ValueError): acceleration_mse(prediction,target)


@pytest.mark.parametrize('case',['shape','nan','integer','graph_shape','graph_dtype','self_loop'])
def test_invalid_graph_or_features(case):
    data=features(); graph=chain()
    if case=='shape': data=torch.zeros(4,7)
    elif case=='nan': data[0,0]=float('nan')
    elif case=='integer': data=data.long()
    elif case=='graph_shape': graph=graph[:2]
    elif case=='graph_dtype': graph=graph.float()
    else: graph[0,0]=True
    with pytest.raises(ValueError): model()(data,graph)


@pytest.mark.parametrize('change',[dict(hidden_dim=0),dict(hidden_dim=True),dict(spatial_scale=0),
    dict(speed_scale=float('nan')),dict(count_scale=-1),dict(negative_slope=1),dict(init_seed=-1)])
def test_invalid_configuration(change):
    with pytest.raises(ValueError): replace(config(),**change)


def test_checkpoint_roundtrip_and_overwrite_refusal(tmp_path):
    net=model(); path=tmp_path/'untrained.pt'
    save_model(path,net)
    restored=load_model(path)
    torch.testing.assert_close(restored(features(),chain()),net(features(),chain()),atol=0,rtol=0)
    with pytest.raises(FileExistsError): save_model(path,net)


def test_legacy_checkpoint_rejected(tmp_path):
    path=tmp_path/'legacy.pt'; torch.save({'state_dict':{}},path)
    with pytest.raises(ValueError,match='incompatible'): load_model(path)


def test_real_episode_arrays_forward_and_mse_without_training():
    from src.imitation.data import episode_arrays
    from src.reference.circle import CircleTeacher
    from src.simulator.acceleration_episode import AccelerationEpisode,AccelerationEpisodeConfig
    cfg=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    cfg=replace(cfg,horizon=2)
    teacher=CircleTeacher.from_yaml('config/circle-teacher.yaml',cfg)
    result=AccelerationEpisode(cfg).execute(teacher,12,4)
    arrays=episode_arrays(result,teacher.mission)
    prediction=model()(torch.tensor(arrays['node_features'],dtype=torch.float32),torch.tensor(arrays['adjacency']))
    loss=acceleration_mse(prediction,torch.tensor(arrays['targets'],dtype=torch.float32))
    assert loss.ndim==0 and torch.isfinite(loss)


def test_output_saturation_uses_vector_norm():
    net=model()
    with torch.no_grad():
        net.decoder[-1].weight.zero_()
        net.decoder[-1].bias.copy_(torch.tensor([3.,4.]))
    output=net(features(1),torch.zeros(1,1,dtype=torch.bool))
    torch.testing.assert_close(output,torch.tensor([[1.2,1.6]]))


def test_empty_and_dense_neighbors_have_finite_derivatives():
    net=model(); data=features().requires_grad_(); graph=chain()
    graph[0]=False; graph[:,0]=False
    prediction=net(data,graph)
    derivatives=torch.autograd.grad(prediction.square().mean(),tuple(net.parameters()))
    assert all(torch.isfinite(g).all() for g in derivatives)
