import numpy as np
import pytest
import torch
from src.imitation.vector_gat import VectorGAT,VectorConfig,save_vector,load_vector,barrier_penalty


def inputs():
    own=torch.tensor([[1.,2.,.1,.2,3.,5.,3.,24.]])
    edges=torch.tensor([[[.6,.1,-.2,0.],[1.,-.2,0.,.1]]]);mask=torch.ones(1,2,dtype=torch.bool)
    return own,edges,mask


def test_exact_equilibrium_and_no_qp(monkeypatch):
    import src.reference.safety as safety
    monkeypatch.setattr(safety,'filter_command',lambda *a,**k:pytest.fail('Filtre caché'))
    model=VectorGAT(VectorConfig(8));own,edges,mask=inputs();own[:,4:6]=own[:,:2];own[:,2:4]=0;edges[:,:,2:]=0
    with torch.no_grad():model.coefficients[-1].bias.fill_(2.)
    torch.testing.assert_close(model(own,edges,mask),torch.zeros(1,2),atol=0,rtol=0)


def test_rotation_translation_and_neighbor_permutation():
    model=VectorGAT(VectorConfig(8));own,edges,mask=inputs()
    with torch.no_grad():model.coefficients[-1].weight.normal_(0,.1)
    expected=model(own,edges,mask);R=torch.tensor([[0.,-1.],[1.,0.]])
    transformed=own.clone()
    for sl in (slice(0,2),slice(2,4),slice(4,6)):transformed[:,sl]=own[:,sl]@R.T
    transformed[:,:2]+=5;transformed[:,4:6]+=5
    rotated=edges.clone();rotated[:,:,:2]=edges[:,:,:2]@R.T;rotated[:,:,2:]=edges[:,:,2:]@R.T
    torch.testing.assert_close(model(transformed,rotated,mask),expected@R.T,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(model(own,edges.flip(1),mask.flip(1)),expected)


def test_bounded_isolated_save_and_gradients(tmp_path):
    model=VectorGAT(VectorConfig(8));own,edges,mask=inputs();mask[:]=False
    command=model(own,edges,mask);assert torch.linalg.vector_norm(command)<=2.000001
    command.square().sum().backward();assert model.log_kp.grad is not None
    save_vector(tmp_path/'model.pt',model);loaded=load_vector(tmp_path/'model.pt')
    torch.testing.assert_close(loaded(own,edges,mask),model(own,edges,mask))
    assert barrier_penalty(command,edges,mask)==0


def test_training_margin_and_empty_neighbor_list():
    model=VectorGAT(VectorConfig(8));own,_,_=inputs()
    prediction=model(own,torch.empty(1,0,4),torch.empty(1,0,dtype=torch.bool))
    assert torch.isfinite(prediction).all()
    edges=torch.tensor([[[.5,0.,-.4,0.]]]);mask=torch.ones(1,1,dtype=torch.bool)
    assert barrier_penalty(torch.zeros(1,2),edges,mask,.5)>barrier_penalty(torch.zeros(1,2),edges,mask,.4)


def test_swept_minimum_uses_last_partial_block():
    from src.imitation.audit_vector import minimum_separation
    p=np.zeros((132,2,2));p[:,0,0]=np.linspace(-1.,1.,132);p[:,1,1]=.3
    assert minimum_separation(p)==pytest.approx(.3)
