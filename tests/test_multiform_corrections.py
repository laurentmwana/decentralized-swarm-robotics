import numpy as np
import pytest
import torch

from src.imitation.multiform_corrections import risk_weights,teacher_labels,load_corrections
from src.imitation.train_multiform_corrections import weighted_mse
from src.reference.shape_control import ShapeGains
from src.imitation.local_target_model import LocalTargetGAT,LocalTargetConfig,save_model,load_model
from src.imitation.multiform_data import sha256
import json


def test_risk_uses_only_visible_neighbors_and_approach():
    edges=np.array([[[2.,0.,-2.,0.],[.1,0.,0.,0.]],[[2.,0.,1.,0.],[.1,0.,0.,0.]]])
    mask=np.array([[True,False],[True,False]])
    np.testing.assert_array_equal(risk_weights(edges,mask),[8,1])
    np.testing.assert_array_equal(risk_weights(edges[:,::-1],mask[:,::-1]),[8,1])


def test_teacher_labels_use_own_goal_and_local_edges_and_skip_classical_phases():
    features=np.zeros((2,1,8));features[:,:,4]=1;features[:,:,7]=1
    arrays=dict(node_features=features,relative_neighbors=np.zeros((2,1,1,4)),adjacency=np.zeros((2,1,1),bool),
        learn_mask=np.array([[True],[False]]),times=np.array([0.,.1]))
    labels,feasible=teacher_labels(arrays,ShapeGains(),2.)
    np.testing.assert_allclose(labels[:,0],[[2,0],[0,0]])
    assert feasible[:,0].tolist()==[True,False]


def test_weighted_loss_and_gradient_have_declared_importance():
    prediction=torch.tensor([[1.,1.],[3.,3.]],requires_grad=True)
    loss=weighted_mse(prediction,torch.zeros_like(prediction),torch.tensor([1.,3.]))
    assert float(loss.detach())==7
    loss.backward();assert prediction.grad[1,0]>prediction.grad[0,0]
    with pytest.raises(ValueError):weighted_mse(prediction,torch.zeros_like(prediction),torch.tensor([0.,1.]))


def test_test_partition_cannot_be_loaded_as_corrections(tmp_path):
    with pytest.raises(ValueError,match='train/validation'):load_corrections(tmp_path,tmp_path,'test')


def test_training_optimizes_and_saves_without_loading_test(tmp_path,monkeypatch):
    import src.imitation.train_multiform_corrections as module
    dataset=tmp_path/'data';dataset.mkdir();(dataset/'manifest.json').write_text('{}')
    source=tmp_path/'source';source.mkdir()
    model=LocalTargetGAT(LocalTargetConfig(hidden_dim=8),2.)
    save_model(source/'best.pt',model)
    (source/'run.json').write_text(json.dumps(dict(checkpoint='best.pt')))
    corrections=tmp_path/'corrections';corrections.mkdir()
    cfg=dict(epochs=2,patience=2,learning_rate=.001,batch_size=2,seed=42,correction_multiplier=3.)
    (corrections/'manifest.json').write_text(json.dumps(dict(training=cfg,source_model_sha256=sha256(source/'best.pt'))))
    arrays=(torch.zeros(3,8),torch.zeros(3,2,4),torch.zeros(3,2,dtype=torch.bool),torch.ones(3,2)*.1)
    monkeypatch.setattr(module,'prepare_data',lambda path:({},dict(train=arrays,validation=arrays),dict(train={},validation={})))
    loaded=[]
    def corrections_for(path,data,split):
        loaded.append(split)
        return tuple(a.numpy() for a in arrays)+(np.ones(3,dtype=np.float32),)
    monkeypatch.setattr(module,'load_corrections',corrections_for)
    report=module.train(dataset,corrections,source,tmp_path/'trained')
    assert loaded==['train','validation'] and report['status']=='trained'
    assert report['epochs_completed']==2 and report['best_epoch'] in (1,2)
    assert load_model(tmp_path/'trained/best.pt').metadata()==model.metadata()
    assert sha256(source/'best.pt')==report['source_model_sha256']
