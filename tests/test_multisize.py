import numpy as np
import pytest
from dataclasses import replace
import json
from src.imitation.multisize import PROFILES,controlled_start,sparse_examples
from src.reference.reconnection import rendezvous_targets
from src.reference.allocation import LocalSlotAllocator,ClaimPacket,Claim
from src.reference.safety import project_acceleration
from src.reference.shapes import named_shape
from src.reference.shape_control import execute_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.imitation.multisize_train import collate_local,LocalSamples
from src.imitation.multiform_data import sha256


@pytest.mark.parametrize('count',[12,24,64])
def test_profiles_fit_rendezvous_and_simple_targets(count):
    profile=PROFILES[count];config=profile.config
    p=controlled_start(profile,8100)
    np.testing.assert_array_equal(p,controlled_start(profile,8100))
    assert np.abs(p).max()<profile.arena_size/2-.1
    rendezvous_targets(count,config,profile.reconnection,.4)
    for shape in ('square','circle','line'):
        named_shape(shape,count,width=profile.width).validate_environment(config,.4)
    assert config.horizon>sum(profile.reconnection.steps(.1))+count**2


def test_cached_packets_do_not_hide_new_claims_or_invalid_messages():
    targets=np.array([[0.,0.],[1.,0.]])
    a=LocalSlotAllocator(1,targets[0],targets);b=LocalSlotAllocator(0,targets[0],targets)
    assert b.broadcast() is b.broadcast()
    a.advance([b.broadcast()]);first=a.broadcast()
    assert a.advance([b.broadcast()])==1 and a.broadcast() is first
    with pytest.raises(ValueError):a.advance([ClaimPacket(0,'wrong',(Claim(0,0,0),))])


def test_feasible_projection_is_exact_and_independent_copy():
    nominal=np.array([.1,.2]);out=project_acceleration(nominal,[[1.,0.]],[-1.],2.)
    np.testing.assert_array_equal(out,nominal);out[:]=99
    np.testing.assert_array_equal(nominal,[.1,.2])


def test_sparse_examples_reconstruct_exact_local_neighbors():
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),horizon=40,communication_radius=3.)
    mission=named_shape('square',4,width=1.)
    result=execute_shape(mission,cfg,mission.reference_points(),safety_filter=True)
    arrays=sparse_examples(result,stride=1)
    assert len(arrays['own'])==24*4 and arrays['offsets'][-1]==len(arrays['edges'])
    for sample in (0,len(arrays['own'])-1):
        step,robot=arrays['steps'][sample],arrays['robots'][sample]
        visible=result.episode.adjacency[step,robot]
        expected=result.episode.positions[step,visible]-result.episode.positions[step,robot]
        a,b=arrays['offsets'][sample:sample+2]
        np.testing.assert_allclose(arrays['edges'][a:b,:2],expected,atol=1e-7)
        assert arrays['own'][sample,6]==3


def test_collation_masks_padding_and_handles_isolated_robot():
    own=np.zeros(8,np.float32);label=np.zeros(2,np.float32)
    batch=collate_local([(own,np.empty((0,4),np.float32),label),(own,np.ones((2,4),np.float32),label)])
    assert batch[1].shape==(2,2,4)
    assert batch[2].tolist()==[[False,False],[True,True]]
    assert not batch[1][0].any()


def test_csr_loader_balances_counts_and_never_loads_other_partition(tmp_path,monkeypatch):
    entries=[]
    for split in ('train','validation'):
        for count in (12,24,64):
            own=np.zeros((3,8),np.float32);own[:,7]=count;own[:,0]=0 if split=='train' else 99
            name=f'{split}-{count}.npz'
            np.savez_compressed(tmp_path/name,own=own,labels=np.zeros((3,2),np.float32),edges=np.ones((3,4),np.float32),offsets=np.arange(4))
            entries.append(dict(split=split,count=count,shape=split,samples=name,sha256=sha256(tmp_path/name)))
    manifest=dict(kind='multi_size_local_target_imitation',status='complete',quality_status='passed',episodes=entries,
                  shape_splits=dict(train=['train'],validation=['validation']))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    original=np.load;loaded=[]
    def tracked(path,**kwargs):loaded.append(path.name);return original(path,**kwargs)
    monkeypatch.setattr(np,'load',tracked)
    dataset=LocalSamples(tmp_path,'train',42,2)
    assert len(dataset)==6 and all(v['selected']==2 for v in dataset.selection.values())
    assert all(name.startswith('train-') for name in loaded)
    with pytest.raises(ValueError):LocalSamples(tmp_path,'test',42,2)


def test_safety_cache_invalidates_when_observed_obstacle_changes(monkeypatch):
    import src.reference.shape_control as control
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),horizon=20,communication_radius=3.)
    mission=named_shape('square',4,width=1.);calls=[];original=control.filter_command
    def tracked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(control,'filter_command',tracked)
    execute_shape(mission,cfg,mission.reference_points(),safety_filter=True)
    cached=len(calls);calls.clear();counter=[0]
    def sensor(position):
        counter[0]+=1
        return np.array([[20.+counter[0]*.01,0.,.2]])
    execute_shape(mission,cfg,mission.reference_points(),safety_filter=True,obstacle_observer=sensor)
    # Sans reconnexion, les 16 premiers pas sont l'allocation immobile.
    assert cached<4*4 and len(calls)==4*4


def test_mixed_count_training_saves_model_and_balances_examples(tmp_path):
    from src.imitation.multisize_train import train
    from src.imitation.local_target_model import load_model
    dataset=tmp_path/'dataset';dataset.mkdir();entries=[]
    for split in ('train','validation'):
        for count in (12,24,64):
            own=np.zeros((3,8),np.float32);own[:,7]=count
            name=f'{split}-{count}.npz'
            np.savez_compressed(dataset/name,own=own,labels=np.zeros((3,2),np.float32),edges=np.ones((3,4),np.float32),offsets=np.arange(4))
            entries.append(dict(split=split,count=count,shape=split,samples=name,sha256=sha256(dataset/name)))
    manifest=dict(kind='multi_size_local_target_imitation',status='complete',quality_status='passed',episodes=entries,
        shape_splits=dict(train=['train'],validation=['validation']),
        model=dict(hidden_dim=8,spatial_scale=28.,speed_scale=1.,count_scale=64.,init_seed=42),
        training=dict(epochs=2,batch_size=3,learning_rate=.001,patience=2,seed=42,max_samples_per_count=2))
    (dataset/'manifest.json').write_text(json.dumps(manifest))
    report=train(dataset,tmp_path/'run')
    assert report['status']=='trained' and report['epochs_completed']==2
    assert all(entry['selected']==2 for entry in report['selection']['train'].values())
    assert load_model(tmp_path/'run/best.pt').config.count_scale==64.
