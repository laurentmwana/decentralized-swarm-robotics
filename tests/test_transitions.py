"""Continuité physique, versions de mission et contacts aux frontières."""
from dataclasses import replace

import numpy as np
import pytest

from src.reference.allocation import Claim,ClaimPacket,LocalSlotAllocator
from src.reference.metrics import collision_events
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import ShapeGains
from src.reference.shapes import named_shape
from src.reference.transitions import FormationStage, TransitionProtocol, execute_sequence, evaluate_sequence
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def config():
    return AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')


def test_velocity_position_and_clock_survive_a_change_during_motion():
    mission=named_shape('square',4,width=2.)
    initial=mission.reference_points()*.7
    velocity=np.tile([.1,0.],(4,1))
    times=[]
    def policy(observation,target,phase,count):
        times.append(observation.time)
        return np.array([.2,0.])
    result=execute_sequence([FormationStage(mission,20),FormationStage(named_shape('rectangle',4,width=2.),20)],
        config(),initial,initial_velocities=velocity,motion_policy=policy,guard_motion=False)
    first,second=result.segments
    assert np.linalg.norm(first.episode.velocities[-1])>0
    np.testing.assert_array_equal(first.episode.velocities[-1],second.episode.velocities[0])
    np.testing.assert_array_equal(first.episode.positions[-1],second.episode.positions[0])
    np.testing.assert_array_equal(result.episode.velocities[0],velocity)
    assert result.episode.positions.shape==(41,4,2)
    np.testing.assert_allclose(np.diff(result.episode.times),.1)
    assert np.all(np.diff(times)>=-1e-12)
    assert any(t>=2.1 for t in times)
    assert evaluate_sequence(result)['boundary_states_preserved']


def test_repeated_shape_gets_new_epoch_and_delayed_claims_are_discarded():
    a=named_shape('A',4,width=2.)
    b=named_shape('square',4,width=2.)
    result=execute_sequence([FormationStage(a,30),FormationStage(b,30),FormationStage(a,30)],
        config(),a.reference_points(),motion_policy=lambda *args:np.zeros(2))
    epochs=[segment.final_packets[0].mission for segment in result.segments]
    assert len(set(epochs))==3
    assert result.stale_messages_per_step[:30].sum()==0
    assert result.stale_messages_per_step[30:60].sum()>0
    assert result.stale_messages_per_step[60:].sum()>0
    assert len(np.unique(result.segments[-1].assignments[-1]))==4


def test_old_claim_on_same_geometry_cannot_reenter_later_occurrence():
    points=named_shape('square',4,width=1.).reference_points()
    previous=LocalSlotAllocator(0,points[0],points,mission_epoch='stage:0').broadcast()
    previous=ClaimPacket(previous.sender,previous.mission,(Claim(1,9999,0),))
    current=LocalSlotAllocator(1,points[1],points,mission_epoch='stage:2')
    before=current.broadcast()
    _,discarded=current.advance_current_mission([previous])
    assert discarded==1
    assert current.broadcast()==before


def test_new_mission_preferences_use_current_own_position_not_old_slot():
    points=named_shape('square',4,width=1.).reference_points()
    old=LocalSlotAllocator(0,points[0],points,mission_epoch='stage:0')
    new=LocalSlotAllocator(0,points[2],points,mission_epoch='stage:1')
    assert old.slot==0 and new.slot==2


def test_contact_across_boundary_is_counted_once_globally():
    mission=named_shape('square',4,width=1.)
    positions=np.array([[0.,0.],[0.,0.],[1.,0.],[0.,1.]])
    result=execute_sequence([FormationStage(mission,20),FormationStage(mission,20)],config(),positions,
        motion_policy=lambda *args:np.zeros(2))
    report=evaluate_sequence(result)
    assert report['collisions_total']==collision_events(result.episode.positions,.1)[0]==1
    assert [s['collisions'] for s in report['stages']]==[1,0]
    assert all(s['contact_present'] for s in report['stages'])
    assert not report['success']


def test_too_short_mission_is_preserved_as_failure_without_time_imputation():
    mission=named_shape('square',4,width=1.)
    result=execute_sequence([FormationStage(mission,5),FormationStage(mission,5)],config(),mission.reference_points(),
        safety_filter=True,reconnection=ReconnectionConfig())
    report=evaluate_sequence(result)
    assert len(report['stages'])==2
    assert report['elapsed_time_s']==pytest.approx(1.)
    assert not report['success']
    assert all(s['stabilization_time_s'] is None and not s['allocation_resolved'] for s in report['stages'])
    assert (result.assignments==-1).all()


@pytest.mark.parametrize('case',['empty','steps','count','geometry','velocity'])
def test_invalid_sequence_contract_is_rejected(case):
    mission=named_shape('square',4,width=1.)
    with pytest.raises(ValueError):
        if case=='empty':execute_sequence([],config(),mission.reference_points())
        elif case=='steps':FormationStage(mission,0)
        elif case=='count':execute_sequence([FormationStage(mission,50),FormationStage(named_shape('A',5,width=2.),50)],config(),mission.reference_points())
        elif case=='geometry':execute_sequence([FormationStage(named_shape('square',4,width=20.),50)],config(),mission.reference_points())
        else:execute_sequence([FormationStage(mission,50)],config(),mission.reference_points(),initial_velocities=np.full((4,2),2.))


def test_export_contains_one_time_axis_and_policy_reset_at_boundary(tmp_path):
    mission=named_shape('square',4,width=1.)
    result=execute_sequence([FormationStage(mission,20),FormationStage(mission,20)],config(),mission.reference_points())
    output=tmp_path/'sequence.npz';result.save(output)
    with np.load(output,allow_pickle=False) as archive:
        assert archive['positions'].shape==(41,4,2)
        assert archive['times'].shape==(41,)
        assert archive['stage_ids'].shape==(40,)
        np.testing.assert_array_equal(archive['boundaries'],[0,20,40])
        assert archive['allocation_consensus'][20]==False
    with pytest.raises(FileExistsError):result.save(output)
