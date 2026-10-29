from dataclasses import replace
import numpy as np
import pytest

from src.reference.navigation import MovingReference,execute_navigation,evaluate_navigation
from src.reference.obstacles import DiskWorld,DiskObstacle,wall_events
from src.reference.shapes import named_shape
from src.reference.shape_control import ShapeGains
from src.reference.safety import filter_command
from src.simulator.acceleration_episode import LocalObservation
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def test_reference_is_continuous_and_stationary_at_waypoints():
    ref=MovingReference(((0,0),(1,1),(2,0)),(2,3))
    for time,point in [(0,(0,0)),(2,(1,1)),(5,(2,0)),(8,(2,0))]:
        p,v,a=ref.sample(time)
        np.testing.assert_allclose(p,point);np.testing.assert_allclose(v,0,atol=1e-12);np.testing.assert_allclose(a,0,atol=1e-12)
    t=1.2;epsilon=1e-5
    np.testing.assert_allclose((ref.sample(t+epsilon)[0]-ref.sample(t-epsilon)[0])/(2*epsilon),ref.sample(t)[1],rtol=1e-7)
    np.testing.assert_allclose((ref.sample(t+epsilon)[1]-ref.sample(t-epsilon)[1])/(2*epsilon),ref.sample(t)[2],rtol=1e-7)


def test_sensor_only_returns_visible_relative_disks_and_copies():
    world=DiskWorld((DiskObstacle((2,0),.5),DiskObstacle((20,0),1)))
    sensed=world.sense(np.array([1.,0]),sensor_range=1,robot_radius=.1,clearance=.15)
    np.testing.assert_allclose(sensed,[[1,0,.75]])
    sensed[:]=99
    np.testing.assert_allclose(world.centers,[[2,0],[20,0]])


def test_swept_obstacle_contact_and_persistent_contact_count_once():
    world=DiskWorld((DiskObstacle((0,0),.3),))
    counts,contacts,initial=world.collision_timeline(np.array([[[-1.,0]],[[1.,0]],[[2.,0]]]),.1)
    assert counts.tolist()==[1,0] and contacts.tolist()==[True,False] and initial==0
    counts,_,initial=world.collision_timeline(np.array([[[0.,0]],[[.1,0]],[[1.,0]]]),.1)
    assert initial==1 and counts.sum()==0
    assert wall_events(np.array([[[0.,0]],[[4.9,0]],[[4.9,0]],[[0.,0]],[[4.9,0]]]),10,.1)==2


def test_static_obstacle_uses_full_world_velocity_constraint():
    observation=LocalObservation(np.zeros(2),np.array([.2,0]),np.empty((0,2)),np.empty((0,2)),0.)
    gains=ShapeGains()
    command,feasible=filter_command(observation,np.array([1.,0]),gains,2.,obstacle_vectors=[[.5,0,.4]])
    r=np.array([-.5,0]);w=observation.velocity
    bound=-w@w-2*gains.barrier_alpha*(r@w)-.5*gains.barrier_alpha**2*(.5**2-.4**2)
    assert feasible and r@command>=bound-1e-9
    assert command[0]<0
    # No disks gives the unchanged robot-only filter.
    empty,_=filter_command(observation,np.array([1.,0]),gains,2.)
    np.testing.assert_allclose(empty,[1,0])


def test_moving_frame_callback_only_receives_local_physical_neighbor_vectors():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',4,width=1.)
    calls=[]
    def policy(obs,target,phase,count):
        calls.append((obs,target.copy(),phase))
        return np.zeros(2)
    result=execute_navigation(shape,MovingReference(((0,0),(.1,0)),(2,)),DiskWorld(),config,
        shape.reference_points(),preparation_steps=250,navigation_steps=4,motion_policy=policy)
    for offset,(obs,target,phase) in enumerate(calls[-16:]):
        step,index=divmod(offset,4)
        center,velocity,_=result.reference.sample(step*.1)
        np.testing.assert_allclose(obs.position,result.episode.positions[250+step,index]-center)
        np.testing.assert_allclose(obs.velocity,result.episode.velocities[250+step,index]-velocity)
        np.testing.assert_allclose(target,result.offsets[result.slots[index]])
        neighbors=result.episode.adjacency[250+step,index]
        np.testing.assert_allclose(obs.relative_positions,result.episode.positions[250+step,neighbors]-result.episode.positions[250+step,index])


def test_navigation_continuity_and_stationary_formation_does_not_arrive():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',4,width=1.)
    # The reference feedforward still moves a zero-output controller.
    result=execute_navigation(shape,MovingReference(((0,0),(1,0)),(2,)),DiskWorld(),config,
        shape.reference_points(),preparation_steps=250,navigation_steps=40,motion_policy=lambda *args:np.zeros(2))
    np.testing.assert_array_equal(result.episode.positions[250],result.preparation.episode.positions[-1])
    np.testing.assert_array_equal(result.episode.velocities[250],result.preparation.episode.velocities[-1])
    assert result.episode.positions.shape==(291,4,2)
    frozen=result.episode.positions.copy();frozen[250:]=frozen[250]
    velocities=result.episode.velocities.copy();velocities[250:]=0
    result=replace(result,episode=replace(result.episode,positions=frozen,velocities=velocities))
    report=evaluate_navigation(result)
    assert not report['success'] and report['arrival_time_s'] is None
    assert report['destination_error_m']>.1


@pytest.mark.parametrize('points,durations', [([(0,0)],[]), ([(0,0),(0,0)],[1]), ([(0,0),(1,0)],[0])])
def test_invalid_routes(points,durations):
    with pytest.raises(ValueError):MovingReference(points,durations)


def test_navigation_window_before_reference_end_is_failure():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',4,width=1.)
    result=execute_navigation(shape,MovingReference(((0,0),(1,0)),(10,)),DiskWorld(),config,
        shape.reference_points(),preparation_steps=250,navigation_steps=2)
    report=evaluate_navigation(result)
    assert not report['success'] and report['arrival_time_s'] is None
    assert 'arrival_or_stabilization_deadline' in report['failure_reasons']


def test_minimum_clearance_also_measures_swept_segment():
    world=DiskWorld((DiskObstacle((0,0),.3),))
    assert world.minimum_clearance(np.array([[[-1.,0]],[[1.,0]]]),.1)==pytest.approx(-.4)
