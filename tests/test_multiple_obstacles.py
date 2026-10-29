import numpy as np
import pytest

from demo.reference_server import navigation_scene,validate_request,LocalReferenceServer,handler_for,ReferenceService
from src.reference.route_planning import plan_formation_route
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.simulator.acceleration_episode import LocalObservation
from src.reference.safety import filter_command
from src.reference.shape_control import ShapeGains


@pytest.mark.parametrize('count',range(5,11))
def test_multiple_obstacles_are_distinct_and_whole_route_is_clear(count):
    reference,world,_=navigation_scene('multiple',count)
    assert len(world.obstacles)==count
    assert len(np.unique(world.centers,axis=0))==count
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',12,width=6.)
    route=plan_formation_route(shape,reference,world,config)
    assert route.points[0]==reference.points[0] and route.points[-1]==reference.points[-1]
    positions=np.asarray(route.points)[:,None,:]+shape.reference_points()[None,:,:]
    assert world.minimum_clearance(positions,config.physics.robot_radius)>.35


@pytest.mark.parametrize('count',[True,4,11,5.5,'10'])
def test_invalid_multiple_obstacle_counts_are_rejected(count):
    with pytest.raises(ValueError):validate_request(dict(scenario='multiple',obstacle_count=count))


def test_multiple_obstacles_default_and_model_domain_are_explicit():
    assert validate_request(dict(scenario='multiple'))['obstacle_count']==5
    with pytest.raises(ValueError):validate_request(dict(scenario='multiple',controller='multisize',robots=24))
    with pytest.raises(ValueError):validate_request(dict(scenario='formation',obstacle_count=10))


@pytest.mark.parametrize('normal',[(1,0),(-1,0),(0,1),(0,-1)])
def test_wall_guard_brakes_before_contact_without_changing_physics(normal):
    normal=np.asarray(normal,dtype=float)
    observation=LocalObservation(normal*4.5,normal*.6,np.empty((0,2)),np.empty((0,2)),0.)
    command,feasible=filter_command(observation,normal*2,ShapeGains(),2,wall_half_extent=4.75)
    assert feasible and command@normal<=-.95+1e-9
    assert np.linalg.norm(command)<=2+1e-9


def test_second_server_cannot_silently_serve_an_old_version_on_same_port(tmp_path):
    service=ReferenceService(tmp_path)
    server=LocalReferenceServer(('127.0.0.1',0),handler_for(service))
    try:
        with pytest.raises(OSError):LocalReferenceServer(server.server_address,handler_for(service))
    finally:server.server_close();service.executor.shutdown()
