from dataclasses import replace
import numpy as np
import pytest

from demo.reference_server import navigation_scene
from src.reference.route_planning import plan_formation_route
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


def test_detour_preserves_endpoints_and_swept_clearance_for_every_slot():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',12,width=6.)
    reference,world,_=navigation_scene('direct')
    planned=plan_formation_route(shape,reference,world,config)
    assert len(planned.points)>2
    assert planned.points[0]==reference.points[0] and planned.points[-1]==reference.points[-1]
    # Swept geometry, not just clear waypoints.
    positions=np.asarray(planned.points)[:,None,:]+shape.reference_points()[None,:,:]
    assert world.minimum_clearance(positions,config.physics.robot_radius)>.35
    for time in np.linspace(0,planned.duration,500):
        _,velocity,_=planned.sample(time)
        assert np.linalg.norm(velocity)<=.2+1e-10


def test_blocked_final_slot_is_rejected_without_changing_destination():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',12,width=6.)
    reference,world,_=navigation_scene('blocked')
    with pytest.raises(ValueError,match='Destination impossible'):
        plan_formation_route(shape,reference,world,config)


def test_planner_does_not_cross_walls_to_escape_an_obstacle():
    config=AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml')
    shape=named_shape('square',12,width=6.)
    reference,world,_=navigation_scene('direct')
    small=replace(config,physics=replace(config.physics,arena_size=8.5))
    with pytest.raises(ValueError):plan_formation_route(shape,reference,world,small)
