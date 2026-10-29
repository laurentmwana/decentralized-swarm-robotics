from dataclasses import replace
import numpy as np
import pytest

from demo.geometry import fit_geometry
from demo.reference_server import ReferenceService,SHAPES
from src.imitation.multisize import PROFILES
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig


@pytest.mark.parametrize('count',[8,12,16,24,64])
def test_every_offered_compatible_geometry_satisfies_original_safety_checks(count):
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),communication_radius=3.)
    width=6.
    if count==64:cfg=PROFILES[count].config;width=PROFILES[count].width
    for identifier in SHAPES:
        fit=fit_geometry(identifier,count,cfg,width)
        if fit['compatible']:
            shape=named_shape(identifier,count,width=fit['width_m'])
            shape.validate_environment(cfg,.4)
            assert fit['minimum_spacing_m']>=.4-1e-12
        else:
            assert fit['width_m'] is None and 'Réduisez' in fit['message']


@pytest.mark.parametrize('count',[12,24,64])
def test_validated_structured_shapes_keep_their_exact_original_dimensions(count):
    profile=PROFILES[count]
    for identifier in ('geometries_rectangle','letters_C','numbers_3'):
        fit=fit_geometry(identifier,count,profile.config,profile.width)
        assert fit['compatible'] and not fit['adjusted'] and fit['width_m']==profile.width


def test_dense_arrow_is_enlarged_and_disconnected_small_swarm_is_shrunk():
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),communication_radius=3.)
    large=fit_geometry('geometries_arrow',24,cfg,6.)
    assert large['compatible'] and large['adjusted'] and large['width_m']>6
    small=fit_geometry('letters_X',8,cfg,6.)
    assert small['compatible'] and small['adjusted'] and small['width_m']<6


def test_preview_does_not_enqueue_jobs_and_impossible_request_is_explained(tmp_path):
    service=ReferenceService(tmp_path)
    try:
        result=service.geometry_options(dict(robots=24,shape='numbers_2',controller='teacher'))
        assert not result['options']['numbers_2']['compatible']
        assert service.submitted==0 and not service.jobs
        with pytest.raises(ValueError,match='Réduisez l’effectif'):
            service.submit(dict(robots=24,shape='numbers_2',controller='teacher'))
        assert service.submitted==0
    finally:service.executor.shutdown()


def test_navigation_fit_reserves_wall_space_for_every_reference_waypoint():
    cfg=replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),communication_radius=3.)
    centers=np.array([[-1.,0.],[-1.,1.],[1.,1.],[1.,0.]])
    fit=fit_geometry('geometries_line',24,cfg,6.,reference_points=centers,navigation=True)
    if fit['compatible']:
        shape=named_shape('geometries_line',24,width=fit['width_m'])
        vertices=np.concatenate(shape.strokes)
        assert np.abs(centers[:,None,:]+vertices[None,:,:]).max()<=4.55+1e-12
    else:assert fit['maximum_width_m']<fit['minimum_width_m']*1.021


def test_preview_detects_obstructed_destination_before_job_submission(tmp_path):
    service=ReferenceService(tmp_path)
    try:
        preview=service.geometry_options(dict(scenario='blocked',shape='geometries_square'))
        selected=preview['options']['geometries_square']
        assert not selected['compatible'] and 'Destination impossible' in selected['message']
        assert service.submitted==0
    finally:service.executor.shutdown()


def test_all_structured_24_robot_options_keep_safe_spacing_and_connected_targets():
    cfg=PROFILES[24].config
    for identifier in SHAPES:
        fit=fit_geometry(identifier,24,cfg,12.)
        if fit['compatible']:named_shape(identifier,24,width=fit['width_m']).validate_environment(cfg,.4)
