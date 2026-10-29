"""Raccordement d'inférence ; fixtures non entraînées, aucune qualité apprise affirmée."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from src.imitation.local_target_model import LocalTargetGAT, LocalTargetConfig, save_model
from src.imitation.local_target_evaluate import evaluate_run
from src.imitation.multiform_data import sha256
from src.reference.reconnection import ReconnectionConfig
from src.reference.shape_control import execute_shape, ShapeGains
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from test_multiform_data import pilot


def test_raw_motion_policy_never_calls_teacher_and_has_no_motion_filter(monkeypatch):
    import src.reference.shape_control as shape
    monkeypatch.setattr(shape,'assigned_command',lambda *a,**k: pytest.fail('Professeur consulté en mode brut'))
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),horizon=160)
    mission = named_shape('square',4,width=1.)
    phases=[]
    def policy(observation,target,phase,count):
        phases.append(phase)
        assert target.shape==(2,) and count==4
        target[:]=999  # Le destinataire reçoit une copie, pas la consigne commune.
        return np.zeros(2)
    result=execute_shape(mission,config,mission.reference_points(),gains=ShapeGains(detour_only_on_constraint=True),
        reconnection=ReconnectionConfig(),safety_filter=True,motion_policy=policy,guard_motion=False)
    assert set(phases)=={0,3}
    assert not result.safety_interventions[np.isin(result.phases,(0,3))].any()
    assert np.abs(result.targets).max()<1


def test_evaluation_reads_only_requested_split_pairs_states_and_preserves_weights(pilot,tmp_path,monkeypatch):
    import src.imitation.local_target_evaluate as evaluation
    model=LocalTargetGAT(LocalTargetConfig(),2.)
    folder=tmp_path/'synthetic-run';folder.mkdir();save_model(folder/'best.pt',model)
    (folder/'run.json').write_text(json.dumps(dict(kind='multiform_local_target_training',status='trained',
        synthetic_fixture=True,checkpoint='best.pt',model=model.metadata(),dataset_manifest_sha256=sha256(pilot/'manifest.json'))))
    original=evaluation.load_episode
    seen=[]
    def tracked(dataset,entry,split):
        assert split=='validation'
        seen.append(entry['shape'])
        return original(dataset,entry,split)
    monkeypatch.setattr(evaluation,'load_episode',tracked)
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k: pytest.fail('Optimiseur créé pendant comparaison'))
    output=tmp_path/'evaluation';before=(folder/'best.pt').read_bytes()
    result=evaluate_run(folder,pilot,output,modes=('raw',))
    assert result['status']=='complete' and result['split']=='validation'
    assert seen==['geometries_triangle']
    assert result['summary']['raw']['motion_safety_intervention_robot_steps']==0
    assert (output/'protocol.json').exists()
    assert (folder/'best.pt').read_bytes()==before
    with np.load(output/'teacher-geometries_triangle-7200.npz') as a,np.load(output/'raw-geometries_triangle-7200.npz') as b:
        np.testing.assert_array_equal(a['positions'][0],b['positions'][0])


@pytest.mark.parametrize('split,modes',[('train',('raw',)),('test',()),('validation',('raw','raw'))])
def test_invalid_evaluation_requests_rejected(tmp_path,split,modes):
    with pytest.raises(ValueError): evaluate_run(tmp_path,tmp_path,tmp_path/'unused',split=split,modes=modes)
