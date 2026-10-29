"""Export des campagnes continues et validation des paramètres publics."""
import json

import numpy as np
import pytest
import yaml

from src.reference.evaluate_transitions import duration_steps,evaluate_transitions


def configuration():
    return dict(robot_count=4,width=1.,stage_seconds=15.,communication_radius=3.,seeds=[7400],
                sequences=dict(return_shape=['square','square']))


@pytest.mark.parametrize('value',[0,-1,True,float('nan'),.15])
def test_invalid_or_unaligned_mission_duration_rejected(value):
    with pytest.raises(ValueError):duration_steps(value,.1)


def test_campaign_exports_continuous_reproducible_teacher_trace(tmp_path):
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(configuration()))
    output=tmp_path/'campaign'
    report=evaluate_transitions(config,output)
    assert report['status']=='complete'
    assert report['summary']['teacher']['sequences']==1
    assert report['summary']['teacher']['stages']['episodes']==2
    assert report['summary']['teacher']['boundary_states_preserved']
    assert report['summary']['teacher']['stale_packets_discarded']>0
    assert json.loads((output/'protocol.json').read_text())['status']=='incomplete'
    with np.load(output/'teacher-return_shape-7400.npz',allow_pickle=False) as archive:
        assert archive['positions'].shape==(301,4,2)
        assert archive['times'][-1]==30.
    with pytest.raises(FileExistsError):evaluate_transitions(config,output)


@pytest.mark.parametrize('case',['name','seed','model'])
def test_invalid_campaign_rejected_before_writing(tmp_path,case):
    raw=configuration()
    if case=='name':raw['sequences']={'../escape':['square']}
    elif case=='seed':raw['seeds']=[7400,7400]
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump(raw))
    output=tmp_path/'invalid'
    with pytest.raises(ValueError):
        evaluate_transitions(config,output,modes=('guarded',) if case=='model' else None)
    assert not output.exists()
