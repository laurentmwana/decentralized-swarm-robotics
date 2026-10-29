import json
import numpy as np
import pytest

from src.reference.evaluate_planned_navigation import audit_metrics
from demo import reference_server


@pytest.mark.parametrize('mutation,expected',[(None,[]),('formation_error_m',['formation_error_m']),('arrival_time_s',['stabilization_hold'])])
def test_saved_state_audit_rejects_wrong_metrics_and_premature_arrival(tmp_path,mutation,expected):
    offsets=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])
    positions=np.tile(offsets,(14,1,1))
    metadata=dict(preparation_steps=0,reference=dict(points=[[-1,0],[0,0]],durations=[.2]),
        config=dict(physics=dict(dt=.1)),protocol=dict(formation_threshold=.1,arrival_threshold=.1,
            speed_threshold=.05,hold_seconds=1.,transit_formation_limit=.2))
    np.savez(tmp_path/'trajectory.npz',positions=positions,velocities=np.zeros_like(positions),offsets=offsets,metadata=json.dumps(metadata))
    metrics=dict(formation_error_m=0.,max_navigation_formation_error_m=0.,destination_error_m=0.,
        success=True,arrival_time_s=.2,collisions=0,preparation_success=True)
    if mutation:metrics[mutation]=.5 if mutation=='formation_error_m' else 0.
    campaign=dict(reports=[dict(trajectories=dict(guarded='trajectory.npz'),modes=dict(guarded=metrics))])
    result=audit_metrics(tmp_path,campaign)
    assert result['trajectories']==1
    assert ([e for failure in result['failures'] for e in failure['errors']])==expected


def test_public_summary_rejects_missing_or_duplicate_cases(tmp_path,monkeypatch):
    original=json.loads(reference_server.PLANNED_EVALUATION.read_text(encoding='utf-8'))
    assert reference_server.verified_navigation_summary()['guarded']['successes']==8
    original['reports'][-1]=original['reports'][0]
    path=tmp_path/'evaluation.json';path.write_text(json.dumps(original),encoding='utf-8')
    monkeypatch.setattr(reference_server,'PLANNED_EVALUATION',path)
    assert reference_server.verified_navigation_summary() is None
