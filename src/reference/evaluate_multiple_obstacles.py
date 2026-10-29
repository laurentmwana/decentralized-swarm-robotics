"""Evaluate every public obstacle count with paired unchanged controllers."""
import argparse
import json
from pathlib import Path
import numpy as np

from demo.reference_server import ReferenceService,EXPECTED_MODEL_SHA
from .audit_navigation import audit_campaign
from .evaluate_planned_navigation import audit_metrics


def audit_shared_preparation(folder,campaign):
    checked=0;failures=[]
    for report in campaign['reports']:
        if report['obstacle_count']==1:continue
        with np.load(Path(folder)/report['trajectories']['teacher'],allow_pickle=False) as teacher, np.load(Path(folder)/report['trajectories']['guarded'],allow_pickle=False) as guarded:
            stop=json.loads(str(teacher['metadata']))['preparation_steps']+1
            if not all(np.array_equal(teacher[key][:stop],guarded[key][:stop]) for key in ('positions','velocities')):
                failures.append(dict(obstacle_count=report['obstacle_count'],seed=report['seed']))
            checked+=1
    return dict(paired_preparations=checked,failures=failures)


def evaluate(output,primary_seed=8901):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=[(count,primary_seed) for count in range(5,11)]+[(10,primary_seed+1),(1,primary_seed+1)]
    campaign=dict(kind='multiple_obstacles_campaign',status='registered',model_sha256=EXPECTED_MODEL_SHA,
        protocol=dict(cases=cases,shape='geometries_square',robots=12,weights_updated=False,
            obstacle_layout='near_final_slots_v2',
            obstacle_map='fixed_declared_scene',formation_threshold_m=.1,transit_limit_m=.2,
            destination_threshold_m=.1,speed_threshold_m_s=.05,hold_seconds=1.),reports=[],summary={})
    def save():
        (output/'evaluation.json').write_text(json.dumps(campaign,indent=2,allow_nan=False),encoding='utf-8')
    save();service=ReferenceService()
    try:
        for count,seed in cases:
            name=f'obstacles-{count}-seed-{seed}'
            manifest=service.simulate(dict(scenario='direct' if count==1 else 'multiple',obstacle_count=count,controller='comparison',seed=seed),output/name)
            campaign['reports'].append(dict(obstacle_count=count,seed=seed,initial_positions=manifest['initial_positions'],
                modes=manifest['reports'],planner=manifest['navigation_planner'],
                trajectories={mode:f'{name}/{files["archive"]}' for mode,files in manifest['files'].items()}))
            save();print(name,{mode:(m['success'],m['collisions']) for mode,m in manifest['reports'].items()},flush=True)
        for mode in ('teacher','guarded'):
            rows=[r['modes'][mode] for r in campaign['reports']]
            campaign['summary'][mode]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
                collisions=sum(r['collisions'] for r in rows),max_formation_error_m=max(r['max_navigation_formation_error_m'] for r in rows),
                max_destination_error_m=max(r['destination_error_m'] for r in rows))
        campaign['status']='complete';save()
        audit=audit_campaign(output);audit['metric_checks']=audit_metrics(output,campaign)
        audit['shared_preparation']=audit_shared_preparation(output,campaign)
        (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
        if audit['failures'] or audit['metric_checks']['failures'] or audit['shared_preparation']['failures']:raise RuntimeError('Audit failed')
        return campaign
    finally:service.executor.shutdown()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=8901)
    args=parser.parse_args();print(json.dumps(evaluate(args.output,args.seed)['summary'],indent=2))
