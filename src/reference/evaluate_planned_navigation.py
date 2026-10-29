"""Paired, reproducible evaluation of the static-map formation planner."""
import argparse
import json
from pathlib import Path
import numpy as np

from demo.reference_server import ReferenceService,EXPECTED_MODEL_SHA
from .audit_navigation import audit_campaign


def audit_metrics(folder,campaign):
    """Recompute thresholds from saved physical states, independently of reports."""
    failures=[];checked=0
    for report in campaign['reports']:
        for mode,filename in report['trajectories'].items():
            with np.load(Path(folder)/filename,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']));start=meta['preparation_steps']
                p=data['positions'][start:];v=data['velocities'][start:];offsets=data['offsets']
                center=p.mean(axis=1)-offsets.mean(axis=0)
                distance=np.linalg.norm(p[:,:,None,:]-center[:,None,None,:]-offsets[None,None,:,:],axis=-1)
                error=np.maximum(distance.min(axis=1).max(axis=1),distance.min(axis=2).max(axis=1))
                destination=np.linalg.norm(center-np.asarray(meta['reference']['points'][-1]),axis=1)
                dt=meta['config']['physics']['dt'];protocol=meta['protocol']
                stable=(error<=protocol['formation_threshold'])&(destination<=protocol['arrival_threshold'])
                stable&=np.linalg.norm(v,axis=-1).max(axis=1)<=protocol['speed_threshold']
                stable&=np.arange(len(p))*dt>=sum(meta['reference']['durations'])-1e-9
                hold=int(np.ceil(protocol['hold_seconds']/dt-1e-9))
                m=report['modes'][mode];errors=[]
                for key,actual in [('formation_error_m',error[-1]),('max_navigation_formation_error_m',error.max()),('destination_error_m',destination[-1])]:
                    if not np.isclose(m[key],actual,atol=1e-10,rtol=1e-10):errors.append(key)
                if m['success']:
                    step=int(round(m['arrival_time_s']/dt))-start
                    if not (0<=step and step+hold<len(stable) and stable[step:step+hold+1].all()):errors.append('stabilization_hold')
                    if error.max()>protocol['transit_formation_limit']:errors.append('transit_formation_limit')
                    if m['collisions'] or not m['preparation_success']:errors.append('success_prerequisites')
                checked+=1
                if errors:failures.append(dict(trajectory=filename,errors=errors))
    return dict(trajectories=checked,failures=failures)


def evaluate(output, *, seeds=(8600,8601), shapes=('geometries_square','geometries_rectangle','geometries_triangle')):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    service=ReferenceService()
    campaign=dict(kind='planned_navigation_campaign',status='registered',model_sha256=EXPECTED_MODEL_SHA,
        protocol=dict(seeds=list(seeds),shapes=list(shapes),scenarios=['direct'],
            robot_count=12,formation_threshold_m=.1,transit_limit_m=.2,
            destination_threshold_m=.1,hold_seconds=1.,speed_threshold_m_s=.05,
            planner='classical_static_scene_map',weights_updated=False),reports=[],summary={})
    def save():
        (output/'evaluation.json').write_text(json.dumps(campaign,indent=2,allow_nan=False),encoding='utf-8')
    save()
    cases=[(shape,'direct',seed) for shape in shapes for seed in seeds]
    cases.extend([('geometries_square',scenario,seeds[0]) for scenario in ('straight','bypass')])
    for shape,scenario,seed in cases:
        name=f'{shape}-{scenario}-{seed}'
        manifest=service.simulate(dict(shape=shape,scenario=scenario,seed=seed,robots=12,controller='comparison'),output/name)
        row=dict(shape=shape,scenario=scenario,seed=seed,initial_positions=manifest['initial_positions'],
            modes=manifest['reports'],planner=manifest['navigation_planner'],
            trajectories={mode:f'{name}/{files["archive"]}' for mode,files in manifest['files'].items()})
        campaign['reports'].append(row);save()
        print(name,{mode:(m['success'],m['collisions']) for mode,m in row['modes'].items()},flush=True)
    for mode in ('teacher','guarded'):
        rows=[row['modes'][mode] for row in campaign['reports']]
        campaign['summary'][mode]=dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
            collisions=sum(r['collisions'] for r in rows),max_formation_error_m=max(r['max_navigation_formation_error_m'] for r in rows),
            max_destination_error_m=max(r['destination_error_m'] for r in rows))
    campaign['status']='complete';save()
    audit=audit_campaign(output)
    audit['metric_checks']=audit_metrics(output,campaign)
    (output/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    if audit['failures'] or audit['metric_checks']['failures']:raise RuntimeError('Trajectory audit failed')
    return campaign,audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();campaign,audit=evaluate(args.output)
    print(json.dumps(dict(summary=campaign['summary'],audit_failures=audit['failures']),indent=2))
