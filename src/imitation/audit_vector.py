"""Audit des états réservés : physique, contacts, stabilisation et mouvement non filtré."""
import argparse
from dataclasses import asdict
import json
import math  
from pathlib import Path

import numpy as np

from .multiform_data import sha256
from .multisize import PROFILES
from src.reference.metrics import collision_events
from src.reference.obstacles import wall_events
from src.reference.shape_control import coverage_error
from src.reference.shapes import named_shape


def minimum_separation(p):
    i,j=np.triu_indices(p.shape[1],1);minimum=float('inf')
    for start in range(0,len(p)-1,128):
        end=min(start+128,len(p)-1)
        a=p[start:end][:,i]-p[start:end][:,j]
        b=p[start+1:end+1][:,i]-p[start+1:end+1][:,j]
        delta=b-a;t=np.clip(-(a*delta).sum(-1)/np.maximum((delta*delta).sum(-1),1e-30),0,1)
        minimum=min(minimum,float(np.linalg.norm(a+t[...,None]*delta,axis=-1).min()))
    return minimum


def audit(run,folder,output):
    run,folder,output=map(Path,(run,folder,output));report=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    if report.get('status')!='complete' or report.get('split')!='fresh_test':raise ValueError('Test final terminé requis')
    digest=sha256(run/'best.pt');errors=[];trajectories=0;interventions=0;initial={};minimum={str(n):float('inf') for n in PROFILES}
    expected={(n,s,seed) for n in PROFILES for s in ('geometries_rectangle','letters_C','numbers_3') for seed in (8500,8501)}
    actual={(r['count'],r['shape'],r['seed']) for r in report['reports']}
    if actual!=expected or len(report['reports'])!=18:errors.append('Réserve incomplète ou dupliquée')
    if digest!=report['model_sha256']:errors.append('Poids modifiés')
    for row in report['reports']:
        n,name,seed=row['count'],row['shape'],row['seed'];profile=PROFILES[n];config=profile.config
        targets=named_shape(name,n,width=profile.width).reference_points()
        for mode,key in [('raw','metrics'),('teacher','teacher_metrics')]:
            metrics=row[key];prefix=f'{mode}-{n}-{name}-{seed}'
            try:
                with np.load(folder/f'{prefix}.npz',allow_pickle=False) as data:
                    p,v,a=data['positions'],data['velocities'],data['accelerations'];trajectories+=1
                    assert p.shape==(config.horizon+1,n,2) and v.shape==p.shape and a.shape==p[:-1].shape
                    assert all(np.isfinite(x).all() for x in (p,v,a))
                    assert np.linalg.norm(v,axis=-1).max()<=1.+1e-9
                    assert np.linalg.norm(a,axis=-1).max()<=2.+1e-9
                    half=config.physics.arena_size/2-.1
                    assert np.abs(p).max()<=half+1e-9
                    np.testing.assert_allclose(np.diff(data['times']),.1,atol=1e-12)
                    velocity=v[:-1]+a*.1
                    velocity*=np.minimum(1.,1./np.maximum(np.linalg.norm(velocity,axis=-1,keepdims=True),1e-12))
                    proposed=p[:-1]+velocity*.1
                    contact=((proposed>=half)&(velocity>0))|((proposed<=-half)&(velocity<0))
                    np.testing.assert_allclose(p[1:],np.clip(proposed,-half,half),atol=1e-10)
                    np.testing.assert_allclose(v[1:],np.where(contact,0.,velocity),atol=1e-10)
                    robots=collision_events(p,.1)[0];walls=wall_events(p,config.physics.arena_size,.1)
                    assert robots==metrics['robot_collisions'] and walls==metrics['wall_collisions']
                    assert robots+walls==metrics['collisions']==0 and metrics['success']
                    time=metrics['stabilization_time_s'];assert time is not None
                    step=round(time/.1)
                    assert step+10<len(p)
                    for state in p[step:step+11]:assert coverage_error(state,targets)<=.1+1e-9
                    assert np.linalg.norm(v[step:step+11],axis=-1).max()<=.05+1e-9
                    assert abs(coverage_error(p[-1],targets)-metrics['formation_error_m'])<1e-9
                    initial_key=(n,seed)
                    if initial_key not in initial:initial[initial_key]=p[0].copy()
                    else:np.testing.assert_array_equal(initial[initial_key],p[0])
                    if mode=='raw':minimum[str(n)]=min(minimum[str(n)],minimum_separation(p))
                if mode=='raw':
                    with np.load(folder/f'{prefix}-control.npz',allow_pickle=False) as control:
                        moving=np.isin(control['phases'],(0,3))
                        count=int(control['safety_interventions'][moving].sum());interventions+=count
                        assert count==0 and control['allocation_consensus'][-1]
                        assert not control['safety_infeasible'][moving].any()
            except (AssertionError,KeyError,ValueError,OSError) as exc:errors.append(dict(trajectory=prefix,error=str(exc)))
    result=dict(kind='vector_gat_reserved_audit',success=not errors,model_sha256=digest,trajectories=trajectories,
        raw_trajectories=18,teacher_trajectories=18,distinct_initial_states=len(initial),
        raw_motion_filter_interventions=interventions,minimum_swept_center_separation_m={n:v if math.isfinite(v) else None for n,v in minimum.items()},
        physical_contact_distance_m=.2,errors=errors,summary=report['summary'],teacher_summary=report['teacher_summary'])
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('run','folder','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    if not audit(args.run,args.folder,args.output)['success']:raise SystemExit(1)


if __name__=='__main__':main()
