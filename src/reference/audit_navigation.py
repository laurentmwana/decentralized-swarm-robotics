"""Audit indépendant des trajectoires exportées et des comptes de contacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .metrics import collision_events
from .obstacles import DiskWorld,DiskObstacle,wall_events


def audit_campaign(folder):
    folder=Path(folder);campaign=json.loads((folder/'evaluation.json').read_text(encoding='utf-8'))
    failures=[];trajectories=0;max_speed=0.;max_acceleration=0.;min_clearance=None
    for report in campaign['reports']:
        initial=None
        for mode,filename in report['trajectories'].items():
            with np.load(folder/filename,allow_pickle=False) as data:
                trajectories+=1
                p,v,a=data['positions'],data['velocities'],data['accelerations']
                meta=json.loads(str(data['metadata']));physics=meta['config']['physics'];dt=physics['dt']
                world=DiskWorld(DiskObstacle(tuple(o['center']),o['radius']) for o in meta['obstacles'])
                collisions=collision_events(p,physics['robot_radius'])[0]
                events,_,at_start=world.collision_timeline(p,physics['robot_radius'])
                counts=dict(robot_collisions=collisions,obstacle_collisions=int(events.sum())+at_start,
                            wall_collisions=wall_events(p,physics['arena_size'],physics['robot_radius']))
                errors=[]
                if any(report['modes'][mode][key]!=value for key,value in counts.items()):errors.append('collision_counts')
                if not all(np.isfinite(value).all() for value in (p,v,a)):errors.append('finite_states')
                if not np.allclose(np.diff(data['times']),dt,atol=1e-12):errors.append('clock')
                expected_velocity=v[:-1]+a*dt
                norms=np.linalg.norm(expected_velocity,axis=-1,keepdims=True)
                expected_velocity*=np.minimum(1.,physics['max_speed']/np.maximum(norms,1e-12))
                proposed=p[:-1]+expected_velocity*dt
                half=physics['arena_size']/2-physics['robot_radius']
                wall_contact=((proposed>=half)&(expected_velocity>0))|((proposed<=-half)&(expected_velocity<0))
                if not np.allclose(p[1:],np.clip(proposed,-half,half),atol=1e-10):errors.append('position_integrator')
                if not np.allclose(v[1:],np.where(wall_contact,0.,expected_velocity),atol=1e-9):errors.append('velocity_integrator')
                speed=float(np.linalg.norm(v,axis=-1).max());acceleration=float(np.linalg.norm(a,axis=-1).max())
                max_speed=max(max_speed,speed);max_acceleration=max(max_acceleration,acceleration)
                if speed>physics['max_speed']+1e-9:errors.append('speed_limit')
                if acceleration>physics['max_acceleration']+1e-9:errors.append('acceleration_limit')
                if initial is None:initial=p[0].copy()
                elif not np.array_equal(initial,p[0]):errors.append('unpaired_initial_state')
                if not np.array_equal(p[0],np.asarray(report['initial_positions'])):errors.append('declared_initial_state')
                clearance=world.minimum_clearance(p,physics['robot_radius'])
                if clearance is not None:min_clearance=clearance if min_clearance is None else min(min_clearance,clearance)
                if errors:failures.append(dict(trajectory=filename,errors=errors))
    return dict(trajectories=trajectories,failures=failures,max_speed_m_s=max_speed,
                max_acceleration_m_s2=max_acceleration,minimum_swept_obstacle_clearance_m=min_clearance,
                model_sha256=campaign['model_sha256'],summary=campaign['summary'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+');parser.add_argument('--output',required=True)
    parser.add_argument('--checkpoint',required=True)
    args=parser.parse_args()
    checkpoint_hash=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    reports={folder:audit_campaign(folder) for folder in args.folders}
    result=dict(checkpoint_sha256=checkpoint_hash,campaigns=reports,
        success=all(not r['failures'] and r['model_sha256']==checkpoint_hash for r in reports.values()))
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['success']:raise SystemExit(1)


if __name__=='__main__':main()
