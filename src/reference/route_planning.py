"""Classical translation planner using the declared static scene map.

Every robot slot sweeps a disk along each segment. The learned controller still
receives only its original local observations; the map belongs to this planner.
"""
import heapq
import math
import numpy as np

from .navigation import MovingReference


def plan_formation_route(shape, reference, world, config, *, clearance=.35, resolution=.1, speed=.2):
    if not all(math.isfinite(value) and value>0 for value in (clearance,resolution,speed)):
        raise ValueError('Marge, résolution et vitesse finies positives requises')
    offsets=shape.reference_points()
    half=config.physics.arena_size/2-config.physics.robot_radius-clearance
    lower=-half-offsets.min(axis=0);upper=half-offsets.max(axis=0)
    centers=(world.centers[:,None,:]-offsets[None,:,:]).reshape(-1,2)
    radii=np.repeat(world.radii+config.physics.robot_radius+clearance,len(offsets))

    def clear(a,b):
        a=np.asarray(a);b=np.asarray(b)
        if np.any(a<lower-1e-9) or np.any(a>upper+1e-9) or np.any(b<lower-1e-9) or np.any(b>upper+1e-9):return False
        delta=b-a
        fraction=np.clip(((centers-a)*delta).sum(axis=1)/max(float(delta@delta),1e-30),0,1)
        return bool(np.all(np.linalg.norm(a+fraction[:,None]*delta-centers,axis=1)>radii))

    points=[np.asarray(reference.points[0])]
    for destination in reference.points[1:]:
        start=points[-1];goal=np.asarray(destination)
        if not clear(start,start):
            raise ValueError('Formation de départ incompatible avec les obstacles ou les murs. Choisissez une autre forme ou un autre parcours.')
        if not clear(goal,goal):
            raise ValueError('Destination impossible pour cette formation : une place finale est occupée par un obstacle ou trop proche d’un mur. Choisissez un parcours avec une destination libre.')
        if clear(start,goal):points.append(goal);continue
        # Grid anchored at the leg start, with exact start and exact destination.
        def position(key):return start+resolution*np.asarray(key)
        origin=(0,0);cost={origin:0.};parents={};queue=[(float(np.linalg.norm(goal-start)),0.,origin)]
        reached=None
        while queue:
            _,g,key=heapq.heappop(queue)
            if g>cost[key]+1e-12:continue
            point=position(key)
            if clear(point,goal):reached=key;break
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                neighbor=(key[0]+dx,key[1]+dy);candidate=position(neighbor)
                new=g+resolution*math.hypot(dx,dy)
                if new>=cost.get(neighbor,float('inf')) or not clear(point,candidate):continue
                cost[neighbor]=new;parents[neighbor]=key
                heapq.heappush(queue,(new+float(np.linalg.norm(goal-candidate)),new,neighbor))
        if reached is None:raise ValueError('Aucun passage trouvé pour conserver cette formation dans l’arène. Choisissez une formation plus petite ou un autre parcours.')
        path=[goal,position(reached)]
        while reached!=origin:reached=parents[reached];path.append(position(reached))
        path.reverse();anchor=0
        while anchor<len(path)-1:
            far=len(path)-1
            while not clear(path[anchor],path[far]):far-=1
            points.append(path[far]);anchor=far
    distances=np.linalg.norm(np.diff(points,axis=0),axis=1)
    # Quintic interpolation peaks at 1.875 times its average segment speed.
    durations=tuple(max(4.,1.875*float(length)/speed) for length in distances)
    return MovingReference(tuple(map(tuple,points)),durations)
