"""Choose explicit feasible dimensions without weakening physical constraints."""
from functools import lru_cache
import numpy as np

from src.reference.shapes import named_shape


@lru_cache(maxsize=512)
def geometry_limits(identifier,count):
    shape=named_shape(identifier,count,width=1.)
    points=shape.reference_points()
    distances=np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    minimum=float(distances.min())
    # Largest edge of a minimum spanning tree: exact connectivity threshold.
    visited=np.zeros(count,dtype=bool);visited[0]=True
    nearest=distances[0].copy();largest=0.
    for _ in range(count-1):
        index=int(np.argmin(np.where(visited,np.inf,nearest)))
        largest=max(largest,float(nearest[index]));visited[index]=True
        nearest=np.minimum(nearest,distances[index])
    vertices=np.concatenate(shape.strokes)
    return minimum,largest,vertices.min(axis=0),vertices.max(axis=0)


def fit_geometry(identifier,count,config,requested_width,*,reference_points=((0.,0.),),navigation=False):
    minimum,connection,low,high=geometry_limits(identifier,count)
    centers=np.asarray(reference_points,dtype=float)
    half=config.physics.arena_size/2-config.physics.robot_radius-(.35 if navigation else .15)
    wall_limit=float('inf')
    for axis in range(2):
        if high[axis]>0:wall_limit=min(wall_limit,(half-centers[:,axis].max())/high[axis])
        if low[axis]<0:wall_limit=min(wall_limit,(-half-centers[:,axis].min())/low[axis])
    lower=.4/minimum
    upper=min(config.communication_radius/connection,wall_limit)
    if minimum*requested_width>=.4-1e-12 and connection*requested_width<=config.communication_radius+1e-12 and requested_width<=wall_limit+1e-12:
        width=requested_width
    elif lower*1.01<=upper*.99:
        # Prefer extra room when enlargement is necessary; retain original valid
        # dimensions and avoid forcing infeasible extra margin in narrow arenas.
        preferred=max(requested_width,.5/minimum) if requested_width<lower else requested_width
        width=float(np.clip(preferred,lower*1.01,upper*.99))
    else:
        return dict(compatible=False,width_m=None,adjusted=False,
            message=f'Cette forme ne peut pas accueillir {count} robots dans cette arène tout en gardant les distances de sécurité et les communications. Réduisez l’effectif ou choisissez une autre forme.',
            minimum_width_m=lower,maximum_width_m=upper)
    return dict(compatible=True,width_m=width,adjusted=abs(width-requested_width)>1e-9,
        minimum_spacing_m=minimum*width,communication_link_m=connection*width,
        message=f'Largeur adaptée à {width:.2f} m pour garder au moins 0,40 m entre les places.' if abs(width-requested_width)>1e-9 else f'Largeur : {width:.2f} m. Espacement et communications compatibles.')
