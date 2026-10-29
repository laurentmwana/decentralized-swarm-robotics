"""Disques statiques, capteur local et contacts continus entre deux états."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiskObstacle:
    center: tuple[float,float]
    radius: float

    def __post_init__(self):
        center=np.asarray(self.center,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(self.radius) or self.radius<=0:
            raise ValueError('Centre 2D fini et rayon d’obstacle positif requis')
        object.__setattr__(self,'center',tuple(float(v) for v in center))


class DiskWorld:
    def __init__(self,obstacles=()):
        self.obstacles=tuple(obstacles)
        if any(not isinstance(o,DiskObstacle) for o in self.obstacles):
            raise ValueError('Obstacles DiskObstacle requis')
        self.centers=np.array([o.center for o in self.obstacles],dtype=float).reshape(-1,2)
        self.radii=np.array([o.radius for o in self.obstacles],dtype=float)
        self.centers.setflags(write=False);self.radii.setflags(write=False)

    def sense(self,position,*,sensor_range,robot_radius,clearance):
        position=np.asarray(position,dtype=float)
        if position.shape!=(2,) or not np.isfinite(position).all() or not all(np.isfinite(v) and v>0 for v in (sensor_range,robot_radius,clearance)):
            raise ValueError('État et paramètres de capteur locaux valides requis')
        relative=self.centers-position
        visible=np.linalg.norm(relative,axis=-1)-self.radii<=sensor_range
        # Aucun centre lointain n'est fourni au contrôleur.
        return np.column_stack((relative[visible],self.radii[visible]+robot_radius+clearance)).copy()

    def minimum_clearance(self,positions,robot_radius):
        if not self.obstacles:return None
        p=np.asarray(positions,dtype=float)
        relative=p[...,None,:]-self.centers
        distance=np.linalg.norm(relative,axis=-1).min(axis=tuple(range(relative.ndim-2)))
        if p.ndim==3 and len(p)>1:
            start=relative[:-1];delta=relative[1:]-start
            fraction=np.clip(-(start*delta).sum(axis=-1)/np.maximum((delta**2).sum(axis=-1),1e-30),0,1)
            swept=np.linalg.norm(start+fraction[...,None]*delta,axis=-1).min(axis=(0,1))
            distance=np.minimum(distance,swept)
        return float((distance-self.radii-robot_radius).min())

    def collision_timeline(self,positions,robot_radius):
        p=np.asarray(positions,dtype=float)
        if p.ndim!=3 or p.shape[-1]!=2 or len(p)<1 or not np.isfinite(p).all() or not np.isfinite(robot_radius) or robot_radius<=0:
            raise ValueError('Trajectoires finies et rayon physique positif requis')
        relative=p[:,:,None,:]-self.centers
        threshold=(self.radii+robot_radius)**2+1e-12
        initial=int(((relative[0]**2).sum(axis=-1)<=threshold).sum())
        counts=[];contacts=[]
        for start,end in zip(relative[:-1],relative[1:]):
            delta=end-start;norm2=(delta**2).sum(axis=-1)
            fraction=np.clip(-(start*delta).sum(axis=-1)/np.maximum(norm2,1e-30),0,1)
            nearest=start+fraction[...,None]*delta
            contact=(nearest**2).sum(axis=-1)<=threshold
            at_start=(start**2).sum(axis=-1)<=threshold
            counts.append(int((contact & ~at_start).sum()));contacts.append(bool(contact.any()))
        return np.asarray(counts,dtype=int),np.asarray(contacts,dtype=bool),initial


def wall_events(positions,arena_size,robot_radius):
    p=np.asarray(positions,dtype=float)
    if p.ndim!=3 or p.shape[-1]!=2 or len(p)<1 or not np.isfinite(p).all():raise ValueError('Trajectoires 2D finies requises')
    if not np.isfinite(arena_size) or not np.isfinite(robot_radius) or not 0<robot_radius<arena_size/2:
        raise ValueError('Arène et rayon physique positifs compatibles requis')
    half=arena_size/2-robot_radius
    contact=np.concatenate((p<=-half+1e-12,p>=half-1e-12),axis=-1)
    return int(contact[0].sum()+(contact[1:] & ~contact[:-1]).sum())
