"""Profils multi-effectifs et exemples locaux CSR, sans matrice dense persistée."""
from dataclasses import dataclass,asdict
import math

import numpy as np

from src.common.seeds import SeedManager
from src.reference.reconnection import ReconnectionConfig,rendezvous_targets
from src.reference.shapes import named_shape
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.simulator.dynamics import DynamicsConfig


@dataclass(frozen=True)
class SizeProfile:
    count:int
    arena_size:float
    width:float
    gather_seconds:float
    formation_seconds:float=40.

    @property
    def reconnection(self):return ReconnectionConfig(gather_seconds=self.gather_seconds,braking_seconds=2.)

    @property
    def config(self):
        horizon=math.ceil((self.gather_seconds+2.+self.formation_seconds)/.1)+self.count**2
        return AccelerationEpisodeConfig(DynamicsConfig(self.arena_size,.1,1.,2.,.1),horizon,3.,self.arena_size/2-1.)


PROFILES={12:SizeProfile(12,10.,6.,8.),24:SizeProfile(24,18.,12.,15.),64:SizeProfile(64,32.,28.,30.)}


def controlled_start(profile,seed):
    """Anneau proche des propres rendez-vous ID ; pas un départ dispersé général."""
    targets=rendezvous_targets(profile.count,profile.config,profile.reconnection,.4)
    rng=SeedManager(seed).rng('multisize_near_rendezvous')
    noise=rng.uniform(-.035,.035,size=targets.shape)
    angle=rng.uniform(-.005,.005)
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    positions=targets@rotation.T+noise
    distances=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    if distances.min()<.45:raise ValueError('Départ contrôlé trop dense')
    return positions


def sparse_examples(result,stride=2):
    if type(stride)is not int or stride<1:raise ValueError('Pas d’échantillonnage positif requis')
    episode=result.episode;n=episode.positions.shape[1]
    own=[];labels=[];edges=[];offsets=[0];sample_steps=[];sample_robots=[]
    for step in range(0,len(result.phases),stride):
        if result.phases[step] not in (0,3):continue
        for robot in range(n):
            if result.safety_infeasible[step,robot]:continue
            p,v=episode.positions[step],episode.velocities[step]
            visible=episode.adjacency[step,robot]
            neighbors=np.column_stack((p[visible]-p[robot],v[visible]-v[robot])).astype(np.float32)
            own.append(np.r_[p[robot],v[robot],result.local_targets[step,robot],result.phases[step],n])
            labels.append(episode.accelerations[step,robot]);edges.extend(neighbors)
            offsets.append(offsets[-1]+len(neighbors));sample_steps.append(step);sample_robots.append(robot)
    return dict(own=np.asarray(own,dtype=np.float32).reshape(-1,8),labels=np.asarray(labels,dtype=np.float32).reshape(-1,2),
        edges=np.asarray(edges,dtype=np.float32).reshape(-1,4),offsets=np.asarray(offsets,dtype=np.int64),
        steps=np.asarray(sample_steps,dtype=np.int32),robots=np.asarray(sample_robots,dtype=np.int16))
