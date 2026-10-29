"""Séquences de missions avec état physique continu et versions d'affectation."""
from dataclasses import dataclass, replace
import math

import numpy as np

from .allocation import _integer
from .metrics import collision_timeline
from .shape_control import ShapeGains, ShapeEpisodeResult, execute_shape, coverage_error
from .shapes import ShapeMission
from src.simulator.acceleration_episode import AccelerationEpisodeResult
from src.simulator.dynamics import _vectors


@dataclass(frozen=True)
class FormationStage:
    mission: ShapeMission
    steps: int

    def __post_init__(self):
        _integer(self.steps,'steps',1)
        if not isinstance(self.mission,ShapeMission):
            raise ValueError('Consigne géométrique ShapeMission requise')


@dataclass(frozen=True)
class TransitionProtocol:
    error_threshold: float = .1
    speed_threshold: float = .05
    hold_seconds: float = 1.

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.error_threshold,self.speed_threshold,self.hold_seconds)):
            raise ValueError('Seuils et maintien finis et positifs requis')


@dataclass(frozen=True)
class SequenceResult:
    episode: AccelerationEpisodeResult
    stages: tuple[FormationStage, ...]
    segments: tuple[ShapeEpisodeResult, ...]
    boundaries: np.ndarray
    stage_ids: np.ndarray
    assignments: np.ndarray
    phases: np.ndarray
    local_targets: np.ndarray
    stale_messages_per_step: np.ndarray

    def save(self, path):
        import json
        from dataclasses import asdict
        from pathlib import Path
        episode = self.episode
        metadata = dict(schema_version=1,kind='continuous_formation_sequence',seed=episode.seed,
            config=asdict(episode.config),stages=[dict(mission=asdict(s.mission),steps=s.steps,epoch=f'stage:{i}')
                                                for i,s in enumerate(self.stages)],
            units=dict(position='m',velocity='m/s',acceleration='m/s^2',time='s'))
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,positions=episode.positions,velocities=episode.velocities,
                requested_accelerations=episode.requested_accelerations,accelerations=episode.accelerations,
                adjacency=episode.adjacency,times=episode.times,boundaries=self.boundaries,stage_ids=self.stage_ids,
                assignments=self.assignments,phases=self.phases,local_targets=self.local_targets,
                stale_messages_per_step=self.stale_messages_per_step,
                allocation_consensus=np.concatenate([s.allocation_consensus[:-1] for s in self.segments]+
                                                   [self.segments[-1].allocation_consensus[-1:]]),
                safety_infeasible=np.concatenate([s.safety_infeasible for s in self.segments]),
                safety_interventions=np.concatenate([s.safety_interventions for s in self.segments]),
                messages_per_step=np.concatenate([s.messages_per_step for s in self.segments]),
                metadata=json.dumps(metadata))


def execute_sequence(stages, config, positions, *, initial_velocities=None, gains=None,
                     reconnection=None, safety_filter=True, motion_policy=None, guard_motion=None,
                     allocation_rounds=None, replay_previous_packets=True, seed=0):
    stages = tuple(stages)
    if not stages or any(not isinstance(s,FormationStage) for s in stages):
        raise ValueError('Séquence non vide de missions typées requise')
    if not isinstance(replay_previous_packets,bool):
        raise ValueError('Option de rejeu de paquets booléenne requise')
    gains = ShapeGains() if gains is None else gains
    count = stages[0].mission.robot_count
    for stage in stages:
        if stage.mission.robot_count != count:
            raise ValueError('L’effectif doit rester constant pendant la séquence')
        stage.mission.validate_environment(config,gains.safety_distance)
    p = _vectors(positions,'positions').copy()
    v = np.zeros_like(p) if initial_velocities is None else _vectors(initial_velocities,'velocities').copy()
    if p.shape != (count,2) or v.shape != p.shape:
        raise ValueError('État initial de séquence incompatible')
    segments, boundaries, pending = [], [0], ()
    for index, stage in enumerate(stages):
        segment = execute_shape(stage.mission,replace(config,horizon=stage.steps),p,initial_velocities=v,
            gains=gains,reconnection=reconnection,safety_filter=safety_filter,motion_policy=motion_policy,
            guard_motion=guard_motion,allocation_rounds=allocation_rounds,seed=seed,
            time_offset=boundaries[-1]*config.physics.dt,mission_epoch=f'stage:{index}',
            previous_packets=pending if replay_previous_packets else (),allow_short_window=True)
        segments.append(segment)
        p, v = segment.episode.positions[-1].copy(), segment.episode.velocities[-1].copy()
        pending = segment.final_packets or pending
        boundaries.append(boundaries[-1]+stage.steps)
    positions = np.concatenate([segments[0].episode.positions]+[s.episode.positions[1:] for s in segments[1:]])
    velocities = np.concatenate([segments[0].episode.velocities]+[s.episode.velocities[1:] for s in segments[1:]])
    graph = np.concatenate([s.episode.adjacency[:-1] for s in segments]+[segments[-1].episode.adjacency[-1:]])
    episode = AccelerationEpisodeResult(positions,velocities,
        np.concatenate([s.episode.requested_accelerations for s in segments]),
        np.concatenate([s.episode.accelerations for s in segments]),graph,
        replace(config,horizon=boundaries[-1]),seed)
    return SequenceResult(episode,stages,tuple(segments),np.asarray(boundaries),
        np.concatenate([np.full(stage.steps,index,dtype=int) for index,stage in enumerate(stages)]),
        np.concatenate([s.assignments[:-1] for s in segments]+[segments[-1].assignments[-1:]]),
        np.concatenate([s.phases for s in segments]),np.concatenate([s.local_targets for s in segments]),
        np.concatenate([s.stale_messages_per_step for s in segments]))


def evaluate_sequence(result, protocol=None):
    protocol = TransitionProtocol() if protocol is None else protocol
    episode = result.episode
    events,contacts,initial_count = collision_timeline(episode.positions,episode.config.physics.robot_radius)
    hold = math.ceil(protocol.hold_seconds/episode.config.physics.dt-1e-9)
    reports = []
    for index,(stage,segment) in enumerate(zip(result.stages,result.segments)):
        start,stop = (int(v) for v in result.boundaries[index:index+2])
        p,v = episode.positions[start:stop+1],episode.velocities[start:stop+1]
        errors = np.array([coverage_error(state,segment.targets) for state in p])
        _,_,initial_contact = collision_timeline(p[:1],episode.config.physics.robot_radius)
        contact_present = bool(initial_contact or contacts[start:stop].any())
        new_events = int(events[start:stop].sum())+(initial_count if index == 0 else 0)
        stable = ((errors <= protocol.error_threshold) &
            (np.linalg.norm(v,axis=-1).max(axis=1) <= protocol.speed_threshold) & segment.allocation_consensus)
        stabilization = None
        if not contact_present:
            for offset in range(len(p)-hold):
                if stable[offset:offset+hold+1].all():
                    stabilization = offset*episode.config.physics.dt
                    break
        reports.append(dict(index=index,shape=stage.mission.identifier,epoch=f'stage:{index}',
            start_time_s=start*episode.config.physics.dt,deadline_time_s=stop*episode.config.physics.dt,
            formation_error_m=float(errors[-1]),stabilization_time_s=stabilization,
            stabilization_absolute_time_s=(start*episode.config.physics.dt+stabilization) if stabilization is not None else None,
            collisions=new_events,contact_present=contact_present,initial_contact=bool(initial_contact),
            success=stabilization is not None,allocation_resolved=bool(segment.allocation_consensus[-1]),
            control_release_relative_time_s=(segment.reconnection_steps+segment.allocation_rounds)*episode.config.physics.dt,
            stale_packets_discarded=int(segment.stale_messages_per_step.sum()),
            packets_delivered=int(segment.messages_per_step.sum()),
            safety_infeasible_robot_steps=int(segment.safety_infeasible.sum()),
            safety_intervention_robot_steps=int(segment.safety_interventions.sum())))
    successes = [r for r in reports if r['success']]
    total_collisions = int(events.sum())+initial_count
    return dict(stages=reports,success=bool(len(successes)==len(reports) and total_collisions==0),
        stage_success_rate_percent=100*len(successes)/len(reports),successful_stages=len(successes),
        total_stages=len(reports),collisions_total=total_collisions,
        formation_error_mean_m=float(np.mean([r['formation_error_m'] for r in reports])),
        stabilization_time_mean_s=float(np.mean([r['stabilization_time_s'] for r in successes])) if successes else None,
        elapsed_time_s=float(episode.times[-1]),stale_packets_discarded=int(result.stale_messages_per_step.sum()),
        boundary_states_preserved=all(np.array_equal(a.episode.positions[-1],b.episode.positions[0]) and
                                      np.array_equal(a.episode.velocities[-1],b.episode.velocities[0])
                                      for a,b in zip(result.segments[:-1],result.segments[1:])))
