"""Transport de formation, repère mobile et obstacles perçus localement."""
from dataclasses import dataclass,asdict,replace
import json
import math
from pathlib import Path

import numpy as np

from .allocation import _integer
from .metrics import collision_events
from .obstacles import DiskWorld,wall_events
from .reconnection import ReconnectionConfig
from .shape_control import ShapeGains,execute_shape,evaluate_shape,ShapeMetricConfig,coverage_error,assigned_command
from .shapes import ShapeMission
from .safety import filter_command
from src.common.seeds import SeedManager
from src.simulator.acceleration_episode import AccelerationEpisode,AccelerationEpisodeResult,LocalObservation,_snapshot
from src.simulator.dynamics import _limit


@dataclass(frozen=True)
class MovingReference:
    points: tuple
    durations: tuple

    def __post_init__(self):
        points=np.asarray(self.points,dtype=float)
        durations=np.asarray(self.durations,dtype=float)
        if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
            raise ValueError('Au moins deux points 2D finis requis')
        if durations.shape!=(len(points)-1,) or not np.isfinite(durations).all() or (durations<=0).any():
            raise ValueError('Une durée positive par tronçon est requise')
        if (np.linalg.norm(np.diff(points,axis=0),axis=1)<=1e-12).any():
            raise ValueError('Deux points consécutifs doivent être distincts')
        object.__setattr__(self,'points',tuple(tuple(float(v) for v in p) for p in points))
        object.__setattr__(self,'durations',tuple(float(v) for v in durations))

    @property
    def duration(self):return sum(self.durations)

    def sample(self,time):
        if not math.isfinite(time) or time<0:raise ValueError('Temps relatif fini positif ou nul requis')
        if time>=self.duration:return np.array(self.points[-1]),np.zeros(2),np.zeros(2)
        cumulative=np.r_[0.,np.cumsum(self.durations)]
        leg=min(int(np.searchsorted(cumulative,time,side='right'))-1,len(self.durations)-1)
        duration=self.durations[leg];s=(time-cumulative[leg])/duration
        f=10*s**3-15*s**4+6*s**5
        df=(30*s**2-60*s**3+30*s**4)/duration
        ddf=(60*s-180*s**2+120*s**3)/duration**2
        start,end=np.asarray(self.points[leg]),np.asarray(self.points[leg+1])
        return start+f*(end-start),df*(end-start),ddf*(end-start)


@dataclass(frozen=True)
class NavigationProtocol:
    formation_threshold: float = .1
    transit_formation_limit: float = .2
    arrival_threshold: float = .1
    speed_threshold: float = .05
    hold_seconds: float = 1.
    sensor_range: float = 3.
    obstacle_clearance: float = .15

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in asdict(self).values()):
            raise ValueError('Seuils et paramètres de navigation finis positifs requis')
        if self.transit_formation_limit<self.formation_threshold:
            raise ValueError('La tolérance transitoire doit couvrir le seuil final')


def translated_shape(shape,center):
    center=np.asarray(center,dtype=float)
    return ShapeMission(shape.name,tuple(tuple(map(tuple,np.asarray(s)+center)) for s in shape.strokes),
                        shape.closed,shape.robot_count,shape.category)


def initial_positions(config,count,seed,world,*,minimum_distance=.45,clearance=.15,max_attempts=20000):
    _integer(count,'count',2);_integer(seed,'seed');_integer(max_attempts,'max_attempts',1)
    if not np.isfinite(minimum_distance) or minimum_distance<=2*config.physics.robot_radius:
        raise ValueError('Séparation initiale supérieure au diamètre requise')
    rng=SeedManager(seed).rng('navigation_initial_positions');positions=[]
    for _ in range(max_attempts):
        angle=rng.uniform(0,2*np.pi);radius=config.spawn_radius*np.sqrt(rng.uniform())
        candidate=radius*np.array([np.cos(angle),np.sin(angle)])
        gap=world.minimum_clearance(candidate[None,None,:],config.physics.robot_radius)
        if gap is not None and gap<clearance:continue
        if positions and np.linalg.norm(np.asarray(positions)-candidate,axis=1).min()<minimum_distance:continue
        positions.append(candidate)
        if len(positions)==count:return np.asarray(positions)
    raise ValueError('Domaine initial impossible dans le budget de tirages')


@dataclass(frozen=True)
class NavigationResult:
    episode: AccelerationEpisodeResult
    preparation: object
    reference: MovingReference
    world: DiskWorld
    offsets: np.ndarray
    slots: np.ndarray
    reference_positions: np.ndarray
    reference_velocities: np.ndarray
    reference_accelerations: np.ndarray
    local_targets: np.ndarray
    sensor_counts: np.ndarray
    safety_infeasible: np.ndarray
    safety_interventions: np.ndarray
    protocol: NavigationProtocol
    classical_preparation: bool = False

    def save(self,path):
        meta=dict(schema_version=1,kind='formation_navigation',config=asdict(self.episode.config),seed=self.episode.seed,
            reference=asdict(self.reference),obstacles=[asdict(o) for o in self.world.obstacles],protocol=asdict(self.protocol),
            preparation_steps=self.preparation.episode.config.horizon,
            preparation_policy='classical' if self.classical_preparation else 'selected_controller',
            wall_guard_half_extent=self.episode.config.physics.arena_size/2-self.episode.config.physics.robot_radius-self.protocol.obstacle_clearance,
            policy_frame='common_translating_frame_plus_classical_reference_acceleration')
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream,positions=self.episode.positions,velocities=self.episode.velocities,
                accelerations=self.episode.accelerations,requested_accelerations=self.episode.requested_accelerations,
                adjacency=self.episode.adjacency,times=self.episode.times,slots=self.slots,offsets=self.offsets,
                reference_positions=self.reference_positions,reference_velocities=self.reference_velocities,
                reference_accelerations=self.reference_accelerations,local_targets=self.local_targets,
                navigation_sensor_counts=self.sensor_counts,safety_infeasible=self.safety_infeasible,
                safety_interventions=self.safety_interventions,metadata=json.dumps(meta))


def execute_navigation(shape,reference,world,config,positions,*,navigation_steps,preparation_steps=400,
                       motion_policy=None,protocol=None,seed=0,classical_preparation=False):
    _integer(navigation_steps,'navigation_steps',1);_integer(preparation_steps,'preparation_steps',1)
    protocol=NavigationProtocol() if protocol is None else protocol
    gains=ShapeGains(detour_only_on_constraint=True);reconnection=ReconnectionConfig()
    for point in reference.points:translated_shape(shape,point).validate_environment(config,gains.safety_distance)
    def sense(position):
        vectors=world.sense(position,sensor_range=protocol.sensor_range,robot_radius=config.physics.robot_radius,
                            clearance=protocol.obstacle_clearance)
        return vectors
    start=np.asarray(reference.points[0])
    wall_half_extent=config.physics.arena_size/2-config.physics.robot_radius-protocol.obstacle_clearance
    preparation=execute_shape(translated_shape(shape,start),replace(config,horizon=preparation_steps),positions,
        gains=gains,reconnection=reconnection,safety_filter=True,motion_policy=None if classical_preparation else motion_policy,guard_motion=True,
        obstacle_observer=sense,wall_half_extent=wall_half_extent,seed=seed)
    p,v=preparation.episode.positions[-1].copy(),preparation.episode.velocities[-1].copy()
    offsets=preparation.targets-start
    count=shape.robot_count;slots=np.full(count,-1,dtype=int);ready=np.zeros(count,dtype=bool)
    for packet in preparation.final_packets:
        claims=packet.claims
        ready[packet.sender]=len(claims)==count and len({c.slot for c in claims})==count
        own=next(c for c in claims if c.owner==packet.sender)
        slots[packet.sender]=own.slot
    simulator=AccelerationEpisode(config)
    hp,hv,commands,bounded,graphs,targets=[p.copy()],[v.copy()],[],[],[],[]
    counts,failures,changes=[],[],[]
    for step in range(navigation_steps):
        center,goal_velocity,goal_acceleration=reference.sample(step*config.physics.dt)
        graph=simulator.adjacency(p);graphs.append(graph)
        step_commands=[];step_targets=p.copy();step_counts=[];step_failures=[];step_changes=[]
        for i in range(count):
            neighbors=graph[i]
            actual=LocalObservation(_snapshot(p[i]),_snapshot(v[i]),_snapshot(p[neighbors]-p[i]),
                _snapshot(v[neighbors]-v[i]),(preparation_steps+step)*config.physics.dt)
            if ready[i]:
                target=offsets[slots[i]]+center;step_targets[i]=target
                moving=LocalObservation(_snapshot(p[i]-center),_snapshot(v[i]-goal_velocity),
                                        actual.relative_positions,actual.relative_velocities,actual.time)
                relative_target=target-center
                nominal=(assigned_command(moving,relative_target,gains) if motion_policy is None else
                         np.asarray(motion_policy(moving,relative_target.copy(),3,count),dtype=float))
                if nominal.shape!=(2,) or not np.isfinite(nominal).all():
                    raise ValueError('Commande locale 2D finie requise')
                nominal=nominal+goal_acceleration
            else:nominal=-actual.velocity/config.physics.dt
            vectors=world.sense(p[i],sensor_range=protocol.sensor_range,robot_radius=config.physics.robot_radius,
                                clearance=protocol.obstacle_clearance)
            command,feasible=filter_command(actual,nominal,gains,config.physics.max_acceleration,obstacle_vectors=vectors,
                wall_half_extent=wall_half_extent)
            before=_limit(np.asarray(nominal)[None,:],config.physics.max_acceleration)[0]
            step_commands.append(command);step_counts.append(len(vectors));step_failures.append(not feasible)
            step_changes.append(np.linalg.norm(command-before)>1e-6)
        command=np.asarray(step_commands);commands.append(command.copy());bounded.append(_limit(command,config.physics.max_acceleration))
        p,v=simulator.engine.step(p,v,command)
        hp.append(p.copy());hv.append(v.copy());targets.append(step_targets)
        counts.append(step_counts);failures.append(step_failures);changes.append(step_changes)
    graphs.append(simulator.adjacency(p))
    base=preparation.episode
    episode=AccelerationEpisodeResult(np.concatenate((base.positions,np.asarray(hp)[1:])),
        np.concatenate((base.velocities,np.asarray(hv)[1:])),np.concatenate((base.requested_accelerations,np.asarray(commands))),
        np.concatenate((base.accelerations,np.asarray(bounded))),np.concatenate((base.adjacency[:-1],np.asarray(graphs))),
        replace(config,horizon=preparation_steps+navigation_steps),seed)
    samples=[reference.sample(t*config.physics.dt) for t in range(navigation_steps+1)]
    refp=np.concatenate((np.tile(start,(preparation_steps,1)),np.asarray([s[0] for s in samples])))
    refv=np.concatenate((np.zeros((preparation_steps,2)),np.asarray([s[1] for s in samples])))
    refa=np.concatenate((np.zeros((preparation_steps,2)),np.asarray([s[2] for s in samples])))
    return NavigationResult(episode,preparation,reference,world,offsets,slots,refp,refv,refa,
        np.concatenate((preparation.local_targets,np.asarray(targets))),np.asarray(counts),
        np.concatenate((preparation.safety_infeasible,np.asarray(failures))),
        np.concatenate((preparation.safety_interventions,np.asarray(changes))),protocol,classical_preparation)


def evaluate_navigation(result):
    episode=result.episode;protocol=result.protocol;start=result.preparation.episode.config.horizon
    positions=episode.positions[start:];velocities=episode.velocities[start:]
    centers=positions.mean(axis=1)-result.offsets.mean(axis=0)
    shape_errors=np.array([coverage_error(p-c,result.offsets) for p,c in zip(positions,centers)])
    tracking=np.array([coverage_error(p,result.offsets+ref) for p,ref in zip(positions,result.reference_positions[start:])])
    destination=np.linalg.norm(centers-np.asarray(result.reference.points[-1]),axis=-1)
    pair_collisions=collision_events(episode.positions,episode.config.physics.robot_radius)[0]
    obstacle_events,_,obstacle_initial=result.world.collision_timeline(episode.positions,episode.config.physics.robot_radius)
    obstacle_collisions=int(obstacle_events.sum())+obstacle_initial
    walls=wall_events(episode.positions,episode.config.physics.arena_size,episode.config.physics.robot_radius)
    total=pair_collisions+obstacle_collisions+walls
    prep_time=start*episode.config.physics.dt
    prep=evaluate_shape(result.preparation,ShapeMetricConfig(.1,.05,1.,prep_time))
    stable=((shape_errors<=protocol.formation_threshold)&(destination<=protocol.arrival_threshold)&
        (np.linalg.norm(velocities,axis=-1).max(axis=1)<=protocol.speed_threshold)&
        (np.arange(len(positions))*episode.config.physics.dt>=result.reference.duration-1e-9))
    hold=math.ceil(protocol.hold_seconds/episode.config.physics.dt-1e-9)
    arrival=None;transit_ok=bool(shape_errors.max()<=protocol.transit_formation_limit)
    if prep['success'] and total==0 and transit_ok:
        for step in range(len(positions)-hold):
            if stable[step:step+hold+1].all():arrival=prep_time+step*episode.config.physics.dt;break
    reasons=[]
    if not prep['success']:reasons.append('preparation_failed')
    if total:reasons.append('collision')
    if not transit_ok:reasons.append('in_transit_formation_error')
    if not stable[-hold-1:].all() or len(stable)<=hold:reasons.append('arrival_or_stabilization_deadline')
    return dict(success=arrival is not None,arrival_time_s=arrival,preparation_success=prep['success'],
        formation_error_m=float(shape_errors[-1]),max_navigation_formation_error_m=float(shape_errors.max()),
        mean_navigation_formation_error_m=float(shape_errors.mean()),reference_tracking_error_m=float(tracking[-1]),
        max_reference_tracking_error_m=float(tracking.max()),destination_error_m=float(destination[-1]),
        robot_collisions=pair_collisions,obstacle_collisions=obstacle_collisions,wall_collisions=walls,collisions=total,
        minimum_obstacle_clearance_m=result.world.minimum_clearance(episode.positions,episode.config.physics.robot_radius),
        navigation_obstacle_observations=int(result.sensor_counts.sum()),
        safety_infeasible_robot_steps=int(result.safety_infeasible.sum()),
        safety_intervention_robot_steps=int(result.safety_interventions.sum()),
        elapsed_time_s=float(episode.times[-1]),failure_reasons=reasons)
