"""Professeur local de formes, après une phase d'affectation stationnaire explicite."""
from dataclasses import dataclass
import math

import numpy as np

from .allocation import ClaimPacket, LocalSlotAllocator, _integer
from .metrics import CircleMetricConfig, collision_events
from .safety import filter_command
from .reconnection import ReconnectionConfig, rendezvous_targets
from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeResult, LocalObservation, _snapshot
from src.simulator.dynamics import _limit, _vectors


@dataclass(frozen=True)
class ShapeGains:
    attraction: float = 6.
    damping: float = 4.
    avoidance: float = 60.
    safety_distance: float = .4
    barrier_alpha: float = 3.
    detour_distance: float = 1.
    detour_gain: float = 3.
    detour_only_on_constraint: bool = False

    def __post_init__(self):
        if not isinstance(self.detour_only_on_constraint, bool):
            raise ValueError('detour_only_on_constraint booléen requis')
        if not all(np.isfinite(v) and v > 0 for v in (self.attraction, self.damping, self.avoidance, self.safety_distance,
                                                       self.barrier_alpha, self.detour_distance, self.detour_gain)):
            raise ValueError('Gains et distance de sécurité finis et positifs requis')
        if self.detour_distance <= self.safety_distance:
            raise ValueError('Distance de détour supérieure à la sécurité requise')


@dataclass(frozen=True)
class ShapeMetricConfig(CircleMetricConfig):
    """Même protocole SI que le cercle ; erreur de couverture des places."""


def assigned_command(observation, target, gains):
    target = np.asarray(target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all():
        raise ValueError('Place cible 2D finie requise')
    command = gains.attraction*(target-observation.position)-gains.damping*observation.velocity
    for relative in observation.relative_positions:
        distance = np.linalg.norm(relative)
        if distance > 1e-12:
            command -= gains.avoidance*max(0., gains.safety_distance-distance)*relative/distance
    return command


@dataclass(frozen=True)
class ShapeEpisodeResult:
    episode: AccelerationEpisodeResult
    targets: np.ndarray
    assignments: np.ndarray
    knowledge_sizes: np.ndarray
    allocation_consensus: np.ndarray
    messages_per_step: np.ndarray
    allocation_rounds: int
    safety_infeasible: np.ndarray
    reconnection_steps: int = 0
    phases: np.ndarray | None = None
    local_targets: np.ndarray | None = None
    safety_interventions: np.ndarray | None = None
    stale_messages_per_step: np.ndarray | None = None
    final_packets: tuple = ()


def execute_shape(mission, config, positions, *, gains=None, allocation_rounds=None, seed=0, safety_filter=False,
                  reconnection=None, motion_policy=None, guard_motion=None, initial_velocities=None,
                  time_offset=0., mission_epoch=None, previous_packets=(), allow_short_window=False,
                  obstacle_observer=None):
    gains = ShapeGains() if gains is None else gains
    mission.validate_environment(config, gains.safety_distance)
    if not isinstance(safety_filter, bool):
        raise ValueError('safety_filter booléen requis')
    guard_motion = safety_filter if guard_motion is None else guard_motion
    if not isinstance(guard_motion, bool) or (guard_motion and not safety_filter):
        raise ValueError('Protection de mouvement booléenne et filtre compatible requis')
    if motion_policy is not None and not callable(motion_policy):
        raise ValueError('Politique de mouvement appelable requise')
    if obstacle_observer is not None and not callable(obstacle_observer):
        raise ValueError('Capteur local d’obstacles appelable requis')
    if safety_filter and config.communication_radius < gains.detour_distance:
        raise ValueError('La portée doit couvrir la distance de détour')
    positions = _vectors(positions, 'positions').copy()
    n = mission.robot_count
    if positions.shape != (n, 2):
        raise ValueError('Positions initiales incompatibles avec la consigne')
    _integer(seed, 'seed')
    rounds = n*n if allocation_rounds is None else allocation_rounds
    _integer(rounds, 'allocation_rounds', 1)
    gather_steps = brake_steps = 0
    intermediate = None
    if reconnection is not None:
        if not isinstance(reconnection, ReconnectionConfig) or not safety_filter:
            raise ValueError('Reconnexion typée et filtre de sécurité requis')
        gather_steps, brake_steps = reconnection.steps(config.physics.dt)
        intermediate = rendezvous_targets(n, config, reconnection, gains.safety_distance)
    allocation_start = gather_steps+brake_steps
    release = allocation_start+rounds
    if not isinstance(allow_short_window, bool):
        raise ValueError('allow_short_window booléen requis')
    if release >= config.horizon and not allow_short_window:
        raise ValueError('L’horizon doit laisser du temps après l’affectation')
    if not math.isfinite(time_offset) or time_offset < 0:
        raise ValueError('Temps initial fini positif ou nul requis')
    if mission_epoch is not None and (not isinstance(mission_epoch, str) or not mission_epoch):
        raise ValueError('Version de mission non vide requise')
    previous_packets = tuple(previous_packets)
    if any(not isinstance(p, ClaimPacket) or p.sender >= n for p in previous_packets):
        raise ValueError('Paquets précédents incompatibles avec l’effectif')
    if len({p.sender for p in previous_packets}) != len(previous_packets) or (previous_packets and mission_epoch is None):
        raise ValueError('Versions distinctes et un ancien paquet par émetteur requis')
    pending = {p.sender:p for p in previous_packets}
    simulator = AccelerationEpisode(config)
    velocity = np.zeros_like(positions) if initial_velocities is None else _vectors(initial_velocities,'velocities').copy()
    if velocity.shape != positions.shape:
        raise ValueError('Vitesses initiales incompatibles avec les positions')
    simulator.engine.step(positions, velocity, velocity)
    targets = mission.reference_points()
    agents = None if reconnection is not None else [LocalSlotAllocator(i, positions[i], targets,mission_epoch=mission_epoch) for i in range(n)]
    hp, hv, requested, bounded, graphs, messages = [positions.copy()], [velocity.copy()], [], [], [], []
    assignments = [[a.slot for a in agents] if agents is not None else [-1]*n]
    knowledge = [[a.knowledge_size for a in agents] if agents is not None else [0]*n]
    consensus = [False]
    infeasible = []
    phases = []
    local_targets = []
    interventions = []
    stale_messages = []
    def motion(observation, target, phase):
        command = (assigned_command(observation, target, gains) if motion_policy is None else
                   np.asarray(motion_policy(observation, target.copy(), phase, n), dtype=float))
        if command.shape != (2,) or not np.isfinite(command).all():
            raise ValueError('La politique de mouvement doit produire deux accélérations finies')
        return command
    safety_cache={}
    def protect(observation, command, robot_id):
        before = _limit(np.asarray(command)[None, :], config.physics.max_acceleration)[0]
        obstacles=() if obstacle_observer is None else obstacle_observer(observation.position.copy())
        signature=tuple(np.asarray(value,dtype=float).tobytes() for value in (
            observation.position,observation.velocity,observation.relative_positions,
            observation.relative_velocities,command,obstacles))
        previous=safety_cache.get(robot_id)
        if previous is not None and previous[0]==signature:
            return previous[1].copy(),previous[2],previous[3]
        after, feasible = filter_command(observation, command, gains, config.physics.max_acceleration,obstacle_vectors=obstacles)
        changed=np.linalg.norm(after-before)>1e-6
        safety_cache[robot_id]=(signature,after.copy(),feasible,changed)
        return after, feasible, changed
    for step in range(config.horizon):
        discarded = 0
        step_targets = positions.copy()
        failures = np.zeros(n, dtype=bool)
        changed = np.zeros(n, dtype=bool)
        graph = simulator.adjacency(positions)
        graphs.append(graph)
        if step < allocation_start:
            phase = 0 if step < gather_steps else 1
            commands = []
            for i in range(n):
                neighbors = graph[i]
                observation = LocalObservation(_snapshot(positions[i]), _snapshot(velocity[i]),
                    _snapshot(positions[neighbors]-positions[i]), _snapshot(velocity[neighbors]-velocity[i]),
                    time_offset+step*config.physics.dt)
                command = (motion(observation, intermediate[i], phase) if phase == 0 else
                           -observation.velocity/config.physics.dt)
                if phase == 0:
                    step_targets[i] = intermediate[i]
                if phase != 0 or guard_motion:
                    command, feasible, changed[i] = protect(observation, command,i)
                    failures[i] = not feasible
                commands.append(command)
            commands = np.asarray(commands)
            messages.append(0)
        elif step < release:
            phase = 2
            if agents is None:
                agents = [LocalSlotAllocator(i, positions[i], targets,mission_epoch=mission_epoch) for i in range(n)]
            packets = tuple(a.broadcast() for a in agents)
            delivered = 0
            for i, agent in enumerate(agents):
                neighbors = np.flatnonzero(graph[i])
                inbox = [packets[j] for j in neighbors]
                if step == allocation_start:
                    inbox.extend(pending[j] for j in neighbors if j in pending)
                _, stale = agent.advance_current_mission(inbox)
                discarded += stale; delivered += len(inbox)
            commands = -velocity/config.physics.dt if reconnection is not None else np.zeros_like(positions)
            if reconnection is not None:
                for i in range(n):
                    neighbors = graph[i]
                    observation = LocalObservation(_snapshot(positions[i]), _snapshot(velocity[i]),
                        _snapshot(positions[neighbors]-positions[i]), _snapshot(velocity[neighbors]-velocity[i]),
                        time_offset+step*config.physics.dt)
                    commands[i], feasible, changed[i] = protect(observation, commands[i],i)
                    failures[i] = not feasible
            messages.append(delivered)
        else:
            phase = 3
            commands = []
            for i, agent in enumerate(agents):
                # Aucun verdict global transmis au robot : état local seulement.
                if not agent.locally_consistent:
                    commands.append(np.zeros(2))
                    continue
                neighbors = graph[i]
                observation = LocalObservation(_snapshot(positions[i]), _snapshot(velocity[i]),
                    _snapshot(positions[neighbors]-positions[i]), _snapshot(velocity[neighbors]-velocity[i]),
                    time_offset+step*config.physics.dt)
                command = motion(observation, targets[agent.slot], phase)
                step_targets[i] = targets[agent.slot]
                if guard_motion:
                    command, feasible, changed[i] = protect(observation, command,i)
                    failures[i] = not feasible
                commands.append(command)
            commands = np.asarray(commands)
            messages.append(0)
        requested.append(commands.copy())
        phases.append(phase)
        local_targets.append(step_targets)
        infeasible.append(failures)
        interventions.append(changed)
        stale_messages.append(discarded)
        bounded.append(_limit(commands, config.physics.max_acceleration))
        positions, velocity = simulator.engine.step(positions, velocity, commands)
        hp.append(positions.copy()); hv.append(velocity.copy())
        assignments.append([a.slot for a in agents] if agents is not None else [-1]*n)
        knowledge.append([a.knowledge_size for a in agents] if agents is not None else [0]*n)
        # Mesure hors contrôleur ; ne pilote pas les commandes ni l'arrêt.
        agreed = agents is not None and len({a.broadcast().claims for a in agents}) == 1
        consensus.append(step+1 >= release and agreed and all(a.locally_consistent for a in agents))
    graphs.append(simulator.adjacency(positions))
    episode = AccelerationEpisodeResult(np.asarray(hp), np.asarray(hv), np.asarray(requested), np.asarray(bounded),
                                        np.asarray(graphs), config, seed)
    return ShapeEpisodeResult(episode, targets, np.asarray(assignments), np.asarray(knowledge),
                              np.asarray(consensus), np.asarray(messages), rounds, np.asarray(infeasible),
                              allocation_start, np.asarray(phases, dtype=np.int8), np.asarray(local_targets), np.asarray(interventions),
                              np.asarray(stale_messages), tuple(a.broadcast() for a in agents) if agents is not None else ())


def coverage_error(positions, targets):
    positions, targets = np.asarray(positions, dtype=float), np.asarray(targets, dtype=float)
    if positions.shape != targets.shape or positions.ndim != 2 or positions.shape[1] != 2 or len(positions) < 1:
        raise ValueError('Positions et places compatibles (N, 2) requises')
    if not np.isfinite(positions).all() or not np.isfinite(targets).all():
        raise ValueError('Positions et places finies requises')
    distances = np.linalg.norm(positions[:, None]-targets[None, :], axis=-1)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()))


def evaluate_shape(result, protocol):
    episode = result.episode
    if episode.times[-1]+1e-12 < protocol.deadline_seconds:
        raise ValueError('Épisode trop court pour le protocole')
    end = math.floor(protocol.deadline_seconds/episode.config.physics.dt+1e-9)
    positions, velocities = episode.positions[:end+1], episode.velocities[:end+1]
    errors = np.array([coverage_error(p, result.targets) for p in positions])
    collisions, contacts, initial_collision = collision_events(positions, episode.config.physics.robot_radius)
    stable = ((errors <= protocol.error_threshold) &
              (np.linalg.norm(velocities, axis=-1).max(axis=1) <= protocol.speed_threshold) &
              result.allocation_consensus[:end+1])
    hold = math.ceil(protocol.hold_seconds/episode.config.physics.dt-1e-9)
    stabilization = None
    if collisions == 0:
        for start in range(len(positions)-hold):
            if stable[start:start+hold+1].all() and not contacts[start:start+hold].any():
                stabilization = float(start*episode.config.physics.dt)
                break
    start = result.reconnection_steps
    stop = min(start+result.allocation_rounds, end)
    allocation_graphs = episode.adjacency[start:stop+1]
    graph_changes = int(np.any(allocation_graphs[1:] != allocation_graphs[:-1], axis=(1, 2)).sum())
    allocation_speed = float(np.linalg.norm(episode.velocities[start:stop+1], axis=-1).max()) if stop > start else None
    return dict(formation_error_m=float(errors[-1]), stabilization_time_s=stabilization, collisions=collisions,
                success=stabilization is not None, initial_collision=initial_collision,
                allocation_resolved=bool(result.allocation_consensus[end]),
                allocation_time_s=result.allocation_rounds*episode.config.physics.dt,
                reconnection_time_s=result.reconnection_steps*episode.config.physics.dt,
                control_release_time_s=(result.reconnection_steps+result.allocation_rounds)*episode.config.physics.dt,
                allocation_graph_changed_steps=graph_changes, max_allocation_speed_m_s=allocation_speed,
                allocation_stationary=allocation_speed is not None and allocation_speed <= 1e-7 and graph_changes == 0,
                packets_delivered=int(result.messages_per_step[:end].sum()),
                safety_infeasible_robot_steps=int(result.safety_infeasible[:end].sum()),
                safety_intervention_robot_steps=int(result.safety_interventions[:end].sum()) if result.safety_interventions is not None else 0,
                motion_safety_intervention_robot_steps=int(result.safety_interventions[:end][np.isin(result.phases[:end], (0, 3))].sum()) if result.safety_interventions is not None else 0)
