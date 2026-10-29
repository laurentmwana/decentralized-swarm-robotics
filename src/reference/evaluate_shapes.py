"""Pilote reproductible du professeur à places attribuées, sans apprentissage."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .allocation import _integer
from .catalog import catalog_entries
from .metrics import summarize
from .shape_control import ShapeGains, ShapeMetricConfig, execute_shape, evaluate_shape
from .shapes import named_shape
from .reconnection import ReconnectionConfig
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from src.imitation.data import nonoverlapping_positions


def evaluate_shapes(output, *, names=None, seeds=(7000,), robot_count=12, width=6.,
                    horizon=400, communication_radius=3., safety_filter=False, initialization='ring',
                    reconnect=False, risk_detours=False):
    names = [e['identifier'] for e in catalog_entries()] if names is None else list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Formes non vides et non dupliquées requises')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Graines non vides et uniques requises')
    for seed in seeds: _integer(seed, 'seed')
    _integer(robot_count, 'robot_count', 2)
    if initialization not in ('ring', 'dispersed'):
        raise ValueError('Initialisation ring ou dispersed requise')
    config = replace(AccelerationEpisodeConfig.from_yaml('config/acceleration-simulator.yaml'),
                     horizon=horizon, communication_radius=communication_radius)
    missions = [named_shape(name, robot_count, width=width) for name in names]
    if len({m.identifier for m in missions}) != len(missions):
        raise ValueError('Alias de formes dupliquées')
    if not isinstance(reconnect, bool) or not isinstance(risk_detours, bool):
        raise ValueError('Options de reconnexion et détour booléennes requises')
    reconnection = ReconnectionConfig() if reconnect else None
    if reconnect and not safety_filter:
        raise ValueError('La reconnexion exige le filtre de sécurité')
    extra_steps = sum(reconnection.steps(config.physics.dt)) if reconnect else 0
    if horizon <= extra_steps+robot_count*robot_count:
        raise ValueError('Horizon insuffisant après les tours d’affectation')
    protocol = ShapeMetricConfig(.1, .05, 1., horizon*config.physics.dt)
    gains = ShapeGains(detour_only_on_constraint=risk_detours)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    payload = dict(schema_version=1, status='incomplete', kind='stationary_allocation_then_local_pd_teacher',
                   simulator=asdict(config), protocol=asdict(protocol), gains=asdict(gains),
                   allocation=dict(rounds=robot_count**2, round_seconds=config.physics.dt,
                                   unique_ids='0_to_N_minus_1', priority='lower_id_first',
                                   packets='one_hop_per_round_claims_only'),
                   initialization=dict(kind=initialization, minimum_separation_m=.45, max_spawn_attempts=20000),
                   robot_count=robot_count, width=width, seeds=seeds,
                   safety_filter=safety_filter,
                   reconnection=asdict(reconnection) if reconnection is not None else None,
                   phase_names={0:'rendezvous', 1:'braking', 2:'allocation', 3:'formation'},
                   shapes=[m.identifier for m in missions], reports=[])
    def save():
        (output/'evaluation.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    save()
    for seed in seeds:
        rotation = np.random.default_rng(seed).uniform(0, 2*np.pi)
        angles = rotation+np.arange(robot_count)*2*np.pi/robot_count
        radius = .45/(2*np.sin(np.pi/robot_count))
        initial = (radius*np.column_stack((np.cos(angles), np.sin(angles))) if initialization == 'ring' else
                   nonoverlapping_positions(config, robot_count, seed, .45, 20000))
        # Consigner les conditions AVANT le premier épisode de cette graine.
        payload.setdefault('initial_states', {})[str(seed)] = initial.tolist()
        save()
        for mission in missions:
            entry = dict(seed=seed, shape=mission.identifier, mission=asdict(mission))
            try:
                mission.validate_environment(config, gains.safety_distance)
            except ValueError as error:
                entry.update(status='infeasible', reason=str(error))
                payload['reports'].append(entry); save()
                continue
            result = execute_shape(mission, config, initial, gains=gains, seed=seed, safety_filter=safety_filter,
                                   reconnection=reconnection)
            prefix = f'{mission.identifier}-{seed}'
            result.episode.save(output/f'{prefix}.npz')
            np.savez_compressed(output/f'{prefix}-allocation.npz', targets=result.targets,
                assignments=result.assignments, knowledge_sizes=result.knowledge_sizes,
                allocation_consensus=result.allocation_consensus, messages_per_step=result.messages_per_step,
                safety_infeasible=result.safety_infeasible, phases=result.phases)
            entry.update(status='evaluated', metrics=evaluate_shape(result, protocol), trajectory=f'{prefix}.npz',
                         allocation_trace=f'{prefix}-allocation.npz')
            payload['reports'].append(entry); save()
    measured = [r['metrics'] for r in payload['reports'] if r['status'] == 'evaluated']
    payload.update(status='complete', summary=summarize(measured) if measured else None,
                   requested_episodes=len(missions)*len(seeds), infeasible_episodes=sum(r['status']=='infeasible' for r in payload['reports']),
                   success_rate_all_requested_percent=100*sum(r['success'] for r in measured)/(len(missions)*len(seeds)))
    save()
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shapes', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int, default=[7000])
    parser.add_argument('--robots', type=int, default=12)
    parser.add_argument('--width', type=float, default=6.)
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--communication-radius', type=float, default=3.)
    parser.add_argument('--safety-filter', action='store_true')
    parser.add_argument('--initialization', choices=['ring', 'dispersed'], default='ring')
    parser.add_argument('--reconnect', action='store_true')
    parser.add_argument('--risk-detours', action='store_true', help='Activer le détour seulement si la commande viole une contrainte')
    args = parser.parse_args(argv)
    result = evaluate_shapes(args.output, names=args.shapes, seeds=args.seeds, robot_count=args.robots,
                             width=args.width, horizon=args.horizon, communication_radius=args.communication_radius,
                             safety_filter=args.safety_filter, initialization=args.initialization,
                             reconnect=args.reconnect, risk_detours=args.risk_detours)
    print(json.dumps(dict(summary=result['summary'], infeasible=result['infeasible_episodes'],
                         requested=result['requested_episodes']), indent=2))


if __name__ == '__main__':
    main()
