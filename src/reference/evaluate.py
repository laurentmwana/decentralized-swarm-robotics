"""Campagne circulaire reproductible sans entraînement."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig
from .circle import CircleTeacher
from .metrics import CircleMetricConfig, evaluate_circle, summarize


def main(argv=None):
    parser = argparse.ArgumentParser(description='Évaluer le professeur circulaire sans entraînement')
    parser.add_argument('--simulator', default='config/acceleration-simulator.yaml')
    parser.add_argument('--teacher', default='config/circle-teacher.yaml')
    parser.add_argument('--metrics', default='config/circle-metrics.yaml')
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--output', type=Path, default=Path('outputs/circle-evaluation'))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error('Graines distinctes et positives ou nulles requises')
    cfg = AccelerationEpisodeConfig.from_yaml(args.simulator)
    protocol = CircleMetricConfig.from_yaml(args.metrics)
    cfg = replace(cfg, horizon=math.ceil(protocol.deadline_seconds / cfg.physics.dt))
    teacher = CircleTeacher.from_yaml(args.teacher, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in args.seeds:
        result = AccelerationEpisode(cfg).execute(teacher, teacher.mission.robot_count, seed)
        result.save(args.output / f'episode-{seed}.npz')
        reports.append(dict(seed=seed, **evaluate_circle(result, teacher.mission, protocol)))
    payload = dict(simulator=asdict(cfg), teacher=asdict(teacher), protocol=asdict(protocol),
                   reports=reports, summary=summarize(reports))
    (args.output / 'report.json').write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload['summary'], indent=2))
    return payload


if __name__ == '__main__':
    main()
