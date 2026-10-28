"""Exécuter le professeur circulaire sans entraînement : python -m src.reference.demo."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from src.simulator.acceleration_episode import AccelerationEpisode, AccelerationEpisodeConfig
from .circle import CircleTeacher


def main(argv=None):
    parser = argparse.ArgumentParser(description='Simulation circulaire classique, sans entraînement')
    parser.add_argument('--simulator', default='config/acceleration-simulator.yaml')
    parser.add_argument('--teacher', default='config/circle-teacher.yaml')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=Path('outputs/circle-reference'))
    args = parser.parse_args(argv)
    cfg = AccelerationEpisodeConfig.from_yaml(args.simulator)
    teacher = CircleTeacher.from_yaml(args.teacher, cfg)
    result = AccelerationEpisode(cfg).execute(teacher, teacher.mission.robot_count, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    result.save(args.output / 'episode.npz')
    (args.output / 'teacher.json').write_text(json.dumps(asdict(teacher), indent=2), encoding='utf-8')
    print(f'Trajectoires : {args.output / "episode.npz"}')
    return result


if __name__ == '__main__':
    main()
