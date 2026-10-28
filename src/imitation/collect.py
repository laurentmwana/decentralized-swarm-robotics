"""Collecter les exemples du professeur sans entraîner de modèle."""
import argparse
import json

from src.reference.circle import CircleTeacher
from src.reference.metrics import CircleMetricConfig
from src.simulator.acceleration_episode import AccelerationEpisodeConfig
from .data import CollectionConfig, collect_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Collecte supervisée du professeur, sans entraînement')
    parser.add_argument('--simulator', default='config/acceleration-simulator.yaml')
    parser.add_argument('--teacher', default='config/circle-teacher.yaml')
    parser.add_argument('--metrics', default='config/circle-metrics.yaml')
    parser.add_argument('--collection', default='config/imitation-data.yaml')
    parser.add_argument('--output', default='outputs/imitation/circle-v1')
    args = parser.parse_args(argv)
    config = AccelerationEpisodeConfig.from_yaml(args.simulator)
    result = collect_dataset(config, CircleTeacher.from_yaml(args.teacher, config),
                             CircleMetricConfig.from_yaml(args.metrics),
                             CollectionConfig.from_yaml(args.collection), args.output)
    print(json.dumps(dict(status=result['status'], quality_status=result['quality_status'],
                         summary=result['summary']), indent=2))
    return result


if __name__ == '__main__':
    main()
