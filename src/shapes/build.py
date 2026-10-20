"""Interface de ligne de commande du générateur de formes."""

import argparse

from .generator import ShapeDatasetConfig, build_dataset


def main() -> None:
    """Génère le jeu de données configuré et affiche le chemin de l'archive."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shapes.yaml")
    parser.add_argument("--output", default="data/shapes")
    arguments = parser.parse_args()
    print(build_dataset(ShapeDatasetConfig.from_yaml(arguments.config), arguments.output))


if __name__ == "__main__":
    main()
