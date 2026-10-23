"""Configuration versionnée : aucun paramètre ne vit dans le code (AGENT.md règle 4).

Le `ConfigLoader` exige que **toute** clé d'une section soit présente dans le YAML
(pas de défaut silencieux — défaut relevé par l'audit : une clé absente prenait la
valeur par défaut de la dataclass sans rien signaler) et refuse toute clé inconnue
(clés mortes).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


class ConfigError(Exception):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"fichier de configuration absent : {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"contenu YAML invalide (dictionnaire attendu) : {path}")
    return data


def _build(cls: type, raw: dict[str, Any], origin: Path) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"{origin.name} : clés inconnues {sorted(unknown)} pour {cls.__name__}")
    missing = known - set(raw)
    if missing:
        raise ConfigError(f"{origin.name} : clés manquantes {sorted(missing)} pour {cls.__name__}")
    return cls(**{k: raw[k] for k in known})


@dataclass(frozen=True)
class SimulatorConfig:
    arena_size: float
    dt: float
    v_max: float
    r_c: float
    horizon: int
    failure_fraction: float
    record_positions: bool
    spawn_radius: float


@dataclass(frozen=True)
class ShapesConfig:
    render_size: int
    target_radius: float
    sampler_seed: int
    resolutions: tuple[int, ...]
    font_path: str
    geometries: tuple[str, ...]
    digits: tuple[int, ...]
    letters: tuple[str, ...]
    numbers: tuple[int, ...]
    splits: dict[str, dict[str, tuple[str, ...]]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolutions", tuple(int(r) for r in self.resolutions))
        object.__setattr__(self, "geometries", tuple(str(g) for g in self.geometries))
        object.__setattr__(self, "digits", tuple(int(d) for d in self.digits))
        object.__setattr__(self, "letters", tuple(str(l) for l in self.letters))
        object.__setattr__(self, "numbers", tuple(int(n) for n in self.numbers))
        object.__setattr__(
            self,
            "splits",
            {
                kind: {split: tuple(str(x) for x in values) for split, values in parts.items()}
                for kind, parts in self.splits.items()
            },
        )


@dataclass(frozen=True)
class MetricsConfig:
    success_threshold: float
    collision_radius: float
    regularity_reference: str

    def __post_init__(self) -> None:
        if self.success_threshold <= 0:
            raise ValueError(f"seuil de succes invalide : {self.success_threshold}")
        if self.collision_radius <= 0:
            raise ValueError(f"rayon de collision invalide : {self.collision_radius}")


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int
    embedding_dim: int
    message_rounds: int
    n_shapes: int
    aggregator: str
    activation: str
    use_target_geometry: bool
    init_seed: int
    geometry_prior: float = 1.0  # defaut seulement pour la relecture d'anciens snapshots ; model.yaml doit la declarer

    def __post_init__(self) -> None:
        if self.message_rounds not in (1, 2, 3):
            raise ValueError(f"message_rounds doit etre 1, 2 ou 3 (plan phase 4) : {self.message_rounds}")
        if self.aggregator not in ("mean", "normalized_sum", "attention"):
            raise ValueError(f"aggregateur inconnu : {self.aggregator!r}")


@dataclass(frozen=True)
class ExperimentConfig:
    simulator: SimulatorConfig
    shapes: ShapesConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "simulator": self.simulator.__dict__,
            "shapes": {
                k: (list(v) if isinstance(v, tuple) else v) for k, v in self.shapes.__dict__.items()
            },
        }


class ConfigLoader:
    """Charge les sections simulateur et formes depuis des fichiers YAML versionnés."""

    def __init__(self, config_dir: Path | str = "config") -> None:
        self.config_dir = Path(config_dir)

    def load_simulator(self, name: str = "simulator") -> SimulatorConfig:
        path = self.config_dir / f"{name}.yaml"
        return _build(SimulatorConfig, _load_yaml(path), path)

    def load_raw(self, name: str) -> dict[str, Any]:
        """Chargement brut pour les configurations d'expérience sans dataclass dédiée."""
        return _load_yaml(self.config_dir / f"{name}.yaml")

    def load_shapes(self, name: str = "shapes") -> ShapesConfig:
        path = self.config_dir / f"{name}.yaml"
        cfg = _build(ShapesConfig, _load_yaml(path), path)
        self._validate_split(cfg, path)
        return cfg

    def load_model(self, name: str = "model") -> ModelConfig:
        path = self.config_dir / f"{name}.yaml"
        return _build(ModelConfig, _load_yaml(path), path)

    def load_metrics(self, name: str = "metrics") -> MetricsConfig:
        path = self.config_dir / f"{name}.yaml"
        return _build(MetricsConfig, _load_yaml(path), path)

    def load(self) -> ExperimentConfig:
        return ExperimentConfig(simulator=self.load_simulator(), shapes=self.load_shapes())

    @staticmethod
    def _validate_split(cfg: ShapesConfig, origin: Path) -> None:
        universe = {
            "geometries": [str(g) for g in cfg.geometries],
            "digits": [str(d) for d in cfg.digits],
            "letters": [str(l) for l in cfg.letters],
            "numbers": [str(n) for n in cfg.numbers],
        }
        for kind, names in universe.items():
            parts = cfg.splits.get(kind)
            if parts is None:
                raise ConfigError(f"{origin.name} : découpage manquant pour le type {kind!r}")
            seen: list[str] = []
            for split, values in parts.items():
                unknown = set(values) - set(names)
                if unknown:
                    raise ConfigError(
                        f"{origin.name} : split {split!r} de {kind!r} contient des formes inconnues {sorted(unknown)}"
                    )
                seen += list(values)
            if len(seen) != len(set(seen)):
                raise ConfigError(f"{origin.name} : découpage {kind!r} avec des formes dans plusieurs splits")
            if set(seen) != set(names):
                raise ConfigError(f"{origin.name} : découpage {kind!r} incomplet ou surabondant")
