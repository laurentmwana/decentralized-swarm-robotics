"""Synthèse phase 5 : courbes de perte, tableau par variante, budget, cercle formé.

Lit les dossiers standards outputs/training/<variant>/seed_<n>/ (losses.csv,
validation.csv, summary.json, best.pt) — les tableaux sont construits sans relire les
tenseurs. Usage : python -m src.reports.training_summary
"""

from __future__ import annotations

import csv
import json
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from src.common.config import ModelConfig, SimulatorConfig
from src.controllers.gnn import GNNController
from src.simulator.episode import simulate_episode
from src.training.checkpoint import CheckpointManager
from src.training.train import build_model

INPUT = Path("outputs/training")
FIGURES = Path("figures")
TERM_KEYS = ["total", "shape", "coverage", "collisions", "motion"]


def _variants(cfg_training: dict) -> list[str]:
    return sorted(cfg_training["variants"])


def _saved_config(output_dir: Path) -> dict:
    """Lit exclusivement la configuration effective conservée avec les poids."""
    return yaml.safe_load((output_dir / "effective_config.yaml").read_text(encoding="utf-8"))


def _load_runs(variant: str) -> dict[int, Path]:
    """Refuse une synthèse partielle : chaque graine attendue doit être complète."""
    output_dir = INPUT / variant
    if not output_dir.exists():
        return {}
    runs, expected, reference_config = {}, None, None
    required = (
        "summary.json", "best.pt", "losses.csv", "validation.csv", "effective_config.yaml",
        "seeds.json", "version.txt", "targets.json", "status.json",
    )
    for subdir in sorted(output_dir.glob("seed_*")):
        seed = int(subdir.name.removeprefix("seed_"))
        for name in required:
            if not (subdir / name).is_file():
                raise FileNotFoundError(f"artefact manquant pour la graine {seed} : {subdir / name}")
        cfg = _saved_config(subdir)
        comparable = {**cfg, "model": {key: value for key, value in cfg["model"].items() if key != "init_seed"}}
        if reference_config is not None and comparable != reference_config:
            raise ValueError(f"configurations différentes entre graines de {variant}")
        reference_config = comparable
        profile = cfg["training"]["profiles"][cfg["profile"]]
        requested = {int(value) for value in profile["seeds"]}
        if expected is not None and expected != requested:
            raise ValueError(f"graines incompatibles entre les runs de {variant}")
        expected = requested
        seeds = json.loads((subdir / "seeds.json").read_text(encoding="utf-8"))
        status = json.loads((subdir / "status.json").read_text(encoding="utf-8"))
        summary = json.loads((subdir / "summary.json").read_text(encoding="utf-8"))
        if status["status"] != "completed":
            raise ValueError(f"entraînement incomplet : {subdir}")
        if cfg["variant"] != variant or seeds["seed"] != seed or summary["seed"] != seed:
            raise ValueError(f"identité du run incohérente : {subdir}")
        if set(seeds["training_seeds"]) != requested or seeds["validation_seeds"] != profile["validation_seeds"]:
            raise ValueError(f"provenance des graines incohérente : {subdir}")
        runs[seed] = subdir
    if not runs or set(runs) != expected:
        missing = sorted((expected or set()) - set(runs))
        raise FileNotFoundError(f"graines manquantes pour {variant} : {missing or 'aucun run complet'}")
    return runs


def _columns(path: Path, key: str) -> np.ndarray:
    with path.open(encoding="utf-8") as fh:
        return np.array([float(row[key]) for row in csv.DictReader(fh)])


def run() -> dict:
    variants = sorted(path.name for path in INPUT.iterdir() if path.is_dir() and any(path.glob("seed_*"))) if INPUT.exists() else []
    if not variants:
        return {"rows": []}
    all_runs = {variant: _load_runs(variant) for variant in variants}
    saved = _saved_config(next(iter(all_runs[variants[0]].values())))
    cfg_training = saved["training"]
    if set(variants) != set(_variants(cfg_training)):
        raise FileNotFoundError("variantes manquantes ou inattendues dans la campagne sauvegardée")
    for runs in all_runs.values():
        candidate = _saved_config(next(iter(runs.values())))
        if candidate["training"] != saved["training"] or candidate["profile"] != saved["profile"]:
            raise ValueError("budgets différents entre variantes de la campagne")
        if candidate["simulator"] != saved["simulator"] or candidate["shapes"] != saved["shapes"]:
            raise ValueError("simulateur ou formes différents entre variantes")
    # Vérification avant toute écriture de tableaux ou figures.
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant, runs in all_runs.items():
        summary = [json.loads((d / "summary.json").read_text(encoding="utf-8")) for d in runs.values()]
        final_losses = np.array([[r["final_loss"][key] for key in TERM_KEYS] for r in summary])
        scores = np.array([r["best_validation_score"] for r in summary])
        best_iterations = np.array([r["best_iteration"] for r in summary])
        durations = np.array([r["duration_seconds"] for r in summary])
        successes, collisions = [], []
        for seed, output_dir in runs.items():
            with (output_dir / "validation.csv").open(encoding="utf-8") as fh:
                val = list(csv.DictReader(fh))
            best = min(val, key=lambda l: float(l["selection_score"]))
            successes.append(float(best["validation_success"]))
            collisions.append(float(best["validation_collision_rate"]))
        degrees_of_freedom = 1 if len(runs) > 1 else 0
        row = {
            "variant": variant,
            "seed_count": len(runs),
            "final_total_loss": f"{final_losses[:, 0].mean():.3f} ± {final_losses[:, 0].std(ddof=degrees_of_freedom):.3f}",
            **{
                f"loss_{key}": f"{final_losses[:, i].mean():.3f} ± {final_losses[:, i].std(ddof=degrees_of_freedom):.3f}"
                for i, key in enumerate(TERM_KEYS[1:], start=1)
            },
            "validation_distance": f"{scores.mean():.3f} ± {scores.std(ddof=degrees_of_freedom):.3f}",
            "validation_success_rate": f"{np.mean(successes):.2f}",
            "validation_collision_rate": f"{np.mean(collisions):.3f}",
            "best_iteration": f"{best_iterations.mean():.0f}",
            "total_duration_seconds": f"{durations.sum():.0f}",
        }
        rows.append(row)
        _loss_figure(variant, runs)

    if rows:
        with (INPUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _budget(cfg_training, rows, saved["profile"])
        _circle_figure(cfg_training)
    return {"rows": rows}


def _loss_figure(variant: str, runs: dict[int, Path]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TERM_KEYS), figsize=(4 * len(TERM_KEYS), 3.4), sharex=True)
    for seed, output_dir in sorted(runs.items()):
        iterations = np.arange(1, len(_columns(output_dir / "losses.csv", "total")) + 1)
        for axis, key in zip(axes, TERM_KEYS):
            axis.plot(iterations, _columns(output_dir / "losses.csv", key), linewidth=1.2, label=f"graine {seed}")
    for axis, key in zip(axes, TERM_KEYS):
        axis.set_title(key)
        axis.set_xlabel("iteration")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{variant} — pertes par itération, graines superposées")
    fig.tight_layout()
    path = FIGURES / f"training_losses_{variant}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[synthese] figure : {path}")


def _budget(cfg_training: dict, rows: list[dict], profile: str = "standard") -> None:
    profile_config = cfg_training["profiles"][profile]
    text = [
        "# Budget de reglage (phase 5)",
        "",
        "La campagne sauvegardee utilise une configuration commune d'hyperparametres,",
        "les memes iterations, la meme taille d'essaim et les memes graines par variante.",
        "Les poids de perte proviennent des configurations effectives sauvegardees.",
        "Ce rapport ne recense pas les essais historiques ni leur cout : il ne prouve",
        "pas a lui seul l'egalite du budget total de reglage.",
        "",
        f"- Iterations : {profile_config['iterations']} ; robots : {profile_config['n_robots']} ; horizon : {profile_config['horizon']}",
        f"- Graines : {profile_config['seeds']} ; configurations sauvegardees par variante : 1",
        "",
        "| variante | n_graines | duree totale (s) |",
        "|---|---|---|",
    ]
    for row in rows:
        text.append(f"| {row['variant']} | {row['seed_count']} | {row['total_duration_seconds']} |")
    text += [
        "",
        f"Material : {platform.system()} {platform.release()}, {torch.__version__} (CPU), python {platform.python_version()}.",
        "",
    ]
    (INPUT / "budget.md").write_text("\n".join(text), encoding="utf-8")
    print(f"[synthese] budget : {INPUT / 'budget.md'}")


def _circle_figure(cfg_training: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = _variants(cfg_training)
    all_runs = {variant: _load_runs(variant) for variant in variants}
    if any(not runs for runs in all_runs.values()):
        raise FileNotFoundError("une variante de la campagne sauvegardée est absente")
    fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 4.2))
    axes = np.atleast_1d(axes)
    for axis, variant in zip(axes, variants):
        runs = all_runs[variant]
        summary = {g: json.loads((d / "summary.json").read_text(encoding="utf-8")) for g, d in runs.items()}
        representative = sorted(summary, key=lambda g: (summary[g]["best_validation_score"], g))[len(summary) // 2]
        output_dir = runs[representative]
        saved = _saved_config(output_dir)
        cfg_model = ModelConfig(**saved["model"])
        cfg_sim = SimulatorConfig(**saved["simulator"])
        profile = saved["training"]["profiles"][saved["profile"]]
        targets = json.loads((output_dir / "targets.json").read_text(encoding="utf-8"))
        target_record = targets["training_target"]
        target = np.asarray(target_record["points"], dtype=np.float64)
        model = build_model(cfg_model, {}, cfg_sim.v_max)
        CheckpointManager(output_dir).load(model, "best")
        controller = GNNController(cfg_model, cfg_sim.v_max, communication=bool(saved["communication"]))
        controller.model = model
        controller.model.eval()
        result = simulate_episode(
            target, controller, cfg_sim, n_robots=int(profile["n_robots"]),
            episode_seed=int(profile["validation_seeds"][0]), shape_id=int(target_record["shape_id"]),
        )
        axis.scatter(target[:, 0], target[:, 1], s=6, c="lightgray", label="cible")
        axis.scatter(result.positions[-1][:, 0], result.positions[-1][:, 1], s=12, c="tab:blue", label="robots")
        axis.set_title(f"{variant} (graine {representative})")
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Etat final sur le cercle (validation) — meilleur checkpoint par variante")
    fig.tight_layout()
    path = FIGURES / "training_circle_final.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[synthese] figure : {path}")


def main() -> None:
    import argparse

    global INPUT, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    INPUT, FIGURES = args.input, args.figures
    run()


if __name__ == "__main__":
    main()
