"""Tests minuscules et synthétiques de phase 12, sans boucle d'optimisation."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import yaml

from src.common.config import ConfigLoader
from src.common.mission import Mission
from src.common.seeds import SeedManager
from src.controllers.base import Controller
from src.controllers.flocking import FlockingController
from src.controllers.gnn import GNNController
from src.controllers.manual_shape import ManualShapeController
from src.evaluation.robustness import PerturbationLevel, RobustnessStudy, condition_identifier, reformation_step
from src.evaluation.scenario import Scenario
from src.metrics import MetricSuite
from src.models.gnn import SwarmGNN
from src.shapes.index import ShapeIndex
from src.simulator.episode import Episode
from src.simulator.failures import FailureModel
from src.simulator.graph import CommunicationGraph
from src.simulator.perturbations import (
    CommunicationNoise, CompositePerturbation, FaultyRobot, IdentityPerturbation,
    Perturbation, PositionNoise, RangeReduction,
)
from src.simulator.state import SwarmState
from src.training.checkpoint import CheckpointManager


class RecordingController(Controller):
    def __init__(self, velocity=(0.0, 0.0)):
        self.velocity = np.asarray(velocity, dtype=float)
        self.observations = []
        self.graphs = []

    def command(self, state, adjacency, mission):
        self.observations.append(state.positions.copy())
        self.graphs.append(adjacency.copy())
        return np.tile(self.velocity, (state.n_robots, 1))


@pytest.fixture
def cfg_sim():
    return replace(ConfigLoader().load_simulator(), horizon=6, failure_fraction=0.0)


@pytest.fixture
def mission():
    return Mission(0, np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))


@pytest.fixture
def tiny_config(tmp_path):
    cfg = ConfigLoader().load_raw("robustness")
    return {**cfg, "output": str(tmp_path / "study"), "training_root": str(tmp_path / "missing"),
            "controllers": {"manual": {"type": "baseline"}}, "training_seeds": [11, 12],
            "swarm_sizes": [4], "failure_fractions": [0.0], "shapes": ["geometries_circle@4"],
            "repetitions": 2, "evaluation_horizon": 4, "stability_steps": 2,
            "perturbations": {"position_noise": {"strengths": [0.0, 0.1]},
                              "communication_noise": {"strengths": [0.0, 1.0]},
                              "range_reduction": {"strengths": [0.0, 1.0]},
                              "faulty_robot": {"strengths": [0.0, 1.0]}}}


@pytest.fixture
def tiny_index(mission):
    return ShapeIndex([], {"geometries_circle@4": mission.target}, ["geometries_circle"])


@pytest.mark.parametrize("perturbation", [PositionNoise(0.2), CommunicationNoise(0.5), RangeReduction(0.5),
                                         FaultyRobot(0.5), CompositePerturbation((PositionNoise(0.1), CommunicationNoise(0.3), FaultyRobot(0.2)))])
def test_same_seed_same_perturbed_episode(cfg_sim, mission, perturbation):
    episode = Episode(cfg_sim, perturbation)
    a = episode.execute(mission, RecordingController((0.1, 0.2)), 6, 42)
    b = episode.execute(mission, RecordingController((0.1, 0.2)), 6, 42)
    for key in ("positions", "active", "commands", "observed_positions", "adjacency"):
        assert np.array_equal(getattr(a, key), getattr(b, key))


@pytest.mark.parametrize("perturbation", [None, IdentityPerturbation(), PositionNoise(0.0), CommunicationNoise(0.0), RangeReduction(1.0), FaultyRobot(0.0)])
def test_zero_strength_matches_original_algorithm_exactly(cfg_sim, mission, perturbation):
    cfg = replace(cfg_sim, failure_fraction=0.5)
    controller = ManualShapeController(0.7, 0.2, 0.4)
    episode = Episode(cfg)
    seeds = SeedManager(42)
    positions = episode.initial_positions(6, seeds)
    active = np.ones(6, dtype=bool)
    failure = episode.failures.plan(6, cfg.horizon, seeds.rng("failures"))
    history, masks, commands = [], [], []
    # Algorithme ordinaire avant injection, reproduit sans utiliser execute.
    for step in range(cfg.horizon):
        if failure is not None and step == failure.step:
            active[list(failure.indices)] = False
        history.append(positions.copy())
        masks.append(active.copy())
        state = SwarmState(positions, active, step)
        command = controller.command(state, episode.graph.adjacency(state), mission)
        bounded = episode.engine.bound_commands(command, active)
        positions = episode.engine.step(positions, bounded, active)
        commands.append(bounded)
    history.append(positions)
    masks.append(active.copy())
    result = Episode(cfg, perturbation).execute(mission, controller, 6, 42)
    assert np.array_equal(result.positions, np.array(history))
    assert np.array_equal(result.active, np.array(masks))
    assert np.array_equal(result.commands, np.array(commands))
    assert np.array_equal(result.observed_positions, result.positions[:-1])


def test_default_identity_does_not_create_perturbation_rng(cfg_sim, mission, monkeypatch):
    calls = []
    original = SeedManager.rng

    def tracked(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(SeedManager, "rng", tracked)
    Episode(cfg_sim).execute(mission, RecordingController(), 4, 42)
    assert calls == ["initial_positions", "failures"]


def test_zero_hooks_consume_no_rng():
    class NoDraws:
        def __getattr__(self, key):
            pytest.fail(f"tirage inattendu : {key}")

    state = SwarmState(np.zeros((2, 2)), np.ones(2, dtype=bool), 0)
    graph = np.zeros((2, 2), dtype=bool)
    assert PositionNoise(0).observe(state, NoDraws()) is state
    assert CommunicationNoise(0).perturb_graph(graph, state, NoDraws()) is graph
    assert FaultyRobot(0).plan_failures(2, 4, NoDraws()) is None


def test_noise_changes_observation_not_physics_or_graph(cfg_sim, mission):
    noisy_controller = RecordingController()
    truth = Episode(cfg_sim).execute(mission, RecordingController(), 6, 17)
    noisy = Episode(cfg_sim, PositionNoise(0.8)).execute(mission, noisy_controller, 6, 17)
    assert np.array_equal(truth.positions, noisy.positions)
    assert np.array_equal(truth.adjacency, noisy.adjacency)
    assert not np.array_equal(noisy.observed_positions, noisy.positions[:-1])
    assert np.array_equal(noisy.observed_positions, np.array(noisy_controller.observations))


def test_noise_recomputes_target_direction_from_observed_positions(cfg_sim, mission, monkeypatch):
    loader = ConfigLoader()
    cfg_model = replace(loader.load_model(), hidden_dim=4, embedding_dim=2, message_rounds=1, use_target_geometry=True)
    controller = GNNController(cfg_model, cfg_sim.v_max, seed=11)
    recorded = []
    original = controller.target_vector

    def tracked(positions, target_mission):
        direction = original(positions, target_mission)
        recorded.append((positions.copy(), direction.copy()))
        return direction

    monkeypatch.setattr(controller, "target_vector", tracked)
    result = Episode(replace(cfg_sim, horizon=1), PositionNoise(0.3)).execute(mission, controller, 4, 42)
    observed, direction = recorded[0]
    assert np.array_equal(observed, result.observed_positions[0])
    assert np.array_equal(direction, original(observed, mission))
    assert not np.allclose(direction, original(result.positions[0], mission))


def test_remote_positions_do_not_leak_through_noise(cfg_sim, mission, monkeypatch):
    cfg = replace(cfg_sim, horizon=1, r_c=0.2)
    controller = FlockingController(0.5, 0.2, 0.4)
    fixed = np.array([[0.0, 0.0], [0.1, 0.0], [1.8, 1.8]])
    monkeypatch.setattr(Episode, "initial_positions", lambda self, count, seeds: fixed.copy())
    a = Episode(cfg, PositionNoise(0.5)).execute(mission, controller, 3, 42)
    fixed[2] = [-1.8, -1.8]
    b = Episode(cfg, PositionNoise(0.5)).execute(mission, controller, 3, 42)
    assert not a.adjacency[0, 0, 2] and not b.adjacency[0, 0, 2]
    assert np.array_equal(a.commands[0, :2], b.commands[0, :2])


def test_shortened_graph_and_inactive_masks(cfg_sim, mission, monkeypatch):
    fixed = np.array([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.7, 0.0]])
    monkeypatch.setattr(Episode, "initial_positions", lambda self, count, seeds: fixed.copy())
    cfg = replace(cfg_sim, r_c=1.0)
    result = Episode(cfg, CompositePerturbation((RangeReduction(0.4), FaultyRobot(0.5)))).execute(mission, RecordingController(), 4, 42)
    assert result.communication_radius == 0.4
    assert result.adjacency[0, 0, 1] and not result.adjacency[0, 0, 2]
    for graph, active in zip(result.adjacency, result.active):
        assert np.array_equal(graph, graph.T)
        assert not graph.diagonal().any()
        assert not graph[~active].any() and not graph[:, ~active].any()
        distances = np.linalg.norm(fixed[:, None] - fixed[None, :], axis=2)
        assert not graph[distances > 0.4].any()


def test_edge_dropout_is_symmetric_subset_and_can_remove_all(cfg_sim, mission):
    cfg = replace(cfg_sim, r_c=10.0, failure_fraction=0.5)
    normal = Episode(cfg).execute(mission, RecordingController(), 6, 42)
    dropped = Episode(cfg, CommunicationNoise(0.5)).execute(mission, RecordingController(), 6, 42)
    empty = Episode(cfg, CommunicationNoise(1.0)).execute(mission, RecordingController(), 6, 42)
    assert not empty.adjacency.any()
    assert np.array_equal(dropped.adjacency, dropped.adjacency.transpose(0, 2, 1))
    assert not (dropped.adjacency & ~normal.adjacency).any()
    assert dropped.adjacency.sum() < normal.adjacency.sum()


def test_faulty_all_robots_freeze_and_exclude_metrics(cfg_sim, mission):
    result = Episode(cfg_sim, FaultyRobot(1.0)).execute(mission, RecordingController((0.3, 0.0)), 6, 42)
    step = cfg_sim.horizon // 2
    assert result.failure_step == step and result.failure_indices == tuple(range(6))
    assert not result.active[step:].any()
    assert not result.adjacency[step:].any()
    assert not result.commands[step:].any()
    assert np.array_equal(result.positions[step:], np.repeat(result.positions[step:step + 1], len(result.positions) - step, axis=0))
    loader = ConfigLoader()
    values = MetricSuite(loader.load_metrics(), cfg_sim, 1.0).compute(result.positions, result.active, mission.target).values
    assert not values["success"] and values["final_components"] == 0
    assert np.isnan(values["final_distance"])


def test_faulty_and_ordinary_streams_are_independent(cfg_sim, mission):
    cfg = replace(cfg_sim, failure_fraction=0.5)
    result = Episode(cfg, FaultyRobot(0.5)).execute(mission, RecordingController(), 6, 42)
    seeds = SeedManager(42)
    ordinary = FailureModel(0.5).plan(6, cfg.horizon, seeds.rng("failures"))
    faulty = FaultyRobot(0.5).plan_failures(6, cfg.horizon, seeds.rng("perturbations/faulty_robot/failures"))
    assert result.failure_indices == tuple(sorted(set(ordinary.indices) | set(faulty.indices)))
    noisy = Episode(cfg, CompositePerturbation((PositionNoise(0.7), FaultyRobot(0.5)))).execute(mission, RecordingController(), 6, 42)
    assert noisy.failure_indices == result.failure_indices
    assert np.array_equal(noisy.positions[0], result.positions[0])


def test_true_initial_coordinates_identical_across_conditions(cfg_sim, mission):
    start = Episode(cfg_sim).execute(mission, RecordingController(), 6, 42).positions[0]
    for level in (PositionNoise(0.4), CommunicationNoise(0.8), RangeReduction(0.0), FaultyRobot(1.0)):
        for failure_fraction in (0.0, 0.5):
            result = Episode(replace(cfg_sim, failure_fraction=failure_fraction), level).execute(mission, RecordingController((0.1, 0.0)), 6, 42)
            assert np.array_equal(start, result.positions[0])


def test_injection_clamps_illegal_edges_and_bounds_commands(cfg_sim, mission):
    class BadEdgesAndLargeCommands(Perturbation):
        stream_key = "test_commands"

        def perturb_graph(self, adjacency, state, rng):
            return np.ones_like(adjacency)

        def perturb_commands(self, commands, state, rng):
            return np.full_like(commands, 1e6)

    cfg = replace(cfg_sim, failure_fraction=0.5)
    result = Episode(cfg, BadEdgesAndLargeCommands()).execute(mission, RecordingController(), 6, 42)
    for graph, positions, active in zip(result.adjacency, result.positions, result.active):
        assert np.array_equal(graph, CommunicationGraph(cfg.r_c).adjacency(SwarmState(positions, active, 0)))
    assert (np.linalg.norm(result.commands, axis=2) <= cfg.v_max + 1e-12).all()
    assert not result.commands[~result.active[:-1]].any()
    assert (np.abs(result.positions) <= cfg.arena_size / 2).all()


@pytest.mark.parametrize("curve,failure,expected", [
    ([0.1, 0.1, 0.3, 0.2, 0.1, 0.1], 2, 2),
    ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 2, 1),
    ([0.1, 0.1, 0.3, 0.1, 0.3, 0.1], 2, None),
    ([0.3, 0.3, 0.3, 0.1, 0.1, 0.1], 2, None),
    ([0.1, 0.3, 0.3, 0.1, 0.1, 0.1], 2, None),
    ([0.1, 0.1, 0.3, 0.1, 0.1], None, None),
    ([0.1, 0.1, 0.3, 0.2, 0.2], 2, None),
    ([0.1, 0.1, 0.3, 0.1, np.nan, 0.1], 2, None),
    ([0.1, 0.1, 0.3, 0.1], 2, None),
    ([0.1, 0.3, 0.1, 0.1], 1, None),
])
def test_reformation_requires_prior_and_post_stability(curve, failure, expected):
    assert reformation_step(np.array(curve), failure, 0.15, 2) == expected


def test_composition_does_not_reuse_streams():
    with pytest.raises(ValueError, match="dupliqués"):
        CompositePerturbation((PositionNoise(0.1), PositionNoise(0.2)))


@pytest.mark.parametrize("constructor,value", [(PositionNoise, -0.1), (PositionNoise, np.nan),
                                               (CommunicationNoise, 1.1), (RangeReduction, -0.1),
                                               (FaultyRobot, 1.1)])
def test_invalid_strengths_rejected(constructor, value):
    with pytest.raises(ValueError):
        constructor(value)


def test_output_names_do_not_round_conditions_or_strengths():
    a = condition_identifier(Scenario(4, 0.101, "shape@4", 42))
    b = condition_identifier(Scenario(4, 0.102, "shape@4", 42))
    assert a != b
    assert PerturbationLevel("position_noise", 0.101, 1).identifier != PerturbationLevel("position_noise", 0.102, 1).identifier


@pytest.mark.parametrize("change", [
    {"success_threshold": 0.2}, {"stability_steps": 0}, {"unknown_setting": True},
    {"perturbations": {"communication_noise": {"strengths": [0.0, 0.1], "vector_std": 0.1}}},
    {"perturbations": {"position_noise": {"strengths": [0.0, 0.1, 0.1]}}},
    {"perturbations": {"unknown_noise": {"strengths": [0.0]}}},
])
def test_invalid_protocol_rejected_before_output(tiny_config, tiny_index, change, tmp_path):
    with pytest.raises(ValueError):
        RobustnessStudy({**tiny_config, **change}, index=tiny_index)
    assert not (tmp_path / "study").exists()


def test_facade_accepts_optional_perturbation(cfg_sim, mission):
    from src.simulator.episode import simulate_episode

    a = Episode(cfg_sim, RangeReduction(0.0)).execute(mission, RecordingController(), 4, 42)
    b = simulate_episode(mission.target, RecordingController(), cfg_sim, 4, 42,
                         shape_id=mission.shape_id, perturbation=RangeReduction(0.0))
    assert np.array_equal(a.positions, b.positions)
    assert not b.adjacency.any()


def test_missing_checkpoint_preflight_leaves_no_output(tiny_config, tiny_index):
    tiny_config["controllers"] = {"gnn": {"type": "trained", "variant": "gnn"}}
    study = RobustnessStudy(tiny_config, index=tiny_index)
    with pytest.raises(FileNotFoundError, match="run incomplet"):
        study.run()
    assert not study.output.exists()


def test_tiny_study_is_reproducible_and_archives_truth(tiny_config, tiny_index, tmp_path):
    a = RobustnessStudy(tiny_config, index=tiny_index)
    summary = a.run()
    b = RobustnessStudy(tiny_config, output=tmp_path / "second", index=tiny_index)
    b.run()
    assert summary["rows"] == summary["expected_rows"] == 16
    assert (a.output / "episodes.csv").read_bytes() == (b.output / "episodes.csv").read_bytes()
    for filename in ("effective_config.yaml", "seeds.json", "version.txt", "summary.json", "within_seed.csv", "between_seeds.csv", "breakpoints.csv"):
        assert (a.output / filename).is_file()
    assert len(list((a.output / "figures").glob("*.png"))) == 4
    with (a.output / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    starts = {}
    for row in rows:
        with np.load(a.output / row["archive"], allow_pickle=False) as raw, np.load(b.output / row["archive"], allow_pickle=False) as repeated:
            assert raw["positions"].dtype == np.float64
            for key in raw.files:
                assert np.array_equal(raw[key], repeated[key], equal_nan=True)
            assert {"active", "target", "target_radius", "communication_radius", "failure_step", "failure_indices", "observed_positions", "adjacency", "distance_curve"} <= set(raw.files)
            episode_seed = int(row["episode_seed"])
            if episode_seed in starts:
                assert np.array_equal(starts[episode_seed], raw["positions"][0])
            starts[episode_seed] = raw["positions"][0].copy()
            assert raw["positions"].shape[0] == tiny_config["evaluation_horizon"] + 1
            if row["perturbation"] == "communication_noise" and float(row["strength"]) == 1:
                assert not raw["adjacency"].any()
                assert (raw["components_by_step"] == 4).all()
            if row["perturbation"] == "faulty_robot" and float(row["strength"]) == 1:
                assert raw["failure_step"] == 2
                assert not raw["active"][2:].any()
    saved = json.loads((a.output / "summary.json").read_text())
    assert saved["success_threshold"] == 0.15
    assert "NaN" not in (a.output / "summary.json").read_text()
    with pytest.raises(FileExistsError):
        a.run()


def test_reports_separate_episode_and_seed_variability(tiny_config, tiny_index, monkeypatch):
    study = RobustnessStudy(tiny_config, index=tiny_index)
    monkeypatch.setattr(study, "_plot", lambda between, within: None)
    rows = []
    for strength, successes in ((0.0, [True, True]), (0.1, [True, False]), (0.2, [False, False])):
        for seed, distances in ((11, [0.1, 0.3]), (12, [0.5, 0.7])):
            for episode_seed, (distance, success) in enumerate(zip(distances, successes)):
                rows.append({"perturbation": "position_noise", "strength": strength, "controller": "gnn",
                             "training_seed": seed, "episode_seed": episode_seed, "n_robots": 4,
                             "failure_fraction": 0.0, "shape": "circle", "final_distance": distance,
                             "success": success, "collision_rate": 0.0, "normalized_regularity": 0.0,
                             "final_components": 1, "mean_components": 1.0, "final_active_fraction": 1.0,
                             "reformation_step": None})
    # Une autre condition ne doit pas contaminer les agrégats ni les ruptures.
    rows += [{**row, "n_robots": 8, "success": True} for row in rows.copy()]
    summary = study.build_report(rows)
    within = next(row for row in summary["within_seed"] if row["n_robots"] == 4 and row["strength"] == 0 and row["training_seed"] == 11 and row["metric"] == "final_distance")
    between = next(row for row in summary["between_seeds"] if row["n_robots"] == 4 and row["strength"] == 0 and row["metric"] == "final_distance")
    assert within["n"] == within["episode_count"] == 2
    assert within["std"] == pytest.approx(np.std([0.1, 0.3], ddof=1))
    assert between["n"] == 2 and between["mean"] == pytest.approx(0.4)
    assert between["std"] == pytest.approx(np.std([0.2, 0.6], ddof=1))
    breakpoint = next(row for row in summary["breakpoints"] if row["n_robots"] == 4)
    assert breakpoint["first_tested_strength_below_half"] == 0.2
    assert next(row for row in summary["breakpoints"] if row["n_robots"] == 8)["first_tested_strength_below_half"] is None


def test_loaded_checkpoints_are_frozen_and_hashed_without_optimizer(tiny_config, tiny_index, tmp_path):
    loader = ConfigLoader()
    cfg_model = replace(loader.load_model(), hidden_dim=4, embedding_dim=2, message_rounds=1)
    root = tmp_path / "checkpoints"
    config = {**tiny_config, "controllers": {"gnn": {"type": "trained", "variant": "tiny", "training_root": str(root)}},
              "training_seeds": [11], "repetitions": 1, "perturbations": {"position_noise": {"strengths": [0.0]}}}
    directory = root / "tiny" / "seed_11"
    directory.mkdir(parents=True)
    snapshot = {"model": asdict(cfg_model), "simulator": asdict(loader.load_simulator()),
                "shapes": asdict(loader.load_shapes()), "communication": True}
    (directory / "effective_config.yaml").write_text(yaml.safe_dump(snapshot), encoding="utf-8")
    (directory / "summary.json").write_text("{}", encoding="utf-8")
    (directory / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    (directory / "targets.json").write_text("{}", encoding="utf-8")
    model = SwarmGNN(cfg_model, loader.load_simulator().v_max)
    CheckpointManager(directory).save(model, "best", {"seed": 11, "communication": True})
    study = RobustnessStudy(config, index=tiny_index)
    factory = study._factories()["gnn"]
    loaded = factory.build("gnn", 11)
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    before = (directory / "best.pt").read_bytes()
    study.run()
    assert (directory / "best.pt").read_bytes() == before
    metadata = yaml.safe_load((study.output / "effective_config.yaml").read_text())
    assert metadata["training_runs"]["gnn"][11]["checkpoint_sha256"] == hashlib.sha256(before).hexdigest()
    assert metadata["simulator"]["horizon"] == config["evaluation_horizon"]
    seeds = json.loads((study.output / "seeds.json").read_text())
    assert "perturbations/position_noise/observe" in seeds["stream_keys"]
