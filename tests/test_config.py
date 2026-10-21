"""Garde-fous de configuration : pas de défaut silencieux, pas de clé morte (leçons de l'audit)."""

from __future__ import annotations

import pytest

from src.common.config import ConfigError, ConfigLoader


@pytest.fixture()
def loader(tmp_path):
    return ConfigLoader(config_dir=tmp_path)


def _write(tmp_path, text: str):
    (tmp_path / "simulator.yaml").write_text(text, encoding="utf-8")


class TestConfigLoader:
    def test_missing_key_rejected(self, loader, tmp_path):
        _write(tmp_path, "arena_size: 4.0\ndt: 0.1\n")
        with pytest.raises(ConfigError, match="manquantes"):
            loader.load_simulator()

    def test_unknown_key_rejected(self, loader, tmp_path):
        import yaml

        complete = {
            "arena_size": 4.0, "dt": 0.1, "v_max": 1.0, "r_c": 1.5, "horizon": 60,
            "failure_fraction": 0.0, "record_positions": True, "spawn_radius": 1.8,
            "coverage_loss_weight": 0.5,
        }
        _write(tmp_path, yaml.safe_dump(complete))
        with pytest.raises(ConfigError, match="inconnues"):
            loader.load_simulator()

    def test_real_config_loads(self):
        cfg = ConfigLoader().load_simulator()
        assert cfg.arena_size > 0 and cfg.horizon > 0

    def test_incomplete_split_rejected(self, loader, tmp_path):
        import yaml

        bases = ConfigLoader().load_raw("shapes")
        bases["splits"]["geometries"]["train"] = ["circle"]
        (tmp_path / "shapes.yaml").write_text(yaml.safe_dump(bases), encoding="utf-8")
        with pytest.raises(ConfigError, match="incomplet"):
            loader.load_shapes()


class TestSeedManager:
    @pytest.mark.parametrize("key", ["initial_positions", "failures"])
    def test_streams_are_reproducible(self, key):
        import numpy as np

        from src.common.seeds import SeedManager

        expected = SeedManager(142).rng(key).random(16)
        actual = SeedManager(142).rng(key).random(16)
        assert np.array_equal(actual, expected)

    def test_streams_are_independent_of_call_order(self):
        import numpy as np

        from src.common.seeds import SeedManager

        manager = SeedManager(142)
        positions = manager.rng("initial_positions").random(16)
        failures = manager.rng("failures").random(16)
        assert np.array_equal(manager.rng("failures").random(16), failures)
        assert np.array_equal(manager.rng("initial_positions").random(16), positions)
        assert not np.array_equal(positions, failures)

    def test_root_seed_changes_draws(self):
        import numpy as np

        from src.common.seeds import SeedManager

        first = SeedManager(142).rng("initial_positions").random(16)
        second = SeedManager(143).rng("initial_positions").random(16)
        assert not np.array_equal(first, second)
