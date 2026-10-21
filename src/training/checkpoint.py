"""Sauvegarde des poids : meilleur état (choisi sur validation) et dernier état."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.output_dir / f"{name}.pt"

    def save(self, model, name: str, metadata: dict | None = None) -> Path:
        path = self._path(name)
        metadata = dict(metadata or {})
        if hasattr(model, "cfg"):
            # La configuration des poids prime sur une éventuelle surcharge périmée.
            metadata["model_config"] = asdict(model.cfg)
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
        return path

    def load(self, model, name: str) -> dict:
        payload = torch.load(self._path(name), map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"])
        return payload.get("metadata", {})
