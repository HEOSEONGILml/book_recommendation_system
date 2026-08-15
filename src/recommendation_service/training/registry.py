from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .baseline import LinearRankModel


def write_arm_a_artifact(
    directory: str | Path, version: str, model: LinearRankModel, metrics: dict[str, float]
) -> Path:
    target = Path(directory) / version
    target.mkdir(parents=True, exist_ok=False)
    path = target / "model.json"
    path.write_text(json.dumps({
        "version": version, "arm": "A", "created_at": datetime.now(UTC).isoformat(),
        "weights": model.weights, "metrics": metrics,
    }, indent=2), encoding="utf-8")
    return path
