from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.ranking import FEATURE_NAMES
from .calibration import Calibrator, select_calibrator
from .dataset import TrainingExample
from .evaluation import binary_metrics


def _require_lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as exc:
        raise RuntimeError("Arm B requires: uv sync --extra ml") from exc
    return lightgbm


def train_arm_b(
    train: list[TrainingExample], validation: list[TrainingExample]
) -> tuple[Any, Calibrator, dict[str, float]]:
    lightgbm = _require_lightgbm()
    model = lightgbm.LGBMClassifier(
        objective="binary", n_estimators=300, learning_rate=0.04,
        num_leaves=31, min_child_samples=50, random_state=42, verbosity=-1,
    )
    rows = [[example.features[name] for name in FEATURE_NAMES] for example in train]
    model.fit(rows, [example.consumed for example in train], feature_name=list(FEATURE_NAMES))
    raw = [float(model.predict_proba([[example.features[n] for n in FEATURE_NAMES]])[0][1]) for example in validation]
    calibrator = select_calibrator(raw, [example.consumed for example in validation])
    metrics = binary_metrics(
        validation,
        lambda features: calibrator.predict(float(model.predict_proba([[features[n] for n in FEATURE_NAMES]])[0][1])),
    )
    return model, calibrator, metrics


def write_arm_b_artifact(
    directory: str | Path, version: str, model: Any, calibrator: Calibrator, metrics: dict[str, float]
) -> Path:
    target = Path(directory) / version
    target.mkdir(parents=True, exist_ok=False)
    model_path = target / "ranker.txt"
    model.booster_.save_model(str(model_path))
    manifest = {
        "version": version, "arm": "B", "created_at": datetime.now(UTC).isoformat(),
        "feature_names": FEATURE_NAMES, "model_file": model_path.name,
        "calibrator": calibrator.to_dict(), "metrics": metrics,
    }
    manifest_path = target / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (target / "checksums.txt").write_text(f"{checksum}  {model_path.name}\n", encoding="utf-8")
    return manifest_path
