from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.ranking import FEATURE_NAMES
from .dataset import TrainingExample


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Arm C requires: uv sync --extra ml") from exc
    return torch, nn


def stable_bucket(value: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % bucket_count


def create_deep_ranker(
    numeric_dim: int, user_buckets: int, work_buckets: int, embedding_dim: int, hidden_dim: int
) -> Any:
    torch, nn = _require_torch()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.user_embedding = nn.Embedding(user_buckets, embedding_dim)
            self.work_embedding = nn.Embedding(work_buckets, embedding_dim)
            self.network = nn.Sequential(
                nn.Linear(numeric_dim + embedding_dim * 2, hidden_dim),
                nn.ReLU(), nn.Linear(hidden_dim, 1),
            )

        def forward(self, numeric: Any, user_ids: Any, work_ids: Any) -> Any:
            values = torch.cat((numeric, self.user_embedding(user_ids), self.work_embedding(work_ids)), dim=1)
            return self.network(values).squeeze(1)

    return Model()


DEFAULT_CONFIG = {
    "numeric_dim": len(FEATURE_NAMES), "user_buckets": 100_003,
    "work_buckets": 200_003, "embedding_dim": 32, "hidden_dim": 128,
}


def train_arm_c(
    train: list[TrainingExample], validation: list[TrainingExample], epochs: int = 10
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    torch, _ = _require_torch()
    config = dict(DEFAULT_CONFIG)
    model = create_deep_ranker(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    numeric = torch.tensor([[e.features[n] for n in FEATURE_NAMES] for e in train], dtype=torch.float32)
    users = torch.tensor([stable_bucket(e.user_id, config["user_buckets"]) for e in train])
    works = torch.tensor([stable_bucket(e.work_id, config["work_buckets"]) for e in train])
    labels = torch.tensor([e.consumed for e in train], dtype=torch.float32)
    for _ in range(epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(numeric, users, works), labels
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        v_numeric = torch.tensor([[e.features[n] for n in FEATURE_NAMES] for e in validation], dtype=torch.float32)
        v_users = torch.tensor([stable_bucket(e.user_id, config["user_buckets"]) for e in validation])
        v_works = torch.tensor([stable_bucket(e.work_id, config["work_buckets"]) for e in validation])
        probabilities = torch.sigmoid(model(v_numeric, v_users, v_works))
        targets = torch.tensor([e.consumed for e in validation])
        brier = float(((probabilities - targets) ** 2).mean())
    return model, config, {"count": float(len(validation)), "brier": brier}


def write_arm_c_artifact(
    directory: str | Path, version: str, model: Any, config: dict[str, Any], metrics: dict[str, float]
) -> Path:
    torch, _ = _require_torch()
    target = Path(directory) / version
    target.mkdir(parents=True, exist_ok=False)
    weights = target / "weights.pt"
    torch.save(model.state_dict(), weights)
    manifest = {
        "version": version, "arm": "C", "created_at": datetime.now(UTC).isoformat(),
        "config": config, "weights": weights.name, "metrics": metrics,
    }
    path = target / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
