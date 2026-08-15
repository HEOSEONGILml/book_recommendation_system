from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..core.ranking import FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class TrainingExample:
    occurred_at: datetime
    user_id: str
    work_id: str
    carousel_type: str
    features: dict[str, float]
    consumed: float


def load_csv(path: str | Path) -> list[TrainingExample]:
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        examples = [
            TrainingExample(
                occurred_at=datetime.fromisoformat(row["occurred_at"]),
                user_id=row["user_id"],
                work_id=row["work_id"],
                carousel_type=row["carousel_type"],
                features={name: float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES},
                consumed=float(row["consumed"]),
            )
            for row in csv.DictReader(source)
        ]
    return sorted(examples, key=lambda example: example.occurred_at)


def time_split(
    examples: list[TrainingExample], validation_ratio: float = 0.2
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    if len(examples) < 2:
        raise ValueError("at least two examples are required")
    split_at = max(1, min(len(examples) - 1, int(len(examples) * (1 - validation_ratio))))
    return examples[:split_at], examples[split_at:]
