from __future__ import annotations

import math
from collections.abc import Callable

from .dataset import TrainingExample


def binary_metrics(
    examples: list[TrainingExample], predict: Callable[[dict[str, float]], float]
) -> dict[str, float]:
    probabilities = [min(1 - 1e-8, max(1e-8, predict(example.features))) for example in examples]
    labels = [example.consumed for example in examples]
    logloss = -sum(
        label * math.log(probability) + (1 - label) * math.log(1 - probability)
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)
    brier = sum(
        (probability - label) ** 2
        for label, probability in zip(labels, probabilities, strict=True)
    ) / len(labels)
    return {"count": float(len(labels)), "logloss": logloss, "brier": brier}


def recall_at_k(recommended: list[str], consumed: set[str], k: int) -> float:
    return 0.0 if not consumed else len(set(recommended[:k]) & consumed) / len(consumed)


def catalog_coverage(recommendations: list[list[str]], catalog_size: int) -> float:
    exposed = {work_id for row in recommendations for work_id in row}
    return 0.0 if catalog_size <= 0 else len(exposed) / catalog_size
