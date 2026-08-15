from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.ranking import FEATURE_NAMES
from .dataset import TrainingExample


@dataclass(frozen=True, slots=True)
class LinearRankModel:
    weights: dict[str, float]

    def predict(self, features: dict[str, float]) -> float:
        raw = sum(self.weights[name] * features.get(name, 0.0) for name in FEATURE_NAMES)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, raw))))


def train_arm_a(
    examples: list[TrainingExample], epochs: int = 300, learning_rate: float = 0.05
) -> LinearRankModel:
    weights = {name: 0.0 for name in FEATURE_NAMES}
    for _ in range(epochs):
        gradients = {name: 0.0 for name in FEATURE_NAMES}
        model = LinearRankModel(weights)
        for example in examples:
            error = model.predict(example.features) - example.consumed
            for name in FEATURE_NAMES:
                gradients[name] += error * example.features.get(name, 0.0)
        for name in FEATURE_NAMES:
            weights[name] -= learning_rate * gradients[name] / len(examples)
    return LinearRankModel(weights)
