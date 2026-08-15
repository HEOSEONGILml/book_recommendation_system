from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol


def _clip(value: float) -> float:
    return min(1 - 1e-9, max(1e-9, value))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class Calibrator(Protocol):
    def predict(self, probability: float) -> float: ...
    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IdentityCalibrator:
    def predict(self, probability: float) -> float:
        return _clip(probability)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "identity"}


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    slope: float
    intercept: float

    def predict(self, probability: float) -> float:
        logit = math.log(_clip(probability) / (1 - _clip(probability)))
        return _sigmoid(self.slope * logit + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "platt", "slope": self.slope, "intercept": self.intercept}


@dataclass(frozen=True, slots=True)
class BetaCalibrator:
    positive_log_weight: float
    negative_log_weight: float
    intercept: float

    def predict(self, probability: float) -> float:
        probability = _clip(probability)
        value = (
            self.positive_log_weight * math.log(probability)
            + self.negative_log_weight * math.log(1 - probability)
            + self.intercept
        )
        return _sigmoid(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "beta",
            "positive_log_weight": self.positive_log_weight,
            "negative_log_weight": self.negative_log_weight,
            "intercept": self.intercept,
        }


@dataclass(frozen=True, slots=True)
class IsotonicCalibrator:
    upper_bounds: tuple[float, ...]
    values: tuple[float, ...]

    def predict(self, probability: float) -> float:
        for upper, value in zip(self.upper_bounds, self.values, strict=True):
            if probability <= upper:
                return _clip(value)
        return _clip(self.values[-1])

    def to_dict(self) -> dict[str, Any]:
        return {"type": "isotonic", "upper_bounds": self.upper_bounds, "values": self.values}


def fit_platt(probabilities: list[float], labels: list[float]) -> PlattCalibrator:
    features = [math.log(_clip(p) / (1 - _clip(p))) for p in probabilities]
    slope, intercept = 1.0, 0.0
    for _ in range(500):
        gradient_slope = gradient_intercept = 0.0
        for feature, label in zip(features, labels, strict=True):
            error = _sigmoid(slope * feature + intercept) - label
            gradient_slope += error * feature
            gradient_intercept += error
        scale = 0.05 / max(1, len(labels))
        slope -= scale * gradient_slope
        intercept -= scale * gradient_intercept
    return PlattCalibrator(slope, intercept)


def fit_beta(probabilities: list[float], labels: list[float]) -> BetaCalibrator:
    weights = [1.0, -1.0, 0.0]
    for _ in range(500):
        gradients = [0.0, 0.0, 0.0]
        for probability, label in zip(probabilities, labels, strict=True):
            probability = _clip(probability)
            features = (math.log(probability), math.log(1 - probability), 1.0)
            prediction = _sigmoid(sum(w * x for w, x in zip(weights, features, strict=True)))
            for index, feature in enumerate(features):
                gradients[index] += (prediction - label) * feature
        scale = 0.05 / max(1, len(labels))
        weights = [weight - scale * gradient for weight, gradient in zip(weights, gradients, strict=True)]
    return BetaCalibrator(*weights)


def fit_isotonic(probabilities: list[float], labels: list[float]) -> IsotonicCalibrator:
    blocks = [[p, p, y, 1] for p, y in sorted(zip(probabilities, labels, strict=True))]
    index = 0
    while index < len(blocks) - 1:
        if blocks[index][2] / blocks[index][3] > blocks[index + 1][2] / blocks[index + 1][3]:
            left, right = blocks[index], blocks[index + 1]
            blocks[index : index + 2] = [
                [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            ]
            index = max(0, index - 1)
        else:
            index += 1
    return IsotonicCalibrator(
        tuple(float(block[1]) for block in blocks),
        tuple(float(block[2] / block[3]) for block in blocks),
    )


def calibrator_from_dict(payload: dict[str, Any]) -> Calibrator:
    kind = payload["type"]
    if kind == "platt":
        return PlattCalibrator(payload["slope"], payload["intercept"])
    if kind == "beta":
        return BetaCalibrator(
            payload["positive_log_weight"],
            payload["negative_log_weight"],
            payload["intercept"],
        )
    if kind == "isotonic":
        return IsotonicCalibrator(tuple(payload["upper_bounds"]), tuple(payload["values"]))
    return IdentityCalibrator()


def select_calibrator(probabilities: list[float], labels: list[float]) -> Calibrator:
    if not probabilities:
        return IdentityCalibrator()
    candidates: list[Calibrator] = [
        IdentityCalibrator(),
        fit_platt(probabilities, labels),
        fit_beta(probabilities, labels),
        fit_isotonic(probabilities, labels),
    ]
    return min(
        candidates,
        key=lambda candidate: sum(
            (candidate.predict(probability) - label) ** 2
            for probability, label in zip(probabilities, labels, strict=True)
        ),
    )

