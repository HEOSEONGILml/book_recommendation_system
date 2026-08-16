from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    model_version: str = "ranker-a0-local"
    feature_version: str = "feature-v2"
    policy_version: str = "policy-v2"
    ranker_arm: str = "A0"
    model_manifest: str | None = None
    xquad_lambda: float = 0.2
    epsilon: float = 0.05
    request_timeout_ms: int = 250
    similarity_threshold: float = 0.55

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_version=os.getenv("MILLIE_MODEL_VERSION", "ranker-a0-local"),
            feature_version=os.getenv("MILLIE_FEATURE_VERSION", "feature-v2"),
            policy_version=os.getenv("MILLIE_POLICY_VERSION", "policy-v2"),
            ranker_arm=os.getenv("MILLIE_RANKER_ARM", "A0"),
            model_manifest=os.getenv("MILLIE_MODEL_MANIFEST"),
            xquad_lambda=float(os.getenv("MILLIE_XQUAD_LAMBDA", "0.2")),
            epsilon=float(os.getenv("MILLIE_EPSILON", "0.05")),
            request_timeout_ms=int(os.getenv("MILLIE_REQUEST_TIMEOUT_MS", "250")),
            similarity_threshold=float(os.getenv("MILLIE_SIMILARITY_THRESHOLD", "0.55")),
        )
