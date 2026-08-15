from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..domain import Candidate, RequestContext, UserProfile
from ..training.calibration import calibrator_from_dict
from .ranking import FEATURE_NAMES, build_features


class LightGBMRanker:
    def __init__(self, manifest_path: str | Path) -> None:
        import lightgbm
        path = Path(manifest_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.model = lightgbm.Booster(model_file=str(path.parent / manifest["model_file"]))
        self.calibrator = calibrator_from_dict(manifest["calibrator"])

    def score(self, candidates: Iterable[Candidate], profile: UserProfile, context: RequestContext) -> None:
        materialized = list(candidates)
        for candidate in materialized:
            candidate.features = build_features(candidate, profile)
        rows = [[candidate.features[name] for name in FEATURE_NAMES] for candidate in materialized]
        predictions = self.model.predict(rows) if rows else []
        for candidate, prediction in zip(materialized, predictions, strict=True):
            candidate.rank_score = self.calibrator.predict(float(prediction))


class DeepRanker:
    def __init__(self, manifest_path: str | Path) -> None:
        from ..training.arm_c import _require_torch, create_deep_ranker
        self.torch, _ = _require_torch()
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        self.model = create_deep_ranker(**self.manifest["config"])
        self.model.load_state_dict(self.torch.load(path.parent / self.manifest["weights"], weights_only=True))
        self.model.eval()

    def score(self, candidates: Iterable[Candidate], profile: UserProfile, context: RequestContext) -> None:
        from ..training.arm_c import stable_bucket
        materialized = list(candidates)
        if not materialized:
            return
        for candidate in materialized:
            candidate.features = build_features(candidate, profile)
        config = self.manifest["config"]
        numeric = self.torch.tensor(
            [[c.features[n] for n in FEATURE_NAMES] for c in materialized],
            dtype=self.torch.float32,
        )
        users = self.torch.tensor(
            [stable_bucket(profile.user_id, config["user_buckets"])] * len(materialized),
            dtype=self.torch.long,
        )
        works = self.torch.tensor(
            [stable_bucket(c.item.work_id, config["work_buckets"]) for c in materialized],
            dtype=self.torch.long,
        )
        with self.torch.inference_mode():
            scores = self.torch.sigmoid(self.model(numeric, users, works)).tolist()
        for candidate, score in zip(materialized, scores, strict=True):
            candidate.rank_score = float(score)
