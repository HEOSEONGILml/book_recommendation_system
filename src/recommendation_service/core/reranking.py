from __future__ import annotations

import math
from datetime import datetime

from ..domain import Candidate, Policy, UserProfile


class TemporalXQuAD:
    """관련성 점수에 시간 감쇠된 노출 공정성 점수를 결합한다."""

    def apply(
        self, candidates: list[Candidate], profile: UserProfile, policy: Policy, now: datetime
    ) -> None:
        for candidate in candidates:
            repeated = sum(
                math.exp(-0.05 * max(0.0, (now - timestamp).total_seconds() / 86400))
                for timestamp in profile.exposure_history.get(candidate.item.work_id, [])
            )
            candidate.exposure_fairness = math.exp(-repeated)
            candidate.final_score = (
                (1.0 - policy.xquad_lambda) * candidate.rank_score
                + policy.xquad_lambda * candidate.exposure_fairness
            )
