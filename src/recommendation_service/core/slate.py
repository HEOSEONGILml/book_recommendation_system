from __future__ import annotations

import hashlib
import random

from ..domain import Candidate, CandidateSource, Item, Policy, RequestContext


class SlateComposer:
    """Temporal xQuAD 결과에 ε-greedy 탐색 슬롯을 결합한다."""

    def compose(
        self,
        ranked: list[Candidate],
        exploration_universe: list[Item],
        context: RequestContext,
        policy: Policy,
    ) -> list[Candidate]:
        ordered = sorted(ranked, key=lambda candidate: candidate.final_score, reverse=True)
        seed = int.from_bytes(
            hashlib.sha256(f"{context.request_id}:{policy.policy_version}".encode()).digest()[:8],
            "big",
        )
        rng = random.Random(seed)
        explore = context.limit > 1 and rng.random() < policy.epsilon
        selected = ordered[: context.limit - int(explore)]

        if explore:
            selected_ids = {candidate.item.work_id for candidate in selected}
            feasible = [
                item
                for item in exploration_universe
                if item.work_id not in context.non_personalized_work_ids
                and item.work_id not in selected_ids
            ]
            if feasible:
                item = rng.choice(feasible)
                selected.append(
                    Candidate(
                        item=item,
                        sources={CandidateSource.EXPLORATION},
                        is_exploration=True,
                        propensity=policy.epsilon / len(feasible),
                    )
                )
            else:
                selected = ordered[: context.limit]
        return selected
