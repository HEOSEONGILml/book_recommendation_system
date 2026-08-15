from __future__ import annotations

from ..domain import Candidate, Item, RequestContext, UserProfile


class EligibilityFilters:
    def catalog(self, items: list[Item], profile: UserProfile, context: RequestContext) -> list[Item]:
        return [
            item
            for item in items
            if item.rights_start <= context.now < item.rights_end
            and (item.age_rating == 0 or (profile.age is not None and profile.age >= item.age_rating))
            and item.work_id not in profile.consumed_work_ids
            and item.work_id not in profile.disliked_work_ids
        ]

    @staticmethod
    def remove_non_personalized_duplicates(
        candidates: list[Candidate], context: RequestContext
    ) -> tuple[list[Candidate], int]:
        kept = [
            candidate
            for candidate in candidates
            if candidate.item.work_id not in context.non_personalized_work_ids
        ]
        return kept, len(candidates) - len(kept)
