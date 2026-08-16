from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from math import sqrt

from ..config import Settings
from ..domain import Format, Item, Policy, RankerArm, UserProfile


class InMemoryCatalogRepository:
    def __init__(self, items: list[Item]) -> None:
        self._items = items
        self._by_work = {item.work_id: item for item in items}

    def list_items(self) -> list[Item]:
        return list(self._items)

    def personalized_candidates(self, profile: UserProfile, limit: int) -> list[Item]:
        return sorted(
            self._items,
            key=lambda item: profile.genre_affinity.get(item.genre, 0.0) + item.popularity,
            reverse=True,
        )[:limit]

    def similar_candidates(self, anchor_work_ids: set[str], limit: int) -> list[Item]:
        anchors = [self._by_work[work_id] for work_id in anchor_work_ids if work_id in self._by_work]
        return sorted(
            (item for item in self._items if item.work_id not in anchor_work_ids),
            key=lambda item: max(
                (self._cosine(item.embedding, anchor.embedding) for anchor in anchors),
                default=0.0,
            ),
            reverse=True,
        )[:limit]

    @staticmethod
    def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        denominator = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
        return numerator / denominator if denominator else 0.0

    def exploration_candidates(self, limit: int) -> list[Item]:
        return sorted(self._items, key=lambda item: item.popularity)[:limit]

    def get_item(self, work_id: str) -> Item | None:
        return self._by_work.get(work_id)


class InMemoryUserRepository:
    def __init__(self, profiles: list[UserProfile] | None = None) -> None:
        self._profiles = {profile.user_id: profile for profile in profiles or []}

    def get_profile(self, user_id: str) -> UserProfile:
        profile = self._profiles.setdefault(
            user_id,
            UserProfile(user_id=user_id, library_work_ids={"w_005", "w_011"}),
        )
        return replace(
            profile,
            library_work_ids=set(profile.library_work_ids),
            consumed_work_ids=set(profile.consumed_work_ids),
            disliked_work_ids=set(profile.disliked_work_ids),
            genre_affinity=dict(profile.genre_affinity),
            exposure_history={key: list(value) for key, value in profile.exposure_history.items()},
        )


class StaticPolicyProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def resolve(self, user_id: str) -> Policy:
        return Policy(
            experiment_id="local-readme-v2",
            arm=RankerArm(self.settings.ranker_arm.upper()),
            xquad_lambda=self.settings.xquad_lambda,
            epsilon=self.settings.epsilon,
            model_version=self.settings.model_version,
            feature_version=self.settings.feature_version,
            policy_version=self.settings.policy_version,
        )


def sample_catalog() -> list[Item]:
    now = datetime.now(UTC)
    genres = ["소설", "역사", "과학", "경제", "에세이", "판타지"]
    formats = [Format.EBOOK, Format.AUDIOBOOK, Format.CHATBOOK]
    return [
        Item(
            work_id=f"w_{index:03d}",
            format_id=formats[index % len(formats)],
            title=f"샘플 도서 {index}",
            author_id=f"author_{index % 17:02d}",
            genre=genres[index % len(genres)],
            popularity=max(0.01, 1.0 - index / 75),
            embedding=(index / 60, (index % 7) / 7, (index % 11) / 11),
            rights_end=now + timedelta(days=365),
            age_rating=19 if index % 29 == 0 else 0,
        )
        for index in range(1, 61)
    ]
