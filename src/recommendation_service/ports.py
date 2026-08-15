from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from .domain import Candidate, Item, Policy, RequestContext, UserProfile


class CatalogRepository(Protocol):
    def personalized_candidates(self, profile: UserProfile, limit: int) -> list[Item]: ...
    def cold_start_candidates(self, genres: set[str], limit: int) -> list[Item]: ...
    def exploration_candidates(self, limit: int) -> list[Item]: ...
    def get_item(self, work_id: str) -> Item | None: ...


class UserRepository(Protocol):
    def get_profile(self, user_id: str) -> UserProfile: ...


class PolicyProvider(Protocol):
    def resolve(self, user_id: str) -> Policy: ...


class Ranker(Protocol):
    def score(
        self, candidates: Iterable[Candidate], profile: UserProfile, context: RequestContext
    ) -> None: ...


