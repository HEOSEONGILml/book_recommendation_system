from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FilteredCandidate:
    work_id: str
    reason: str

