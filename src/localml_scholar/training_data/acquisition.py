"""Local, manually curated paper-acquisition queue records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


def _optional_text(value: object, name: str, maximum: int) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text or None.")
    value = value.strip()
    if not value or len(value) > maximum:
        raise ValueError(f"{name} must contain 1 to {maximum} characters.")
    return value


@dataclass(frozen=True)
class PaperAcquisitionItem:
    """One non-fetching suggestion for expanding the local paper corpus."""

    title: str
    reason: str
    category: str
    doi: str | None = None
    arxiv_id: str | None = None
    citation: str | None = None
    status: str = "suggested"

    def __post_init__(self) -> None:
        for name, maximum in (("title", 500), ("reason", 2000), ("category", 100)):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be text.")
            value = value.strip()
            if not value or len(value) > maximum:
                raise ValueError(f"{name} must contain 1 to {maximum} characters.")
            object.__setattr__(self, name, value)
        for name, maximum in (("doi", 300), ("arxiv_id", 100), ("citation", 2000)):
            object.__setattr__(
                self, name, _optional_text(getattr(self, name), name, maximum)
            )
        if self.status not in {"suggested", "obtained", "declined"}:
            raise ValueError("status must be suggested, obtained, or declined.")

    @property
    def item_id(self) -> str:
        payload = "\n".join(
            (self.title.casefold(), self.doi or "", self.arxiv_id or "")
        )
        return "paper_queue_" + hashlib.sha256(payload.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "citation": self.citation,
            "reason": self.reason,
            "category": self.category,
            "status": self.status,
            "fetch_performed": False,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PaperAcquisitionItem:
        if not isinstance(value, dict):
            raise TypeError("Paper acquisition item must be an object.")
        return cls(
            title=value.get("title"),
            doi=value.get("doi"),
            arxiv_id=value.get("arxiv_id"),
            citation=value.get("citation"),
            reason=value.get("reason"),
            category=value.get("category"),
            status=value.get("status", "suggested"),
        )
