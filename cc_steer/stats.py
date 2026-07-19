"""Aggregate corpus statistics for the ``cc-steer stats`` command."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from cc_transcript.mining.store import Stats as IngestionStats

from cc_steer.report import project_label
from cc_steer.store import FeedbackStore, TriageStats
from cc_steer.triage import PROMPT_VERSION

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

GroupBy = Literal["project", "category"]


@dataclass(frozen=True, slots=True)
class StatsReport:
    """Aggregate corpus statistics rendered by the ``stats`` command.

    Attributes:
        ingestion: Ingestion counts by source kind and the scanned-file count.
        triage: Triage coverage and acceptance at ``prompt_version``.
        prompt_version: The judge prompt version the triage snapshot pins.
        by: The dimension the accepted corpus is grouped by, or ``None``.
        groups: Accepted-event counts keyed by group value in descending order,
            empty unless ``by`` is set.
    """

    ingestion: IngestionStats
    triage: TriageStats
    prompt_version: int
    by: GroupBy | None
    groups: Mapping[str, int]

    def render(self) -> str:
        """Renders the human-readable summary printed by the default command."""
        share = f" ({self.triage.accepted / self.triage.judged:.0%})" if self.triage.judged else ""
        return "\n".join(
            [
                f"total: {self.ingestion.total}  files: {self.ingestion.files}",
                *(f"  {kind}: {count}" for kind, count in self.ingestion.by_source.items()),
                f"triaged: {self.triage.judged}/{self.triage.total} (v{self.prompt_version})  "
                f"accepted: {self.triage.accepted}{share}",
                *(f"  {category}: {count}" for category, count in self.triage.by_category.items()),
                *([f"by {self.by}:", *(f"  {key}: {count}" for key, count in self.groups.items())] if self.by else []),
            ]
        )

    def to_dict(self) -> dict[str, object]:
        """Serializes the report to a JSON-ready dictionary."""
        return asdict(self)


async def grouped_counts(store: FeedbackStore, by: GroupBy) -> Mapping[str, int]:
    accepted = [
        row for row in await store.candidates() if row["is_steering"] and row["judge_version"] == PROMPT_VERSION
    ]
    match by:
        case "category":
            return dict(Counter(str(row["category"]) for row in accepted).most_common())
        case _:
            return dict(
                Counter(
                    project_label(str(row["origin_path"])) for row in accepted if row["origin_path"]
                ).most_common()
            )


async def collect_stats(db: Path | None, *, by: GroupBy | None = None) -> StatsReport:
    """Collects ingestion and triage statistics from the feedback store at ``db``.

    Args:
        db: The database path, or ``None`` for the default store.
        by: When set, group the accepted steering corpus by ``project`` or ``category``.

    Returns:
        A :class:`StatsReport` carrying the ingestion counts, triage coverage at the
        current judge prompt version, and the requested grouping.
    """
    async with await FeedbackStore.open(db or FeedbackStore.default_path()) as store:
        return StatsReport(
            ingestion=await store.stats(),
            triage=await store.triage_stats(prompt_version=PROMPT_VERSION),
            prompt_version=PROMPT_VERSION,
            by=by,
            groups=await grouped_counts(store, by) if by else {},
        )
