from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cc_pushback.models import ContextSnapshot, FeedbackCandidate
from cc_pushback.sources.base import dedup_key

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from cc_pushback.repo import Repository

__all__ = ["ISSUES_GLOB", "SupersetIssues"]

SOURCE_KIND = "superset_issue"
ISSUES_GLOB = ".context/cleanup/issues.jsonl"
EMPTY_CONTEXT = ContextSnapshot(before=(), trigger=None, after=())


def issue_text(obj: dict[str, Any], raw: str) -> str:
    match (obj.get("rule"), obj.get("suggested_fix")):
        case (str() as rule, str() as fix):
            return f"{rule}\n{fix}"
        case (str() as rule, _):
            return rule
        case _:
            return raw


def changed_issue_files(repo: Repository, roots: Sequence[Path]) -> Iterator[tuple[Path, float]]:
    known = repo.file_mtimes()
    return (
        (path, mtime)
        for root in roots
        for path in root.rglob(ISSUES_GLOB)
        if (mtime := path.stat().st_mtime)
        if (prev := known.get(str(path))) is None or prev < mtime
    )


class SupersetIssues:
    """Ingests superset cleanup issues from ``.context/cleanup/issues.jsonl``.

    Each JSONL line is one issue. The whole object is preserved as the payload;
    the candidate text is composed from ``rule`` and ``suggested_fix`` when both
    exist, the rule alone when only it exists, else the raw line.
    """

    def candidates_for_file(self, path: Path, mtime: float) -> Iterator[FeedbackCandidate]:
        return (
            FeedbackCandidate(
                dedup_key=dedup_key(str(path), str(number), SOURCE_KIND),
                source_kind=SOURCE_KIND,
                occurred_at=datetime.fromtimestamp(mtime, UTC),
                text=issue_text(json.loads(line), line),
                context=EMPTY_CONTEXT,
                origin_path=path,
                payload=json.loads(line),
            )
            for number, line in enumerate(path.read_text().splitlines())
            if line.strip()
        )
