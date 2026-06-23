from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.sidecar import (
    Finding,
    candidate_text,
    candidates_for,
    closest_session,
    read_findings,
    to_candidate,
    worktree_uuid,
)
from tests.builders import assistant_text, user_text, write_transcript

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.anyio

UUID = "98228586-8a1e-494e-b73b-2c5352422812"


def finding(**overrides: Any) -> dict[str, Any]:
    return {
        "id": "ARCH-101",
        "file": "pkg/telemetry.py",
        "line": 167,
        "rule": "late-binding facade drops async-generator callable kind",
        "severity": "MEDIUM",
        "track": "arch",
        "evidence": "instrument_fn branches only on iscoroutinefunction",
        "suggested_fix": "Add an isasyncgenfunction branch",
    } | overrides


def write_sidecar(root: Path, records: list[dict[str, Any]], *, worktree: str = "expensive-tilapia") -> Path:
    path = root / UUID / "yasyf" / worktree / ".context" / "cleanup" / "issues.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def at(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp)


@pytest.mark.parametrize(
    ("ranges", "when", "expected"),
    [
        pytest.param(
            [("a", "2026-06-06T06:00:00+00:00", "2026-06-06T07:00:00+00:00")],
            "2026-06-06T06:30:00+00:00",
            "a",
            id="single-containment",
        ),
        pytest.param(
            [
                ("near", "2026-06-06T08:33:09+00:00", "2026-06-06T08:37:35+00:00"),
                ("far", "2026-06-06T12:00:00+00:00", "2026-06-06T13:00:00+00:00"),
            ],
            "2026-06-06T08:06:52+00:00",
            "near",
            id="nearest-edge-no-containment",
        ),
        pytest.param(
            [
                ("contains", "2026-06-06T06:00:00+00:00", "2026-06-06T09:00:00+00:00"),
                ("edge", "2026-06-06T08:33:09+00:00", "2026-06-06T08:37:35+00:00"),
            ],
            "2026-06-06T08:06:52+00:00",
            "contains",
            id="containment-beats-nearer-edge",
        ),
    ],
)
def test_closest_session_picks_by_containment_then_edge(
    ranges: list[tuple[str, str, str]], when: str, expected: str
) -> None:
    from pathlib import Path

    triples = [(Path(name), at(start), at(end)) for name, start, end in ranges]
    assert closest_session(triples, at(when)) == Path(expected)


def test_closest_session_empty_returns_none() -> None:
    assert closest_session([], at("2026-06-06T08:06:52+00:00")) is None


def test_read_findings_skips_dismissed(tmp_path: Path) -> None:
    sidecar = write_sidecar(
        tmp_path,
        [
            finding(id="ARCH-001", dismissed=True),
            finding(id="ARCH-101"),
            finding(id="CODE-001", line=1),
        ],
    )
    assert [f.id for f in read_findings(sidecar)] == ["ARCH-101", "CODE-001"]


def test_candidate_text_omits_none_fix() -> None:
    parsed = Finding.parse(finding(suggested_fix="none"))
    assert "Suggested fix" not in candidate_text(parsed)
    assert candidate_text(parsed) == f"{parsed.rule}\n{parsed.evidence}"


def test_candidate_text_includes_fix() -> None:
    parsed = Finding.parse(finding(suggested_fix="Add a branch"))
    assert candidate_text(parsed).endswith("Suggested fix: Add a branch")


def test_worktree_uuid_reads_path_tail(tmp_path: Path) -> None:
    sidecar = write_sidecar(tmp_path, [finding()])
    assert worktree_uuid(sidecar) == UUID


async def test_candidates_for_anchors_and_maps(tmp_path: Path) -> None:
    root = tmp_path / "transcripts"
    session = f"{UUID}-project"
    write_transcript(
        root / session / "sess-1.jsonl",
        [
            user_text("first prompt", uuid="u1", sessionId="sess-1", timestamp="2026-06-06T08:00:00+00:00"),
            assistant_text("reply", uuid="u2", sessionId="sess-1", timestamp="2026-06-06T08:10:00+00:00"),
        ],
    )
    sidecar = write_sidecar(
        tmp_path,
        [finding(id="ARCH-001", dismissed=True), finding(id="ARCH-101"), finding(id="CODE-001", line=1)],
    )
    os.utime(sidecar, (at("2026-06-06T08:05:00+00:00").timestamp(),) * 2)

    candidates = await candidates_for(sidecar, [root])

    assert [c.payload["finding_id"] for c in candidates] == ["ARCH-101", "CODE-001"]
    first = candidates[0]
    assert first.session_id == "sess-1"
    assert first.ref.event_uuid == "u1"
    assert first.occurred_at == at("2026-06-06T08:00:00+00:00")
    assert first.source_kind == "review_comment"
    assert first.signal.reasons == ("sidecar_finding", "medium")
    assert first.payload == {
        "format": "issues_jsonl",
        "file": "pkg/telemetry.py",
        "line_start": 167,
        "line_end": 167,
        "provenance": "surfaced",
        "severity": "MEDIUM",
        "track": "arch",
        "finding_id": "ARCH-101",
    }
    assert first.text.startswith("late-binding facade drops async-generator callable kind\n")
    assert first.text.endswith("Suggested fix: Add an isasyncgenfunction branch")
    assert candidates[0].dedup_key != candidates[1].dedup_key


async def test_candidates_for_no_session_returns_empty(tmp_path: Path) -> None:
    sidecar = write_sidecar(tmp_path, [finding()])
    assert await candidates_for(sidecar, [tmp_path / "empty-transcripts"]) == []


async def test_candidates_for_all_dismissed_returns_empty(tmp_path: Path) -> None:
    root = tmp_path / "transcripts"
    write_transcript(
        root / f"{UUID}-p" / "sess-1.jsonl",
        [user_text("p", uuid="u1", sessionId="sess-1", timestamp="2026-06-06T08:00:00+00:00")],
    )
    sidecar = write_sidecar(tmp_path, [finding(dismissed=True)])
    assert await candidates_for(sidecar, [root]) == []


def test_to_candidate_dedup_key_is_stable() -> None:
    from cc_transcript.activity import SessionActivity
    from cc_transcript.ids import EventRef, SessionId

    from cc_pushback.sidecar import Anchor
    from tests.builders import parse

    events = parse(
        [user_text("p", uuid="u1", sessionId="sess-1", timestamp="2026-06-06T08:00:00+00:00")]
    )
    activity = SessionActivity.from_events(SessionId("sess-1"), events)
    ref = EventRef(SessionId("sess-1"), events[0].meta.uuid)  # type: ignore[union-attr]
    anchor = Anchor(SessionId("sess-1"), ref, activity, at("2026-06-06T08:00:00+00:00"))
    from pathlib import Path

    parsed = Finding.parse(finding())
    one = to_candidate(Path("/x/issues.jsonl"), parsed, anchor)
    two = to_candidate(Path("/x/issues.jsonl"), parsed, anchor)
    assert one.dedup_key == two.dedup_key
