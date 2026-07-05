from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from cc_transcript.context import ContextWindow, TurnRef
from cc_transcript.ids import EventRef, EventUuid, SessionId
from cc_transcript.mining import firm, noise, weak
from cc_transcript.mining.confidence import to_payload

from cc_steer.report import (
    Sample,
    build_summary,
    candidate_pool,
    corpus_stats,
    is_noise,
    parse_summary_json,
    project_label,
)

if TYPE_CHECKING:
    from cc_transcript.mining import CandidateSignal

PROJ = "/h/.claude/projects/-Users-y-Code-proj/sess.jsonl"
OTHER = "/h/.claude/projects/-Users-y-projects-other/sess.jsonl"


def preview_window(*previews: str) -> ContextWindow:
    return ContextWindow(
        anchor=EventRef(SessionId("s1"), EventUuid("u1")),
        before=tuple(TurnRef(role="user", refs=(), preview=preview, tool_digests=()) for preview in previews),
        trigger=None,
        after=(),
        fidelity="summary",
        preview_chars=200,
    )


def make_sample(
    event_id: int,
    kind: str,
    text: str,
    *,
    payload: dict[str, Any] | None = None,
    occurred_at: str = "2026-05-01T12:00:00+00:00",
    session: str = "s1",
    origin: str = PROJ,
    signal: CandidateSignal = firm("transcript_message"),
) -> Sample:
    return Sample(
        id=event_id,
        source_kind=kind,
        occurred_at=occurred_at,
        text=text,
        payload=payload or {},
        window=preview_window(),
        origin_path=origin,
        session_id=session,
        signal=signal,
    )


@pytest.mark.parametrize(
    ("signal", "expected"),
    [
        pytest.param(noise("empty"), True, id="none-confidence-is-noise"),
        pytest.param(weak("bare_marker"), False, id="low-confidence-survives"),
        pytest.param(firm("transcript_message"), False, id="medium-confidence-survives"),
    ],
)
def test_is_noise(signal: CandidateSignal, expected: bool) -> None:
    assert is_noise(make_sample(1, "transcript_message", "any text", signal=signal)) is expected


def test_project_label_decodes_marker() -> None:
    assert project_label(PROJ) == "proj"
    assert project_label(OTHER) == "other"


def corpus() -> list[Sample]:
    return [
        make_sample(
            1,
            "review_comment",
            "a lot of these can be helpers inside the framework",
            occurred_at="2026-04-10T09:00:00+00:00",
            session="A",
        ),
        make_sample(
            2,
            "transcript_message",
            "[Request interrupted by user]",
            occurred_at="2026-05-02T09:00:00+00:00",
            session="B",
            signal=noise("bare_marker"),
        ),
        make_sample(
            3,
            "plan_review",
            "actually do Y instead of X please",
            occurred_at="2026-05-15T09:00:00+00:00",
            session="B",
            origin=OTHER,
        ),
    ]


def test_corpus_stats_counts() -> None:
    stats = corpus_stats(corpus())
    assert stats.total == 3
    assert stats.by_kind == {"review_comment": 1, "transcript_message": 1, "plan_review": 1}
    assert stats.noise == 1
    assert stats.sessions == 2
    assert stats.projects == 2
    assert (stats.first, stats.last) == ("2026-04-10", "2026-05-15")
    assert stats.by_month == {"2026-04": 1, "2026-05": 2}


def test_candidate_pool_excludes_noise_and_caps() -> None:
    samples = [make_sample(i, "transcript_message", f"a substantive piece of steering number {i}") for i in range(12)]
    samples.append(make_sample(99, "transcript_message", "[Request interrupted by user]", signal=noise("bare_marker")))
    pool = candidate_pool(samples)
    assert len(pool["transcript_message"]) == 8
    assert 99 not in {s.id for s in pool["transcript_message"]}


@pytest.mark.anyio
async def test_build_summary_uses_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*_: object, **__: object) -> str:
        return '{"narrative": "Terse and direct.", "highlights": [{"id": 1, "why": "cites a file"}]}'

    monkeypatch.setattr("cc_steer.report.run_claude", fake_run)
    summary = await build_summary(corpus(), model="m")
    assert summary.narrative == "Terse and direct."
    assert summary.highlights == (type(summary.highlights[0])(1, "cites a file"),)


@pytest.mark.anyio
async def test_build_summary_serves_an_all_noise_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*_: object, **__: object) -> str:
        return '{"narrative": "Mostly bare interruptions.", "highlights": [{"id": 2, "why": "made up"}]}'

    monkeypatch.setattr("cc_steer.report.run_claude", fake_run)
    all_noise = [
        make_sample(2, "transcript_message", "[Request interrupted by user]", signal=noise("bare_marker")),
        make_sample(4, "transcript_message", "[Request interrupted by user]", signal=noise("bare_marker")),
    ]
    summary = await build_summary(all_noise, model="m")
    assert summary.highlights == ()
    assert summary.narrative == "Mostly bare interruptions."


@pytest.mark.anyio
async def test_build_summary_raises_when_picks_miss_a_nonempty_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(*_: object, **__: object) -> str:
        return '{"narrative": "x", "highlights": [{"id": 999, "why": "hallucinated"}]}'

    monkeypatch.setattr("cc_steer.report.run_claude", fake_run)
    with pytest.raises(ValueError, match="no valid highlight ids"):
        await build_summary(corpus(), model="m")


@pytest.mark.anyio
async def test_build_summary_raises_when_claude_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    async def boom(*_: object, **__: object) -> str:
        raise subprocess.SubprocessError("claude timed out after 1s")

    monkeypatch.setattr("cc_steer.report.run_claude", boom)
    with pytest.raises(subprocess.SubprocessError):
        await build_summary(corpus(), model="m")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        pytest.param(
            '{"narrative": "x", "highlights": [{"id": 1, "why": "y"}]}',
            ("x", [{"id": 1, "why": "y"}]),
            id="clean",
        ),
        pytest.param(
            'Here is the result:\n{"narrative": "x", "highlights": []}\nDone.',
            ("x", []),
            id="wrapped-in-prose",
        ),
    ],
)
def test_parse_summary_json(raw: str, expected: tuple[str, list[dict[str, Any]]]) -> None:
    assert parse_summary_json(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("not json at all", id="garbage"),
        pytest.param('{"highlights": []}', id="missing-narrative"),
    ],
)
def test_parse_summary_json_raises(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_summary_json(raw)


def test_sample_from_row_round_trips() -> None:
    window = preview_window("hello there")
    payload = {"detector": "plan_reentry", "signal": to_payload(firm("plan_reentry"))}
    row = {
        "id": 5,
        "source_kind": "plan_review",
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "text": "do it differently",
        "payload_json": json.dumps(payload),
        "context_json": window.to_json(),
        "event_uuid": "u5",
        "origin_path": PROJ,
        "session_id": "abc",
    }
    sample = Sample.from_row(row)
    assert sample.id == 5
    assert sample.payload == payload
    assert sample.window == window
    assert sample.signal == firm("plan_reentry")
