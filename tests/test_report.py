from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from cc_pushback.context import ContextSnapshot, ContextTurn
from cc_pushback.report import (
    Sample,
    build_summary,
    candidate_pool,
    corpus_stats,
    is_noise,
    parse_summary_json,
    project_label,
    render_html,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

PROJ = "/h/.claude/projects/-Users-y-Code-proj/sess.jsonl"
OTHER = "/h/.claude/projects/-Users-y-projects-other/sess.jsonl"


def make_sample(
    event_id: int,
    kind: str,
    text: str,
    *,
    payload: dict[str, Any] | None = None,
    occurred_at: str = "2026-05-01T12:00:00+00:00",
    session: str = "s1",
    origin: str = PROJ,
    before: Sequence[ContextTurn] = (),
    trigger: ContextTurn | None = None,
    after: Sequence[ContextTurn] = (),
) -> Sample:
    return Sample(
        id=event_id,
        source_kind=kind,
        occurred_at=occurred_at,
        text=text,
        payload=payload or {},
        context=ContextSnapshot(before=tuple(before), trigger=trigger, after=tuple(after)),
        origin_path=origin,
        session_id=session,
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param("[Request interrupted by user for tool use]", True, id="interrupt-marker"),
        pytest.param("Stop hook feedback:\nError: ...", True, id="hook-error"),
        pytest.param("   ", True, id="whitespace"),
        pytest.param("too short", True, id="under-ten-chars"),
        pytest.param("a lot of these can be helpers inside the framework", False, id="real-pushback"),
    ],
)
def test_is_noise(text: str, expected: bool) -> None:
    assert is_noise(text) is expected


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
    samples = [make_sample(i, "transcript_message", f"a substantive piece of pushback number {i}") for i in range(12)]
    samples.append(make_sample(99, "transcript_message", "[Request interrupted by user]"))
    pool = candidate_pool(samples)
    assert len(pool["transcript_message"]) == 8
    assert 99 not in {s.id for s in pool["transcript_message"]}


@pytest.mark.anyio
async def test_build_summary_heuristic_has_no_narrative() -> None:
    summary = await build_summary(corpus(), use_llm=False, model="m")
    assert summary.narrative is None
    assert summary.highlights
    assert {h.event_id for h in summary.highlights} <= {1, 3}


@pytest.mark.anyio
async def test_build_summary_uses_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cc_pushback.report.claude_available", lambda: True)

    async def fake_run(*_: object, **__: object) -> str:
        return '{"narrative": "Terse and direct.", "highlights": [{"id": 1, "why": "cites a file"}]}'

    monkeypatch.setattr("cc_pushback.report.run_claude", fake_run)
    summary = await build_summary(corpus(), use_llm=True, model="m")
    assert summary.narrative == "Terse and direct."
    assert summary.highlights == (type(summary.highlights[0])(1, "cites a file"),)


@pytest.mark.anyio
async def test_build_summary_falls_back_when_claude_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    monkeypatch.setattr("cc_pushback.report.claude_available", lambda: True)

    async def boom(*_: object, **__: object) -> str:
        raise subprocess.TimeoutExpired(cmd="claude", timeout=1)

    monkeypatch.setattr("cc_pushback.report.run_claude", boom)
    summary = await build_summary(corpus(), use_llm=True, model="m")
    assert summary.narrative is None
    assert summary.highlights


@pytest.mark.parametrize(
    ("raw", "ok"),
    [
        pytest.param('{"narrative": "x", "highlights": [{"id": 1, "why": "y"}]}', True, id="clean"),
        pytest.param('Here is the result:\n{"narrative": "x", "highlights": []}\nDone.', True, id="wrapped-in-prose"),
        pytest.param("not json at all", False, id="garbage"),
        pytest.param('{"highlights": []}', False, id="missing-narrative"),
    ],
)
def test_parse_summary_json(raw: str, ok: bool) -> None:
    assert (parse_summary_json(raw) is not None) is ok


@pytest.mark.anyio
async def test_render_html_escapes_and_includes_controls() -> None:
    trigger = ContextTurn(role="assistant", text="I built the thing", tool_calls=("Edit",))
    samples = [
        make_sample(
            7,
            "review_comment",
            "<script>alert('xss')</script> this should be escaped",
            payload={"format": "superset-inline", "file": ".claude/hooks/style.py", "line_start": 17},
            before=(ContextTurn(role="user", text="please do X"), trigger),
            trigger=trigger,
            after=(ContextTurn(role="user", text="that was wrong"),),
        )
    ]
    summary = await build_summary(samples, use_llm=False, model="m")
    html = render_html(samples, summary)

    assert "&lt;script&gt;alert(&#x27;xss&#x27;)" in html
    assert "<script>alert('xss')" not in html
    assert 'class="badge badge-review_comment"' in html
    assert 'id="search"' in html and 'id="hide-noise"' in html
    assert 'data-kind="review_comment"' in html
    assert "turn-trigger" in html
    assert ".claude/hooks/style.py:17" in html
    assert "<details" in html


def test_sample_from_row_round_trips() -> None:
    snapshot = ContextSnapshot(before=(ContextTurn(role="user", text="hello there"),), trigger=None, after=())
    row = {
        "id": 5,
        "source_kind": "plan_review",
        "occurred_at": "2026-05-01T00:00:00+00:00",
        "text": "do it differently",
        "payload_json": '{"detector": "plan_reentry"}',
        "context_json": snapshot.to_json(),
        "origin_path": PROJ,
        "session_id": "abc",
    }
    sample = Sample.from_row(row)
    assert sample.id == 5
    assert sample.payload == {"detector": "plan_reentry"}
    assert sample.context.before[0].text == "hello there"
