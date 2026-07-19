from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from cc_steer.profile import (
    DISTILL_PROMPT_VERSION,
    Cite,
    Direction,
    DistilledProfile,
    build_profile,
    distill,
    render_json,
    render_markdown,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

pytestmark = pytest.mark.anyio

STEER = "/h/.claude/projects/-Users-y-Code-steer/a.jsonl"
OTHER = "/h/.claude/projects/-Users-y-projects-other/b.jsonl"


def pair(
    event_id: int,
    category: str,
    direction: str,
    *,
    verbatim: str = "no, stop doing that",
    origin: str | None = STEER,
    session: str = "s1",
    occurred_at: str = "2026-06-01T12:00:00+00:00",
    prompt_version: int = 1,
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "pair_index": 0,
        "category": category,
        "direction": direction,
        "direction_verbatim": verbatim,
        "origin_path": origin,
        "session_id": session,
        "occurred_at": occurred_at,
        "prompt_version": prompt_version,
    }


def corpus() -> list[dict[str, Any]]:
    return [
        pair(
            1,
            "style_violation",
            "Use frozen dataclasses.",
            verbatim="no — frozen dataclass, we never pass raw dicts around",
            occurred_at="2026-04-10T09:00:00+00:00",
        ),
        pair(2, "style_violation", "use  frozen\ndataclasses", verbatim="why is this a dict"),
        pair(
            3,
            "style_violation",
            "Keep try blocks minimal.",
            verbatim="only the throwing line goes in the try",
            session="s2",
            origin=OTHER,
        ),
        pair(
            4,
            "wrong_approach",
            "Ship the small fix first.",
            verbatim="don't rewrite the module, just fix the bug",
            session="s2",
            origin=OTHER,
            occurred_at="2026-06-15T09:00:00+00:00",
        ),
    ]


@pytest.mark.unit
def test_build_profile_groups_ranks_and_counts() -> None:
    profile = build_profile(corpus())
    assert profile.repo is None
    assert (profile.pair_count, profile.sessions, profile.max_event_id) == (4, 2, 4)
    assert (profile.first, profile.last) == ("2026-04-10", "2026-06-15")
    assert [entry.category for entry in profile.categories] == ["style_violation", "wrong_approach"]

    style = profile.categories[0]
    assert style.pair_count == 3
    assert style.directions == (
        Direction(text="Use frozen dataclasses.", support=2),
        Direction(text="Keep try blocks minimal.", support=1),
    )
    assert style.checklist == "Use frozen dataclasses."
    assert style.cites == (
        Cite(verbatim="no — frozen dataclass, we never pass raw dicts around", repo="steer"),
        Cite(verbatim="only the throwing line goes in the try", repo="other"),
    )
    assert profile.categories[1].pair_count == 1


@pytest.mark.unit
def test_build_profile_scopes_to_one_repo() -> None:
    rows = [*corpus(), pair(5, "direction", "Pin Python at 3.13.", origin=None, session="s3")]
    scoped = build_profile(rows, repo="other")
    assert scoped.repo == "other"
    assert (scoped.pair_count, scoped.max_event_id, scoped.sessions) == (2, 4, 1)
    assert {entry.category for entry in scoped.categories} == {"style_violation", "wrong_approach"}
    assert build_profile(rows).pair_count == 5  # global keeps the repo-less pair


@pytest.mark.unit
def test_build_profile_raises_on_empty_scope() -> None:
    with pytest.raises(ValueError, match="repo='missing'"):
        build_profile(corpus(), repo="missing")
    with pytest.raises(ValueError, match="no refined pairs"):
        build_profile([])


@pytest.mark.unit
def test_direction_ties_rank_by_recency_and_cites_prefer_substance() -> None:
    rows = [
        pair(1, "direction", "Pick option one.", verbatim="1", occurred_at="2026-05-01T00:00:00+00:00"),
        pair(
            2,
            "direction",
            "Ship binaries via GitHub Releases.",
            verbatim="build it in Go and ship release binaries",
            occurred_at="2026-06-01T00:00:00+00:00",
        ),
    ]
    category = build_profile(rows).categories[0]
    assert category.checklist == "Ship binaries via GitHub Releases."  # the newer of tied directions leads
    assert [d.text for d in category.directions] == ["Ship binaries via GitHub Releases.", "Pick option one."]
    assert category.cites[0] == Cite(verbatim="build it in Go and ship release binaries", repo="steer")


@pytest.mark.unit
def test_render_markdown_reads_like_a_style_guide() -> None:
    rendered = render_markdown(build_profile(corpus(), repo="steer"))
    assert rendered.startswith("# Steering Profile — steer\n")
    assert "2 refined steering pairs across 1 session, 2026-04-10 to 2026-06-01" in rendered
    assert "## Before you ship" in rendered
    assert "- [ ] **Style and conventions** — Use frozen dataclasses." in rendered
    assert "## Style and conventions" in rendered
    assert "Treat these as standing rules. (2 pairs)" in rendered
    assert "1. Use frozen dataclasses. (×2)" in rendered
    assert '> "no — frozen dataclass, we never pass raw dicts around" — steer' in rendered


@pytest.mark.unit
def test_render_markdown_global_header_singletons_and_unknown_category() -> None:
    rendered = render_markdown(build_profile([pair(1, "future_thing", "Do the new thing.", origin=None)]))
    assert rendered.startswith("# Steering Profile — all repos\n")
    assert "1 refined steering pair across 1 session" in rendered
    assert "- [ ] **Future thing** — Do the new thing." in rendered
    assert "## Future thing" in rendered
    assert "(1 pair)" in rendered
    assert "1. Do the new thing.\n" in rendered  # support of one carries no ×count
    assert '> "no, stop doing that"\n' in rendered  # no repo, no attribution


@pytest.mark.unit
def test_render_json_round_trips() -> None:
    parsed = json.loads(render_json(build_profile(corpus())))
    assert parsed["repo"] is None
    assert (parsed["pair_count"], parsed["max_event_id"]) == (4, 4)
    style = parsed["categories"][0]
    assert style["category"] == "style_violation"
    assert style["checklist"] == "Use frozen dataclasses."
    assert style["directions"][0] == {"text": "Use frozen dataclasses.", "support": 2}
    assert style["cites"][0] == {"verbatim": "no — frozen dataclass, we never pass raw dicts around", "repo": "steer"}


@pytest.mark.unit
async def test_distill_caches_on_corpus_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_judge(
        model_cls: type[DistilledProfile], *, tier: str
    ) -> Callable[[str], Awaitable[DistilledProfile]]:
        async def run(prompt: str) -> DistilledProfile:
            calls.append(prompt)
            return model_cls(markdown="# tightened")

        return run

    monkeypatch.setattr("cc_steer.profile.structured_judge", fake_judge)
    profile = build_profile(corpus())

    assert await distill(profile, cache_dir=tmp_path) == "# tightened"
    assert len(calls) == 1
    assert "=== MECHANICAL PROFILE ===" in calls[0]
    assert "# Steering Profile — all repos" in calls[0]
    assert (tmp_path / f"global-4-4-r1-v{DISTILL_PROMPT_VERSION}.md").read_text() == "# tightened"

    assert await distill(profile, cache_dir=tmp_path) == "# tightened"
    assert len(calls) == 1  # unchanged corpus hits the cache

    grown = build_profile([*corpus(), pair(9, "direction", "Pin Python at 3.13.")])
    assert await distill(grown, cache_dir=tmp_path) == "# tightened"
    assert len(calls) == 2  # new max_event_id and pair_count mint a new key
    assert (tmp_path / f"global-5-9-r1-v{DISTILL_PROMPT_VERSION}.md").exists()


@pytest.mark.unit
async def test_distill_scoped_cache_key_carries_the_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_judge(
        model_cls: type[DistilledProfile], *, tier: str
    ) -> Callable[[str], Awaitable[DistilledProfile]]:
        async def run(prompt: str) -> DistilledProfile:
            return model_cls(markdown="scoped")

        return run

    monkeypatch.setattr("cc_steer.profile.structured_judge", fake_judge)
    scoped = build_profile(corpus(), repo="steer")
    assert await distill(scoped, cache_dir=tmp_path) == "scoped"
    assert (tmp_path / f"steer-2-2-r1-v{DISTILL_PROMPT_VERSION}.md").read_text() == "scoped"


@pytest.mark.unit
async def test_distill_rekeys_on_refine_version_bump(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_judge(
        model_cls: type[DistilledProfile], *, tier: str
    ) -> Callable[[str], Awaitable[DistilledProfile]]:
        async def run(prompt: str) -> DistilledProfile:
            calls.append(prompt)
            return model_cls(markdown="# tightened")

        return run

    monkeypatch.setattr("cc_steer.profile.structured_judge", fake_judge)
    original = build_profile(corpus())
    rerefined = build_profile([{**row, "prompt_version": 2} for row in corpus()])
    assert original.refine_version == 1
    assert rerefined.refine_version == 2
    assert (rerefined.pair_count, rerefined.max_event_id) == (original.pair_count, original.max_event_id)

    assert await distill(original, cache_dir=tmp_path) == "# tightened"
    assert await distill(rerefined, cache_dir=tmp_path) == "# tightened"
    assert len(calls) == 2  # same size and event id, but a new refine version re-keys instead of colliding
    assert (tmp_path / f"global-4-4-r2-v{DISTILL_PROMPT_VERSION}.md").exists()
