"""The per-category steering profile built from the refined pairs.

Groups the ``refined_pairs`` view into the recurring themes of one developer's
steering — per category: a checklist line, the distinct directions ranked by how
often they recur, and verbatim quotes with the repository they came from. The
build and both renders are fully mechanical, so ``cc-merge`` can regenerate
reviewer grounding from a database alone; :func:`distill` is the one optional
LLM step, a cached rewrite of the mechanical markdown into tighter prose using
the same spawnllm-backed structured call as the refine stage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript.judge import structured_judge
from pydantic import BaseModel

from cc_steer.report import project_label, truncate

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from spawnllm import TModel

DISTILL_PROMPT_VERSION = 1
PROFILES_DIR = Path.home() / ".cc-steer" / "profiles"
DIRECTIONS_PER_CATEGORY = 12
CITES_PER_CATEGORY = 3
CITE_TEXT_LIMIT = 240

CATEGORY_FRAMING: dict[str, tuple[str, str]] = {
    "wrong_approach": (
        "Approach and design",
        "Plans and strategies this developer rejected. Check a new approach against these before building on it.",
    ),
    "incorrect_change": (
        "Correctness",
        "Changes that landed wrong or broken, and the fixes they had to ask for.",
    ),
    "unwanted_action": (
        "Unasked-for work",
        "Work nobody asked for. Stay inside the request.",
    ),
    "style_violation": (
        "Style and conventions",
        "House conventions the work has crossed. Treat these as standing rules.",
    ),
    "premature": (
        "Finishing the job",
        "Places where work stopped early or was called done too soon. Finish before reporting.",
    ),
    "direction": (
        "Resolved choices",
        "Choices this developer has already settled. Follow the picks instead of re-asking.",
    ),
}

DISTILL_PROMPT = """\
You are rewriting a developer's steering profile — a mechanical digest of the
corrections and choices they send an AI coding assistant — into the style guide
they would have written themselves.

Rules:
- Keep every category section and the opening checklist; tighten inside them.
- Merge directions that say the same thing, keeping the strongest phrasing, and
  drop the support counts from the prose.
- Quotes are evidence: reproduce them character-for-character or drop them,
  never paraphrase them.
- Write rules in the imperative ("do X"), not observations ("the developer
  prefers X").
- Return only the rewritten markdown document.

=== MECHANICAL PROFILE ===
{profile}"""


def direction_key(text: str) -> str:
    return " ".join(text.split()).rstrip(".").casefold()


def pair_repo(row: Mapping[str, object]) -> str | None:
    return project_label(str(row["origin_path"])) if row["origin_path"] else None


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" + ("" if count == 1 else "s")


def category_framing(category: str) -> tuple[str, str]:
    return CATEGORY_FRAMING.get(category) or (
        category.replace("_", " ").capitalize(),
        "What keeps coming up, ranked by recurrence.",
    )


@dataclass(frozen=True, slots=True)
class Direction:
    """One distinct steering direction with its recurrence.

    Attributes:
        text: The distilled one-sentence direction, as first refined.
        support: How many refined pairs voice this direction.
    """

    text: str
    support: int


@dataclass(frozen=True, slots=True)
class Cite:
    """One verbatim steering quote grounding a category.

    Attributes:
        verbatim: The developer's exact words, whitespace-collapsed and capped.
        repo: The repository the steering happened in, when known.
    """

    verbatim: str
    repo: str | None


@dataclass(frozen=True, slots=True)
class CategoryProfile:
    """One steering category's slice of the profile.

    Attributes:
        category: The triage category slug.
        pair_count: How many refined pairs the category holds in scope.
        checklist: The category's one-line check — its strongest direction.
        directions: The distinct directions, ranked by support.
        cites: Verbatim quotes grounding the top-ranked directions.
    """

    category: str
    pair_count: int
    checklist: str
    directions: tuple[Direction, ...]
    cites: tuple[Cite, ...]


@dataclass(frozen=True, slots=True)
class SteeringProfile:
    """A per-category digest of how one developer steers Claude.

    Attributes:
        repo: The single repository in scope, or ``None`` for the global profile.
        pair_count: The total refined pairs in scope.
        sessions: The distinct sessions the pairs came from.
        first: The earliest steering date in scope.
        last: The latest steering date in scope.
        max_event_id: The highest feedback event id in scope — the freshness marker.
        refine_version: The highest refine prompt version in scope — the other
            freshness marker, since re-refining an unchanged corpus rewrites the
            direction text without moving any event id.
        categories: The category slices, largest first.
    """

    repo: str | None
    pair_count: int
    sessions: int
    first: str
    last: str
    max_event_id: int
    refine_version: int
    categories: tuple[CategoryProfile, ...]


class DistilledProfile(BaseModel):
    """The LLM's tightened rewrite of a mechanical profile.

    Attributes:
        markdown: The rewritten profile, ready to serve as reviewer grounding.
    """

    markdown: str


def category_profile(category: str, rows: Sequence[Mapping[str, object]]) -> CategoryProfile:
    votes: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        votes[direction_key(str(row["direction"]))].append(row)
    ranked = sorted(
        votes.values(),
        key=lambda group: (len(group), max(str(row["occurred_at"]) for row in group)),
        reverse=True,
    )[:DIRECTIONS_PER_CATEGORY]
    directions = tuple(Direction(text=" ".join(str(group[0]["direction"]).split()), support=len(group)) for group in ranked)
    quotable = sorted(ranked, key=lambda group: len(str(group[0]["direction_verbatim"])), reverse=True)
    return CategoryProfile(
        category=category,
        pair_count=len(rows),
        checklist=directions[0].text,
        directions=directions,
        cites=tuple(
            Cite(
                verbatim=truncate(" ".join(str(group[0]["direction_verbatim"]).split()), CITE_TEXT_LIMIT),
                repo=pair_repo(group[0]),
            )
            for group in quotable[:CITES_PER_CATEGORY]
        ),
    )


def build_profile(rows: Sequence[Mapping[str, object]], *, repo: str | None = None) -> SteeringProfile:
    """Builds the mechanical steering profile from ``refined_pairs`` rows.

    Args:
        rows: Rows of the ``refined_pairs`` view, e.g. from :meth:`FeedbackStore.pairs`.
        repo: When set, scope to pairs whose transcript came from this repository;
            ``None`` aggregates every repository.

    Returns:
        The profile, categories ordered largest first.

    Raises:
        ValueError: When no refined pairs fall in scope.
    """
    if not (scoped := [row for row in rows if repo is None or pair_repo(row) == repo]):
        raise ValueError(f"no refined pairs in scope (repo={repo!r})")
    groups: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in scoped:
        groups[str(row["category"])].append(row)
    times = sorted(str(row["occurred_at"]) for row in scoped)
    return SteeringProfile(
        repo=repo,
        pair_count=len(scoped),
        sessions=len({row["session_id"] for row in scoped if row["session_id"]}),
        first=times[0][:10],
        last=times[-1][:10],
        max_event_id=max(int(str(row["event_id"])) for row in scoped),
        refine_version=max(int(str(row["prompt_version"])) for row in scoped),
        categories=tuple(
            category_profile(category, members)
            for category, members in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
        ),
    )


def category_section(profile: CategoryProfile) -> list[str]:
    title, lead = category_framing(profile.category)
    return [
        "",
        f"## {title}",
        "",
        f"{lead} ({plural(profile.pair_count, 'pair')})",
        "",
        *(
            f"{rank}. {direction.text}" + (f" (×{direction.support})" if direction.support > 1 else "")
            for rank, direction in enumerate(profile.directions, start=1)
        ),
        "",
        "In their own words:",
        "",
        *(f'> "{cite.verbatim}"' + (f" — {cite.repo}" if cite.repo else "") for cite in profile.cites),
    ]


def render_markdown(profile: SteeringProfile) -> str:
    """Renders the profile as a user-facing markdown style guide."""
    header = [
        f"# Steering Profile — {profile.repo or 'all repos'}",
        "",
        f"How this developer steers Claude, built from {plural(profile.pair_count, 'refined steering pair')} "
        f"across {plural(profile.sessions, 'session')}, {profile.first} to {profile.last}. "
        "Each theme collects the steering that kept recurring. The checklist is the short form; "
        "the themes below carry the ranked directions and the developer's own words.",
        "",
        "## Before you ship",
        "",
        *(f"- [ ] **{category_framing(entry.category)[0]}** — {entry.checklist}" for entry in profile.categories),
    ]
    return "\n".join(header + [line for entry in profile.categories for line in category_section(entry)]) + "\n"


def render_json(profile: SteeringProfile) -> str:
    """Serializes the profile as pretty-printed JSON."""
    return json.dumps(asdict(profile), indent=2)


def cache_path(profile: SteeringProfile, cache_dir: Path) -> Path:
    return cache_dir / (
        f"{profile.repo or 'global'}-{profile.pair_count}-{profile.max_event_id}"
        f"-r{profile.refine_version}-v{DISTILL_PROMPT_VERSION}.md"
    )


async def distill(profile: SteeringProfile, *, tier: TModel = "medium", cache_dir: Path | None = None) -> str:
    """Rewrites the mechanical profile into tighter prose via one cached LLM call.

    The rewrite caches under ``cache_dir`` keyed on the profile's repo, pair
    count, max event id, refine prompt version, and the distill prompt version,
    so an unchanged corpus never re-spends the call.

    Args:
        profile: The mechanical profile to rewrite.
        tier: The abstract model tier to run.
        cache_dir: The cache directory; defaults to ``~/.cc-steer/profiles``.

    Returns:
        The distilled markdown.

    Raises:
        cc_transcript.judge.JudgeError: When the LLM call fails or the response
            does not validate.
    """
    cache_dir = cache_dir or PROFILES_DIR
    if (path := cache_path(profile, cache_dir)).exists():
        return path.read_text()
    rewrite = await structured_judge(DistilledProfile, tier=tier)(
        DISTILL_PROMPT.format(profile=render_markdown(profile))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(rewrite.markdown)
    return rewrite.markdown
