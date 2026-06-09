"""Render the collected feedback corpus into a single self-contained HTML page.

The page leads with a corpus summary and a handful of highlights, then lists every
sample with a kind filter, a free-text search, and an expandable context window. The
summary and highlights are written by the ``claude`` CLI when it is available and
fall back to deterministic heuristics otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING

from cc_pushback.claude import claude_available, run_claude
from cc_pushback.context import ContextSnapshot

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from cc_pushback.context import ContextTurn

CONTEXT_TURN_LIMIT = 700
SAMPLE_TEXT_LIMIT = 400
HIGHLIGHT_POOL_PER_KIND = 8
HEURISTIC_HIGHLIGHTS = 12
NOISE_PREFIXES = ("[Request interrupted", "Stop hook feedback:")

SUMMARY_SYSTEM = """\
You analyze a developer's "pushback" — the corrective feedback they give an AI coding assistant.
You receive corpus statistics and a numbered pool of real feedback samples.
Return ONLY a JSON object, with no prose around it, of exactly this shape:
{"narrative": "<2-4 sentences on the developer's pushback style and recurring themes>",
 "highlights": [{"id": <sample id>, "why": "<one short clause on why it is representative>"}]}
Pick 8-12 highlights, only from the provided sample ids, favoring variety across feedback kinds.
"""

CSS = """
:root{--bg:#0d1117;--panel:#161b22;--border:#30363d;--fg:#e6edf3;--muted:#8b949e;--accent:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
h1,h2{font-weight:600}
header.top{padding:24px;border-bottom:1px solid var(--border)}
header.top .sub{color:var(--muted)}
section{padding:16px 24px}
.stat-cards{display:flex;gap:12px;flex-wrap:wrap}
.stat{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
.stat .n{font-size:20px;font-weight:600}
.stat .l{color:var(--muted);font-size:12px}
table.dist{border-collapse:collapse;margin-top:14px}
table.dist td{padding:2px 10px 2px 0;white-space:nowrap}
.bar{display:inline-block;height:10px;background:var(--accent);border-radius:3px;vertical-align:middle}
.months{display:flex;gap:3px;align-items:flex-end;margin-top:14px}
.mcol{display:flex;flex-direction:column;align-items:center;justify-content:flex-end}
.mcol .m{width:22px;background:var(--accent);border-radius:3px 3px 0 0}
.mcol span{font-size:9px;color:var(--muted);margin-top:3px}
.narrative{background:var(--panel);border:1px solid var(--border);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 18px;max-width:80ch;margin-top:14px}
#controls{position:sticky;top:0;background:var(--bg);display:flex;gap:8px;align-items:center;
flex-wrap:wrap;border-bottom:1px solid var(--border);z-index:2}
.kind-btn{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:14px;
padding:4px 12px;cursor:pointer;font:inherit}
.kind-btn.active{background:var(--accent);color:#0d1117;border-color:var(--accent)}
#search{flex:1;min-width:200px;background:var(--panel);color:var(--fg);border:1px solid var(--border);
border-radius:6px;padding:6px 10px;font:inherit}
#count{color:var(--muted)}
label.noise{color:var(--muted);display:flex;gap:4px;align-items:center;cursor:pointer}
.card{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin:12px 0}
.card header{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#21262d;border:1px solid var(--border)}
.badge-transcript_message{color:#8b949e}.badge-review_comment{color:#7ee787}.badge-plan_review{color:#d2a8ff}
.badge-interrupt_rejection{color:#ff7b72}.badge-superset_issue{color:#ffa657}
time{color:var(--muted);font-size:12px}
.chip{font-size:11px;color:var(--muted);background:#21262d;border-radius:6px;padding:1px 6px}
.text pre{white-space:pre-wrap;word-break:break-word;margin:0;font:inherit}
details.ctx{margin-top:10px}
details.ctx summary{color:var(--accent);cursor:pointer}
.turn{border-left:2px solid var(--border);padding:4px 0 4px 10px;margin:6px 0}
.turn .role{font-size:10px;text-transform:uppercase;color:var(--muted)}
.turn .tools{font-size:10px;color:var(--accent);margin-left:6px}
.turn pre{white-space:pre-wrap;word-break:break-word;margin:2px 0 0;font:inherit;color:var(--muted)}
.turn-user pre{color:var(--fg)}
.turn-trigger{border-left-color:var(--accent)}
.turn-trigger .role::after{content:" \\2190 pushed back on";color:var(--accent)}
.why{color:var(--accent);font-style:italic;margin:0 0 6px}
.highlight{margin:12px 0}
"""

JS = """
const cards=[...document.querySelectorAll('#samples .card')];
const search=document.getElementById('search');
const count=document.getElementById('count');
const hideNoise=document.getElementById('hide-noise');
let kind='all';
function apply(){
  const q=search.value.trim().toLowerCase();
  let shown=0;
  for(const c of cards){
    const okKind=kind==='all'||c.dataset.kind===kind;
    const okNoise=!hideNoise.checked||c.dataset.noise!=='1';
    const okText=!q||c.textContent.toLowerCase().includes(q);
    const vis=okKind&&okNoise&&okText;
    c.style.display=vis?'':'none';
    if(vis)shown++;
  }
  count.textContent=shown+' / '+cards.length;
}
document.querySelectorAll('.kind-btn').forEach(b=>b.addEventListener('click',()=>{
  kind=b.dataset.kind;
  document.querySelectorAll('.kind-btn').forEach(x=>x.classList.toggle('active',x===b));
  apply();
}));
search.addEventListener('input',apply);
hideNoise.addEventListener('change',apply);
apply();
"""


@dataclass(frozen=True, slots=True)
class Sample:
    """One stored feedback event, decoded from a :meth:`FeedbackStore.events` row.

    Attributes:
        id: The event's database id.
        source_kind: Which detector produced it.
        occurred_at: The ISO timestamp of the feedback.
        text: The verbatim pushback text.
        payload: The detector-specific metadata, decoded from ``payload_json``.
        context: The conversational window around the feedback.
        origin_path: The transcript file the event came from.
        session_id: The session the event came from.
    """

    id: int
    source_kind: str
    occurred_at: str
    text: str
    payload: Mapping[str, Any]
    context: ContextSnapshot
    origin_path: str | None
    session_id: str | None

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> Sample:
        """Decodes a :meth:`FeedbackStore.events` row into a :class:`Sample`."""
        return cls(
            id=int(str(row["id"])),
            source_kind=str(row["source_kind"]),
            occurred_at=str(row["occurred_at"]),
            text=str(row["text"]),
            payload=json.loads(str(row["payload_json"])) if row["payload_json"] else {},
            context=ContextSnapshot.from_json(str(row["context_json"])),
            origin_path=str(row["origin_path"]) if row["origin_path"] else None,
            session_id=str(row["session_id"]) if row["session_id"] else None,
        )


@dataclass(frozen=True, slots=True)
class CorpusStats:
    """Aggregate counts describing the whole corpus.

    Attributes:
        total: The total number of samples.
        by_kind: Sample counts keyed by source kind, most common first.
        noise: The number of low-signal samples (bare interrupt markers, hook
            errors, and near-empty messages).
        sessions: The number of distinct sessions.
        projects: The number of distinct originating projects.
        first: The earliest sample date (``YYYY-MM-DD``).
        last: The latest sample date (``YYYY-MM-DD``).
        by_month: Sample counts keyed by ``YYYY-MM``, in chronological order.
    """

    total: int
    by_kind: Mapping[str, int]
    noise: int
    sessions: int
    projects: int
    first: str
    last: str
    by_month: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class Highlight:
    """A standout sample chosen for the summary, with an optional rationale.

    Attributes:
        event_id: The id of the highlighted sample.
        why: A short clause on why it is representative, when one was written.
    """

    event_id: int
    why: str | None = None


@dataclass(frozen=True, slots=True)
class Summary:
    """The corpus overview rendered above the sample list.

    Attributes:
        stats: The aggregate corpus counts.
        highlights: The standout samples chosen for the summary.
        narrative: A prose description of the developer's pushback style, when the
            ``claude`` CLI produced one.
    """

    stats: CorpusStats
    highlights: tuple[Highlight, ...]
    narrative: str | None


def is_noise(text: str) -> bool:
    return len(stripped := text.strip()) < 10 or stripped.startswith(NOISE_PREFIXES)


def project_label(origin_path: str) -> str:
    name = Path(origin_path).parent.name
    return next(
        (name.rsplit(marker, 1)[-1] for marker in ("-Code-", "-projects-", "-worktrees-") if marker in name),
        name.lstrip("-"),
    )


def corpus_stats(samples: Sequence[Sample]) -> CorpusStats:
    times = sorted(s.occurred_at for s in samples)
    return CorpusStats(
        total=len(samples),
        by_kind=dict(Counter(s.source_kind for s in samples).most_common()),
        noise=sum(is_noise(s.text) for s in samples),
        sessions=len({s.session_id for s in samples if s.session_id}),
        projects=len({Path(s.origin_path).parent.name for s in samples if s.origin_path}),
        first=times[0][:10] if times else "",
        last=times[-1][:10] if times else "",
        by_month=dict(sorted(Counter(s.occurred_at[:7] for s in samples).items())),
    )


def candidate_pool(samples: Sequence[Sample]) -> dict[str, list[Sample]]:
    pool: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if not is_noise(sample.text):
            pool[sample.source_kind].append(sample)
    return {
        kind: sorted(items, key=lambda s: len(s.text), reverse=True)[:HIGHLIGHT_POOL_PER_KIND]
        for kind, items in pool.items()
    }


def heuristic_highlight_ids(pool: Mapping[str, Sequence[Sample]]) -> list[int]:
    rows = [s for group in zip_longest(*pool.values()) for s in group if s is not None]
    return [s.id for s in rows[:HEURISTIC_HIGHLIGHTS]]


def summary_prompt(pool: Mapping[str, Sequence[Sample]], stats: CorpusStats) -> str:
    return "\n".join(
        [
            f"Corpus: {stats.total} samples across {stats.sessions} sessions, {stats.first} to {stats.last}.",
            "By kind: " + ", ".join(f"{kind}={n}" for kind, n in stats.by_kind.items()),
            "",
            "Feedback samples (id, kind, text):",
            *(
                f"[{s.id}] ({kind}) {' '.join(s.text.split())[:SAMPLE_TEXT_LIMIT]}"
                for kind, group in pool.items()
                for s in group
            ),
        ]
    )


def parse_summary_json(raw: str) -> tuple[str, list[dict[str, Any]]] | None:
    if not (match := re.search(r"\{.*\}", raw, re.DOTALL)):
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    narrative, picks = data.get("narrative"), data.get("highlights")
    if not isinstance(narrative, str) or not isinstance(picks, list):
        return None
    return narrative, [p for p in picks if isinstance(p, dict) and isinstance(p.get("id"), int)]


def llm_summary(
    pool: Mapping[str, Sequence[Sample]], stats: CorpusStats, model: str
) -> tuple[str, tuple[Highlight, ...]] | None:
    try:
        raw = run_claude(summary_prompt(pool, stats), system=SUMMARY_SYSTEM, model=model)
    except subprocess.SubprocessError:
        return None
    if (parsed := parse_summary_json(raw)) is None:
        return None
    narrative, picks = parsed
    valid = {s.id for group in pool.values() for s in group}
    highlights = tuple(Highlight(pick["id"], pick.get("why")) for pick in picks if pick["id"] in valid)
    return (narrative, highlights) if highlights else None


def build_summary(samples: Sequence[Sample], *, use_llm: bool, model: str) -> Summary:
    """Builds the corpus :class:`Summary`, using the ``claude`` CLI when allowed.

    When ``use_llm`` is set and ``claude`` is on the path, the narrative and
    highlights come from the model; on any failure to produce or parse a result the
    summary falls back to deterministic heuristics, so the export never depends on
    the model succeeding.

    Args:
        samples: The full corpus to summarize.
        use_llm: Whether to consult the ``claude`` CLI for the narrative.
        model: The model to run when consulting ``claude``.

    Returns:
        The assembled :class:`Summary`.
    """
    stats, pool = corpus_stats(samples), candidate_pool(samples)
    if use_llm and claude_available() and (result := llm_summary(pool, stats, model)) is not None:
        return Summary(stats=stats, highlights=result[1], narrative=result[0])
    return Summary(stats=stats, highlights=tuple(map(Highlight, heuristic_highlight_ids(pool))), narrative=None)


def truncate(text: str, limit: int = CONTEXT_TURN_LIMIT) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def render_turn(turn: ContextTurn, *, is_trigger: bool = False) -> str:
    cls = f"turn turn-{turn.role}" + (" turn-trigger" if is_trigger else "")
    tools = f'<span class="tools">{escape(" ".join(turn.tool_calls))}</span>' if turn.tool_calls else ""
    return (
        f'<div class="{cls}"><span class="role">{escape(turn.role)}</span>{tools}'
        f"<pre>{escape(truncate(turn.text))}</pre></div>"
    )


def render_context(ctx: ContextSnapshot) -> str:
    turns = [render_turn(turn, is_trigger=turn == ctx.trigger) for turn in ctx.before]
    if ctx.trigger is not None and ctx.trigger not in ctx.before:
        turns.append(render_turn(ctx.trigger, is_trigger=True))
    turns.extend(render_turn(turn) for turn in ctx.after)
    if not turns:
        return ""
    return f'<details class="ctx"><summary>context ({len(turns)} turns)</summary>{"".join(turns)}</details>'


def meta_chips(sample: Sample) -> str:
    payload = sample.payload
    chips = [str(payload[key]) for key in ("detector", "format", "tool", "severity", "track") if payload.get(key)]
    if file := payload.get("file"):
        line = payload.get("line_start") or payload.get("line")
        chips.append(f"{file}:{line}" if line else str(file))
    if sample.origin_path:
        chips.append(project_label(sample.origin_path))
    return "".join(f'<span class="chip">{escape(chip)}</span>' for chip in chips)


def render_card(sample: Sample) -> str:
    return "".join(
        [
            f'<article class="card" data-kind="{escape(sample.source_kind)}" '
            f'data-noise="{"1" if is_noise(sample.text) else "0"}">',
            f'<header><span class="badge badge-{escape(sample.source_kind)}">{escape(sample.source_kind)}</span>',
            f"<time>{escape(sample.occurred_at[:19])}</time>{meta_chips(sample)}</header>",
            f'<div class="text"><pre>{escape(sample.text)}</pre></div>',
            render_context(sample.context),
            "</article>",
        ]
    )


def render_highlight(sample: Sample, why: str | None) -> str:
    blurb = f'<p class="why">{escape(why)}</p>' if why else ""
    return f'<div class="highlight">{blurb}{render_card(sample)}</div>'


def render_stat_cards(stats: CorpusStats) -> str:
    cards = (
        (stats.total, "samples"),
        (stats.sessions, "sessions"),
        (stats.projects, "projects"),
        (stats.noise, "low-signal"),
        (f"{stats.first} – {stats.last}", "span"),
    )
    return '<div class="stat-cards">' + "".join(
        f'<div class="stat"><div class="n">{escape(str(value))}</div><div class="l">{escape(label)}</div></div>'
        for value, label in cards
    ) + "</div>"


def render_dist(stats: CorpusStats) -> str:
    top = max(stats.by_kind.values(), default=1)
    rows = "".join(
        f"<tr><td>{escape(kind)}</td><td>{n}</td>"
        f'<td><span class="bar" style="width:{round(n / top * 200)}px"></span></td></tr>'
        for kind, n in stats.by_kind.items()
    )
    return f'<table class="dist">{rows}</table>'


def render_months(by_month: Mapping[str, int]) -> str:
    if not by_month:
        return ""
    top = max(by_month.values())
    cols = "".join(
        f'<div class="mcol"><div class="m" style="height:{round(n / top * 72) + 4}px" '
        f'title="{escape(month)}: {n}"></div><span>{escape(month[5:])}</span></div>'
        for month, n in by_month.items()
    )
    return f'<div class="months">{cols}</div>'


def render_controls(stats: CorpusStats) -> str:
    buttons = "".join(
        f'<button class="kind-btn{" active" if kind == "all" else ""}" data-kind="{escape(kind)}">'
        f'{escape(kind)}{"" if kind == "all" else f" {n}"}</button>'
        for kind, n in [("all", stats.total), *stats.by_kind.items()]
    )
    return (
        f'<section id="controls"><div class="kinds">{buttons}</div>'
        f'<input id="search" type="search" placeholder="search text…">'
        f'<label class="noise"><input type="checkbox" id="hide-noise"> hide low-signal</label>'
        f'<span id="count">{stats.total} / {stats.total}</span></section>'
    )


def render_html(samples: Sequence[Sample], summary: Summary) -> str:
    """Renders the whole corpus and its summary into one self-contained HTML page.

    The returned string embeds its own CSS and JavaScript and references no external
    resources, so it can be written to a file and opened directly in a browser.

    Args:
        samples: Every sample to list, in display order.
        summary: The overview to render above the list.

    Returns:
        The complete HTML document.
    """
    by_id = {sample.id: sample for sample in samples}
    highlights = "\n".join(
        render_highlight(by_id[h.event_id], h.why) for h in summary.highlights if h.event_id in by_id
    )
    narrative = f'<div class="narrative">{escape(summary.narrative)}</div>' if summary.narrative else ""
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return "".join(
        [
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>cc-pushback samples</title><style>",
            CSS,
            "</style></head><body>",
            f'<header class="top"><h1>cc-pushback — feedback samples</h1>'
            f'<div class="sub">{summary.stats.total} samples · generated {escape(generated)}</div></header>',
            "<section><h2>Summary</h2>",
            render_stat_cards(summary.stats),
            render_dist(summary.stats),
            render_months(summary.stats.by_month),
            narrative,
            "</section>",
            '<section id="highlights"><h2>Highlights</h2>',
            highlights or "<p>none</p>",
            "</section>",
            render_controls(summary.stats),
            '<section id="samples">',
            "\n".join(render_card(sample) for sample in samples),
            "</section><script>",
            JS,
            "</script></body></html>",
        ]
    )
