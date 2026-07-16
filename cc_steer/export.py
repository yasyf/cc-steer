"""Stage 5 of the pipeline: export the steering lineage as a HuggingFace dataset.

The enrich stage completes the lineage — detector hit, judge verdict, refined
pairs, and code evidence in the shared ``corrections`` ledger. This stage reads
both stores (never writing to either) and materializes one canonical ``traces``
config — one row per judged event — plus three TRL-ready projections: ``sft``
(context + agent action → the user's verbatim steering), ``dpo`` (correcting
edit preferred over faulted edit), and ``kto`` (context + action → would the
user steer). Every config lands as per-split parquet under the output
directory next to a generated dataset card, and optionally pushes to a private
HuggingFace repo.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from cc_transcript.context import ContextWindow
from cc_transcript.corrections import CorrectionLog
from cc_transcript.ids import EventUuid, SessionId
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi

from cc_steer.enrich import SOURCE
from cc_steer.refine import PROMPT_VERSION as REFINE_VERSION
from cc_steer.rendering import (
    NO_STEER,
    Message,
    agent_action_of,
    ask_block,
    assistant_message,
    gate_text,
    has_substantive_content,
    messages,
    render_edit,
    split_of,
    structural_ask_messages,
    watcher_prompt,
)
from cc_steer.report import project_label
from cc_steer.triage import AUDIT_VERSION, PROMPT_VERSION, STEERING_CATEGORIES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from aiosqlite import Row
    from cc_transcript.corrections import Correction

    from cc_steer.store import FeedbackStore

__all__ = ["EmptyWatcherPrompt", "ExportReport", "export"]

SPLITS = ("train", "test")
REVIEW_META_KEYS = ("file", "line_start", "line_end", "format")
# The assistant-authored half of a question_answer payload — the ask the
# watcher may be conditioned on. The user's pick (picked_labels/option_pick)
# is the label and must never join model input.
ASK_META_KEYS = ("question", "header", "recommended_pick")
DPO_SIDES = ("faulted_old", "faulted_new", "correcting_old", "correcting_new")

TRACES_QUERY = """
WITH judge AS (
  SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'judge' AND t.prompt_version = ?
),
auditor AS (
  SELECT t.dedup_key, t.category, ROW_NUMBER() OVER (
    PARTITION BY t.dedup_key ORDER BY t.judged_at DESC, t.id DESC
  ) AS rn
  FROM triage t
  WHERE t.role = 'auditor' AND t.prompt_version = ?
)
SELECT e.dedup_key, e.source_kind, e.session_id, e.event_uuid, e.occurred_at, e.text,
  e.payload_json, e.context_json, e.cc_version, e.origin_path,
  j.category, j.is_steering, j.what_claude_did, j.confidence, j.rationale AS judge_rationale,
  j.model AS judge_model, j.fidelity,
  a.category AS auditor_category
FROM feedback_events e
JOIN judge j ON j.dedup_key = e.dedup_key AND j.rn = 1
LEFT JOIN auditor a ON a.dedup_key = e.dedup_key AND a.rn = 1
WHERE e.quarantined_reason IS NULL
ORDER BY e.id
"""

LATEST_PAIRS_QUERY = """
SELECT dedup_key, pair_index, action, direction_verbatim, direction
FROM latest_refinement
ORDER BY dedup_key, pair_index
"""

GATE_SAMPLES_QUERY = """
SELECT g.sample_key, g.kind, g.offset_turns, g.session_id, g.window_json,
  COALESCE(e.source_kind, '') AS source_kind, COALESCE(j.category, '') AS category
FROM gate_sample g
LEFT JOIN feedback_events e ON e.dedup_key = g.dedup_key
LEFT JOIN latest_judge j ON j.dedup_key = g.dedup_key
WHERE g.dedup_key IS NULL OR e.quarantined_reason IS NULL
ORDER BY g.id
"""

CATEGORIES = {
    "wrong_approach": "rejects the assistant's plan, strategy, or design",
    "incorrect_change": "the code or content the assistant produced is wrong or broken",
    "unwanted_action": "the assistant did something the user did not ask for or want",
    "style_violation": "the work violates the user's conventions or stated preferences",
    "premature": "the assistant stopped early, skipped work, or claimed completion when work remains",
    "direction": "a forward instruction or answer that resolves a decision the assistant faced or raised",
    "operational_directive": "routine logistics that faults nothing and resolves no choice the assistant raised",
    "status_update": "the user reporting state",
    "new_task": "a fresh request or spec, not a reaction to the preceding action",
    "question": "a genuine request for information",
    "other": "none of the above",
}

CARD_CONFIG_YAML = """- config_name: {config}{default}
  data_files:
  - split: train
    path: {config}/train*.parquet
  - split: test
    path: {config}/test*.parquet"""

CARD_TEMPLATE = """\
---
configs:
{configs_yaml}
size_categories:
- 1K<n<10K
tags:
- preference
- dpo
- kto
- sft
- code
- human-feedback
---

# cc-steer traces

One developer's steering of a coding agent: every judged moment the user shaped what
Claude Code did — correcting it, redirecting it, or resolving a choice it raised — mined
from their own transcripts by
[cc-steer](https://github.com/yasyf/cc-steer), judged steering-vs-noise by an
LLM triage judge, refined into atomic {{action, direction}} pairs, and grounded in the
code edits the corrective ones fault. Built to train models that steer a coding agent
the way this user does.

## Configs

| Config | Rows (train/test) | Intended use |
|---|---|---|
{config_table}

`traces` is the canonical superset; the other configs are deterministic projections
carrying exactly their TRL column contract plus `id` and `category`, where `id` joins
every derived row back to its `traces` parent.

## Categories

The judge classifies every event into one of eleven categories; the first six are
steering, the rest noise.

{category_list}

## Splits

`train`/`test` is a group split on the session: an event lands in `test` iff
`int(sha256(session_id), 16) % 10 == 0`. Split membership is computed once on
`traces` and inherited by every derived config through `id`, so no session's rows
straddle the split. `traces` lands at {train_count} train / {test_count} test.

## Class balance

{steering_count} steering vs {noise_count} noise{steering_share} at
judge v{judge_version}. The natural imbalance is preserved; balance at train time
(e.g. KTO's `desirable_weight`), not in the data. `sft` keeps only the steering rows;
`dpo` keeps one row per fully-grounded ledger correction, deduplicated across
dual-detected sibling events (the steering-judged parent wins); `kto` keeps everything —
`label = True` means the user would NOT have steered the action.

## Privacy

A single consenting user's own Claude Code sessions. Session ids, project labels,
and message text are raw and unredacted, so the repo stays private.

## Provenance

detector → judge v{judge_version} (auditor v{audit_version}) → refine
v{refine_version} → enrich (code evidence from the shared cc-transcript
`corrections` ledger, joined by steering anchor). Context turns are the previews
captured at mining time; transcripts are never re-hydrated at export.
"""

CONFIG_USES = {
    "traces": "The canonical superset: context, verdicts, refined pairs, code evidence, and the aftermath.",
    "sft": "TRL conversational prompt-completion: context + agent action → the user's verbatim steering.",
    "dpo": "TRL explicit-prompt preference: the correcting edit (`chosen`) over the faulted edit (`rejected`).",
    "kto": "TRL unpaired preference over every judged event; the only view that uses the noise negatives.",
    "gate": "Turn-level steer/no-steer classification text for the always-on gate, with rewound positive windows.",
    "watcher": "Context + agent action → steering direction or the `NO_STEER` sentinel, for the generative watcher.",
}

MINED_CONFIDENCE = 1.0
LIVE_SOURCE = "live_reaction"
LIVE_EMPTY_WINDOW_REASON = "live_window_render_empty"

# reaction kind -> (label bucket, fire label, confidence); expired carries no label.
LIVE_LABEL: dict[str, tuple[str, bool, float]] = {
    "accepted": ("pos", True, 0.9),
    "edited": ("pos-corrected", True, 0.7),
    "diverged": ("pos-corrected", True, 0.5),
    "dismissed": ("neg", False, 0.8),
    "ignored": ("weak-neg", False, 0.3),
}

RENDER_BLOCK = re.compile(r"(?:\A|\n\n)<(user|assistant)>\n")


class Pair(TypedDict):
    pair_index: int
    action: str
    direction_verbatim: str
    direction: str


class Evidence(TypedDict):
    digest: str | None
    file: str
    faulted_old: str
    faulted_new: str
    origin: str | None
    correction_file: str | None
    correcting_old: str | None
    correcting_new: str | None
    commit: str | None
    overlap: float


class Trace(TypedDict):
    id: str
    session_id: str
    event_uuid: str
    project: str
    occurred_at: str
    cc_version: str
    source_kind: str
    context: list[Message]
    agent_action: str | None
    what_claude_did: str
    user_message: str
    aftermath: list[Message]
    is_steering: bool
    category: str
    confidence: float
    judge_rationale: str
    judge_model: str
    fidelity: str
    auditor_category: str | None
    pairs: list[Pair]
    evidence: list[Evidence]
    split: str
    meta: str


class GateRow(TypedDict):
    id: str
    text: str
    label: bool
    kind: str
    offset_turns: int
    source_kind: str
    category: str
    session_id: str
    split: str
    label_confidence: float


class WatcherRow(TypedDict):
    prompt: list[Message]
    completion: list[Message]
    verbatim: str
    label: bool
    id: str
    category: str
    source_kind: str
    session_id: str
    split: str
    label_confidence: float


class EmptyWatcherPrompt(ValueError):
    """An exported row whose final prompt has no substantive content."""

    def __init__(self, *, dedup_key: str, session_id: str, source_kind: str, view: str = "watcher") -> None:
        self.dedup_key = dedup_key
        self.session_id = session_id
        self.source_kind = source_kind
        self.view = view
        super().__init__(
            f"empty {view} prompt: dedup_key={dedup_key} session_id={session_id} source_kind={source_kind}"
        )


class SftRow(TypedDict):
    prompt: list[Message]
    completion: list[Message]
    id: str
    category: str


class DpoRow(TypedDict):
    prompt: list[Message]
    chosen: list[Message]
    rejected: list[Message]
    id: str
    category: str


class KtoRow(TypedDict):
    prompt: list[Message]
    completion: list[Message]
    label: bool
    id: str
    category: str


@dataclass(frozen=True, slots=True)
class ExportReport:
    """The outcome of one export pass.

    Attributes:
        counts: Row counts keyed by config name, then split name.
        out: The directory the per-config parquet files and dataset card landed in.
        pushed: Whether the configs were pushed to the HuggingFace repo.
        quarantined: Excluded live-reaction counts keyed by quarantine reason.
    """

    counts: Mapping[str, Mapping[str, int]]
    out: Path
    pushed: bool
    quarantined: Mapping[str, int] = field(default_factory=dict)


def evidence_entry(correction: Correction) -> Evidence:
    return {
        "digest": correction.incorrect_digest,
        "file": correction.incorrect_file,
        "faulted_old": correction.incorrect_old,
        "faulted_new": correction.incorrect_new,
        "origin": correction.correction_origin,
        "correction_file": correction.correction_file,
        "correcting_old": correction.correction_old,
        "correcting_new": correction.correction_new,
        "commit": correction.correction_commit,
        "overlap": correction.overlap,
    }


def trace_meta(row: Row) -> str:
    payload = json.loads(str(row["payload_json"]))
    return json.dumps(
        {"signal": payload["signal"]}
        | {key: payload[key] for key in REVIEW_META_KEYS if key in payload}
        | {key: payload[key] for key in ASK_META_KEYS if key in payload}
        | {
            "prompt_version": PROMPT_VERSION,
            "audit_version": AUDIT_VERSION,
            "origin_path": row["origin_path"],
        }
    )


def trace_row(row: Row, pairs: list[Pair], log: CorrectionLog) -> Trace:
    window = ContextWindow.from_json(str(row["context_json"]))
    session_id = str(row["session_id"])
    is_steering = bool(row["is_steering"])
    return {
        "id": row["dedup_key"],
        "session_id": session_id,
        "event_uuid": row["event_uuid"],
        "project": project_label(str(row["origin_path"])),
        "occurred_at": row["occurred_at"],
        "cc_version": row["cc_version"],
        "source_kind": row["source_kind"],
        "context": messages(window.before),
        "agent_action": agent_action_of(window),
        "what_claude_did": row["what_claude_did"],
        "user_message": row["text"],
        "aftermath": messages([turn for turn in (window.trigger, *window.after) if turn is not None]),
        "is_steering": is_steering,
        "category": row["category"],
        "confidence": row["confidence"],
        "judge_rationale": row["judge_rationale"],
        "judge_model": row["judge_model"],
        "fidelity": row["fidelity"],
        "auditor_category": row["auditor_category"],
        "pairs": pairs if is_steering else [],
        "evidence": [
            evidence_entry(correction)
            for correction in log.for_anchor(SessionId(session_id), EventUuid(str(row["event_uuid"])))
            if correction.source == SOURCE
        ],
        "split": split_of(session_id),
        "meta": trace_meta(row),
    }


def validated_prompt(
    prompt: list[Message], *, view: str, dedup_key: str, session_id: str, source_kind: str
) -> list[Message]:
    if not has_substantive_content(prompt):
        raise EmptyWatcherPrompt(
            dedup_key=dedup_key,
            session_id=session_id,
            source_kind=source_kind,
            view=view,
        )
    return prompt


def sft_row(trace: Trace) -> SftRow:
    prompt = [*trace["context"], *assistant_message(trace["agent_action"] or trace["what_claude_did"])]
    return {
        "prompt": validated_prompt(
            prompt,
            view="sft",
            dedup_key=trace["id"],
            session_id=trace["session_id"],
            source_kind=trace["source_kind"],
        ),
        "completion": assistant_message(trace["user_message"]),
        "id": trace["id"],
        "category": trace["category"],
    }


def kto_row(trace: Trace) -> KtoRow:
    return {
        "prompt": validated_prompt(
            trace["context"],
            view="kto",
            dedup_key=trace["id"],
            session_id=trace["session_id"],
            source_kind=trace["source_kind"],
        ),
        "completion": assistant_message(trace["agent_action"] or trace["what_claude_did"]),
        "label": not trace["is_steering"],
        "id": trace["id"],
        "category": trace["category"],
    }


def dpo_row(trace: Trace, entry: Evidence) -> DpoRow:
    return {
        "prompt": validated_prompt(
            trace["context"],
            view="dpo",
            dedup_key=trace["id"],
            session_id=trace["session_id"],
            source_kind=trace["source_kind"],
        ),
        "chosen": assistant_message(
            render_edit(str(entry["correction_file"]), str(entry["correcting_old"]), str(entry["correcting_new"]))
        ),
        "rejected": assistant_message(render_edit(entry["file"], entry["faulted_old"], entry["faulted_new"])),
        "id": trace["id"],
        "category": trace["category"],
    }


def dpo_split(traces: Sequence[Trace]) -> list[DpoRow]:
    return list(
        {
            (trace["session_id"], trace["event_uuid"], entry["digest"]): dpo_row(trace, entry)
            for trace in sorted(traces, key=lambda trace: trace["is_steering"])
            for entry in trace["evidence"]
            if all(entry[side] is not None for side in DPO_SIDES)
        }.values()
    )


def gate_row(row: Row) -> GateRow | None:
    try:
        window = ContextWindow.from_json(str(row["window_json"]))
    except (ValueError, KeyError):
        return None
    validated_prompt(
        watcher_prompt(window),
        view="gate",
        dedup_key=str(row["sample_key"]),
        session_id=str(row["session_id"]),
        source_kind=str(row["source_kind"]),
    )
    return {
        "id": str(row["sample_key"]),
        "text": gate_text(window),
        "label": str(row["kind"]) == "positive_window",
        "kind": str(row["kind"]),
        "offset_turns": int(row["offset_turns"]),
        "source_kind": str(row["source_kind"]),
        "category": str(row["category"]),
        "session_id": str(row["session_id"]),
        "split": split_of(str(row["session_id"])),
        "label_confidence": MINED_CONFIDENCE,
    }


def ask_message_of(trace: Trace) -> Message | None:
    """The payload-derived structural ask for a question_answer trace, or None.

    Recovers the assistant's question for captures whose window never carried
    it (the empty-context defect: the ask lives only in the label-bearing
    trigger turn). Only assistant-authored fields render — question, header,
    recommended pick — never the user's answer.
    """
    if trace["source_kind"] != "question_answer":
        return None
    meta = json.loads(trace["meta"])
    question = str(meta.get("question") or "")
    if not question:
        return None
    recommended = meta.get("recommended_pick")
    block = ask_block(
        question,
        header=str(meta.get("header") or ""),
        recommended=str(recommended) if recommended else "",
    )
    return {"role": "assistant", "content": block}


def watcher_positive(trace: Trace) -> WatcherRow:
    direction = "\n".join(pair["direction"] for pair in trace["pairs"]) or trace["user_message"]
    prompt = structural_ask_messages(trace["context"])
    ask = ask_message_of(trace)
    if ask is not None and not any("[assistant asked" in message["content"] for message in prompt):
        prompt = [*prompt, ask]
    return {
        "prompt": validated_prompt(
            prompt,
            view="watcher",
            dedup_key=trace["id"],
            session_id=trace["session_id"],
            source_kind=trace["source_kind"],
        ),
        "completion": assistant_message(direction),
        "verbatim": trace["user_message"],
        "label": True,
        "id": trace["id"],
        "category": trace["category"],
        "source_kind": trace["source_kind"],
        "session_id": trace["session_id"],
        "split": trace["split"],
        "label_confidence": MINED_CONFIDENCE,
    }


def watcher_negative(row: Row) -> WatcherRow | None:
    try:
        window = ContextWindow.from_json(str(row["window_json"]))
    except (ValueError, KeyError):
        return None
    return {
        "prompt": watcher_prompt(window, render_version=2),
        "completion": assistant_message(NO_STEER),
        "verbatim": "",
        "label": False,
        "id": str(row["sample_key"]),
        "category": str(row["category"]),
        "source_kind": str(row["source_kind"]),
        "session_id": str(row["session_id"]),
        "split": split_of(str(row["session_id"])),
        "label_confidence": MINED_CONFIDENCE,
    }


def watcher_rows(traces: Sequence[Trace], gate_samples: Sequence[Row]) -> list[WatcherRow]:
    positives = [watcher_positive(trace) for trace in traces if trace["is_steering"]]
    negatives = [
        rendered
        for row in gate_samples
        if str(row["kind"]) != "positive_window" and int(row["offset_turns"]) == 0
        if (rendered := watcher_negative(row)) is not None
    ]
    return [*positives, *negatives]


def messages_from_render(render: str) -> list[Message]:
    """Reconstructs the watcher prompt from a proposal's flattened ``window_render``.

    The proposal ledger stores the cascade's exact flattened gate text, not a
    structured window; this inverts :func:`~cc_steer.rendering.gate_text` back into
    role-tagged chat messages so a live reaction trains on the moment the steer fired.
    """
    spans = list(RENDER_BLOCK.finditer(render))
    if not spans:
        return [{"role": "user", "content": render}] if render else []
    return [
        {
            "role": span.group(1),
            "content": render[span.end() : (spans[index + 1].start() if index + 1 < len(spans) else len(render))],
        }
        for index, span in enumerate(spans)
    ]


def live_render_has_substantive_content(reaction: Mapping[str, object]) -> bool:
    return has_substantive_content(messages_from_render(str(reaction["window_render"] or "")))


def live_completion(kind: str, steer: str, reply: str | None) -> str:
    """The training target: the steer if accepted, the user's reply if corrected, else the sentinel."""
    match kind:
        case "accepted":
            return steer
        case "edited" | "diverged":
            return reply or steer
        case _:
            return NO_STEER


def live_watcher_row(reaction: Mapping[str, object], reply_texts: Mapping[str, str]) -> WatcherRow | None:
    """One ``live_reaction`` watcher row: the proposal's moment labelled by how the user reacted."""
    if (entry := LIVE_LABEL.get(str(reaction["kind"]))) is None:
        return None
    _, label, confidence = entry
    session_id = str(reaction["session_id"])
    reply = reply_texts.get(str(reaction["feedback_dedup_key"] or ""))
    steer = str(reaction["steer"] or "")
    prompt = messages_from_render(str(reaction["window_render"] or ""))
    return {
        "prompt": validated_prompt(
            prompt,
            view="watcher",
            dedup_key=f"live:{reaction['proposal_id']}",
            session_id=session_id,
            source_kind=LIVE_SOURCE,
        ),
        "completion": assistant_message(live_completion(str(reaction["kind"]), steer, reply)),
        "verbatim": reply or steer,
        "label": label,
        "id": f"live:{reaction['proposal_id']}",
        "category": str(reaction["kind"]),
        "source_kind": LIVE_SOURCE,
        "session_id": session_id,
        "split": split_of(session_id),
        "label_confidence": confidence,
    }


def live_gate_row(reaction: Mapping[str, object]) -> GateRow | None:
    """One ``live_reaction`` gate row: the proposal's flattened window text with its fire label."""
    if (entry := LIVE_LABEL.get(str(reaction["kind"]))) is None:
        return None
    _, label, confidence = entry
    session_id = str(reaction["session_id"])
    render = str(reaction["window_render"] or "")
    validated_prompt(
        messages_from_render(render),
        view="gate",
        dedup_key=f"live:{reaction['proposal_id']}",
        session_id=session_id,
        source_kind=LIVE_SOURCE,
    )
    return {
        "id": f"live:{reaction['proposal_id']}",
        "text": render,
        "label": label,
        "kind": LIVE_SOURCE,
        "offset_turns": 0,
        "source_kind": LIVE_SOURCE,
        "category": str(reaction["kind"]),
        "session_id": session_id,
        "split": split_of(session_id),
        "label_confidence": confidence,
    }


def config_rows(
    traces: list[Trace],
    gate_samples: Sequence[Row] = (),
    *,
    live_watcher: Sequence[WatcherRow] = (),
    live_gate: Sequence[GateRow] = (),
) -> dict[str, Mapping[str, Sequence[Mapping[str, object]]]]:
    by_split = {split: [trace for trace in traces if trace["split"] == split] for split in SPLITS}
    gate = [*(row for row in (gate_row(sample) for sample in gate_samples) if row is not None), *live_gate]
    watcher = [*watcher_rows(traces, gate_samples), *live_watcher]
    return {
        "traces": by_split,
        "sft": {split: [sft_row(t) for t in ts if t["is_steering"]] for split, ts in by_split.items()},
        "dpo": {split: dpo_split(ts) for split, ts in by_split.items()},
        "kto": {split: [kto_row(t) for t in ts] for split, ts in by_split.items()},
        "gate": {split: [row for row in gate if row["split"] == split] for split in SPLITS},
        "watcher": {split: [row for row in watcher if row["split"] == split] for split in SPLITS},
    }


def config_features() -> dict[str, Features]:
    def message() -> list[dict[str, Value]]:
        return [{"role": Value("string"), "content": Value("string")}]

    keys = {"id": Value("string"), "category": Value("string")}
    return {
        "traces": Features(
            {
                "id": Value("string"),
                "session_id": Value("string"),
                "event_uuid": Value("string"),
                "project": Value("string"),
                "occurred_at": Value("string"),
                "cc_version": Value("string"),
                "source_kind": Value("string"),
                "context": message(),
                "agent_action": Value("string"),
                "what_claude_did": Value("string"),
                "user_message": Value("string"),
                "aftermath": message(),
                "is_steering": Value("bool"),
                "category": Value("string"),
                "confidence": Value("float64"),
                "judge_rationale": Value("string"),
                "judge_model": Value("string"),
                "fidelity": Value("string"),
                "auditor_category": Value("string"),
                "pairs": [
                    {
                        "pair_index": Value("int64"),
                        "action": Value("string"),
                        "direction_verbatim": Value("string"),
                        "direction": Value("string"),
                    }
                ],
                "evidence": [
                    {
                        "digest": Value("string"),
                        "file": Value("string"),
                        "faulted_old": Value("string"),
                        "faulted_new": Value("string"),
                        "origin": Value("string"),
                        "correction_file": Value("string"),
                        "correcting_old": Value("string"),
                        "correcting_new": Value("string"),
                        "commit": Value("string"),
                        "overlap": Value("float64"),
                    }
                ],
                "split": Value("string"),
                "meta": Value("string"),
            }
        ),
        "sft": Features({"prompt": message(), "completion": message()} | keys),
        "dpo": Features({"prompt": message(), "chosen": message(), "rejected": message()} | keys),
        "kto": Features({"prompt": message(), "completion": message(), "label": Value("bool")} | keys),
        "gate": Features(
            {
                "id": Value("string"),
                "text": Value("string"),
                "label": Value("bool"),
                "kind": Value("string"),
                "offset_turns": Value("int64"),
                "source_kind": Value("string"),
                "category": Value("string"),
                "session_id": Value("string"),
                "split": Value("string"),
                "label_confidence": Value("float64"),
            }
        ),
        "watcher": Features(
            {
                "prompt": message(),
                "completion": message(),
                "verbatim": Value("string"),
                "label": Value("bool"),
                "source_kind": Value("string"),
                "session_id": Value("string"),
                "split": Value("string"),
                "label_confidence": Value("float64"),
            }
            | keys
        ),
    }


def dataset_card(counts: Mapping[str, Mapping[str, int]], *, steering_count: int, noise_count: int) -> str:
    return CARD_TEMPLATE.format(
        configs_yaml="\n".join(
            CARD_CONFIG_YAML.format(config=config, default="\n  default: true" if config == "traces" else "")
            for config in counts
        ),
        config_table="\n".join(
            f"| `{config}`{' (default)' if config == 'traces' else ''} "
            f"| {splits['train']}/{splits['test']} | {CONFIG_USES[config]} |"
            for config, splits in counts.items()
        ),
        category_list="\n".join(
            f"- `{category}` — {definition}{' *(steering)*' if category in STEERING_CATEGORIES else ''}"
            for category, definition in CATEGORIES.items()
        ),
        train_count=counts["traces"]["train"],
        test_count=counts["traces"]["test"],
        steering_count=steering_count,
        noise_count=noise_count,
        steering_share=f" ({steering_count / judged:.0%} steering)" if (judged := steering_count + noise_count) else "",
        judge_version=PROMPT_VERSION,
        audit_version=AUDIT_VERSION,
        refine_version=REFINE_VERSION,
    )


async def load_traces(store: FeedbackStore) -> list[Trace]:
    log = CorrectionLog.open()
    pair_cur = await store.store.conn.execute(LATEST_PAIRS_QUERY)
    pairs_by_key: dict[str, list[Pair]] = {}
    async for pair in pair_cur:
        pairs_by_key.setdefault(str(pair["dedup_key"]), []).append(
            Pair(
                pair_index=pair["pair_index"],
                action=pair["action"],
                direction_verbatim=pair["direction_verbatim"],
                direction=pair["direction"],
            )
        )
    cur = await store.store.conn.execute(TRACES_QUERY, (PROMPT_VERSION, AUDIT_VERSION))
    return [trace_row(row, pairs_by_key.get(str(row["dedup_key"]), []), log) async for row in cur]


async def load_live_reactions(
    store: FeedbackStore, shadow_db: Path | None
) -> tuple[list[dict[str, object]], dict[str, str]]:
    """The attributed reactions and the reply texts their corrected labels complete."""
    from cc_steer.watcher.live import LiveConfig, MailboxDelivery, shadow_db_path

    async with await MailboxDelivery.open(shadow_db or shadow_db_path(), config=LiveConfig.shadow()) as mailbox:
        reactions = await mailbox.reactions()
    keys = [key for row in reactions if (key := str(row["feedback_dedup_key"] or ""))]
    if not keys:
        return reactions, {}
    cur = await store.store.conn.execute(
        f"SELECT dedup_key, text FROM feedback_events WHERE dedup_key IN ({','.join('?' for _ in keys)})", keys
    )
    return reactions, {str(row["dedup_key"]): str(row["text"]) async for row in cur}


async def export(
    store: FeedbackStore, *, out: Path, push_to: str | None = None, shadow_db: Path | None = None
) -> ExportReport:
    """Exports the judged corpus as a HuggingFace dataset: ``traces`` plus TRL views.

    Reads the feedback store, the shared ``corrections`` ledger, and the shadow
    ledger's attributed reactions (all read-only) and builds one row per judged
    event at the current judge prompt version — the canonical ``traces`` config —
    then projects the TRL-ready ``sft``, ``dpo``, and ``kto`` configs from the same
    rows. Delivered-steer reactions add ``live_reaction`` rows to ``watcher`` and
    ``gate``, each carrying a ``label_confidence`` (mined rows carry ``1.0``). Labelled
    reactions without substantive ``window_render`` content are excluded and counted
    by quarantine reason in the report. Every config is written as per-split parquet
    under ``out/<config>/<split>.parquet`` next to a generated dataset card at
    ``out/README.md``; with ``push_to``, every config is also pushed to that private
    HuggingFace repo and the card uploaded. Splits are a deterministic group split
    on the session hash, computed once on ``traces`` and inherited by every derived
    row.

    Args:
        store: The open feedback store.
        out: The directory to write the parquet files and dataset card into.
        push_to: The HuggingFace dataset repo to push every config to as a
            private dataset; None skips the push.
        shadow_db: The shadow ledger to read reactions from; None uses the default.

    Returns:
        The export report with per-config row counts and live-reaction quarantines.
    """
    traces = await load_traces(store)
    gate_cur = await store.store.conn.execute(GATE_SAMPLES_QUERY)
    gate_samples = [row async for row in gate_cur]
    reactions, reply_texts = await load_live_reactions(store, shadow_db)
    labelled_reactions = [reaction for reaction in reactions if str(reaction["kind"]) in LIVE_LABEL]
    live_reactions = [reaction for reaction in labelled_reactions if live_render_has_substantive_content(reaction)]
    features = config_features()
    by_config = config_rows(
        traces,
        gate_samples,
        live_watcher=[
            row
            for reaction in live_reactions
            if (row := live_watcher_row(reaction, reply_texts)) is not None
        ],
        live_gate=[row for reaction in live_reactions if (row := live_gate_row(reaction)) is not None],
    )
    built = {
        config: DatasetDict(
            {
                split: Dataset.from_dict(
                    {name: [row[name] for row in rows] for name in features[config]}, features=features[config]
                )
                for split, rows in splits.items()
            }
        )
        for config, splits in by_config.items()
    }
    counts = {config: {split: len(rows) for split, rows in splits.items()} for config, splits in by_config.items()}
    for config, splits in built.items():
        (out / config).mkdir(parents=True, exist_ok=True)
        for split, dataset in splits.items():
            dataset.to_parquet(out / config / f"{split}.parquet")
    card = out / "README.md"
    card.write_text(
        dataset_card(
            counts,
            steering_count=sum(trace["is_steering"] for trace in traces),
            noise_count=sum(not trace["is_steering"] for trace in traces),
        )
    )
    if push_to is not None:
        for config, splits in built.items():
            splits.push_to_hub(push_to, config_name=config, private=True)
        HfApi().upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=push_to, repo_type="dataset")
    return ExportReport(
        counts=counts,
        out=out,
        pushed=push_to is not None,
        quarantined={LIVE_EMPTY_WINDOW_REASON: len(labelled_reactions) - len(live_reactions)},
    )
