"""Stage 5 of the pipeline: export the pushback lineage as a HuggingFace dataset.

The enrich stage completes the lineage — detector hit, judge verdict, refined
pairs, and code evidence in the shared ``corrections`` ledger. This stage reads
both stores (never writing to either) and materializes one canonical ``traces``
config — one row per judged event — plus three TRL-ready projections: ``sft``
(context + agent action → the user's verbatim pushback), ``dpo`` (correcting
edit preferred over faulted edit), and ``kto`` (context + action → would the
user push back). Every config lands as per-split parquet under the output
directory next to a generated dataset card, and optionally pushes to a private
HuggingFace repo.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cc_transcript.context import ContextWindow
from cc_transcript.corrections import CorrectionLog
from cc_transcript.ids import EventUuid, SessionId

from cc_pushback.refine import PROMPT_VERSION as REFINE_VERSION
from cc_pushback.report import project_label
from cc_pushback.triage import AUDIT_VERSION, PROMPT_VERSION, PUSHBACK_CATEGORIES

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cc_transcript.context import TurnRef
    from cc_transcript.corrections import Correction
    from datasets import Features

    from cc_pushback.store import FeedbackStore

__all__ = ["ExportReport", "export"]

SOURCE = "cc-pushback"
SPLITS = ("train", "test")
REVIEW_META_KEYS = ("file", "line_start", "line_end", "format")
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
  j.category, j.is_pushback, j.what_claude_did, j.confidence, j.rationale AS judge_rationale,
  j.model AS judge_model, j.fidelity,
  a.category AS auditor_category
FROM feedback_events e
JOIN judge j ON j.dedup_key = e.dedup_key AND j.rn = 1
LEFT JOIN auditor a ON a.dedup_key = e.dedup_key AND a.rn = 1
ORDER BY e.id
"""

LATEST_PAIRS_QUERY = """
WITH gens AS (
  SELECT dedup_key, prompt_version, model, refined_at,
    ROW_NUMBER() OVER (
      PARTITION BY dedup_key ORDER BY prompt_version DESC, refined_at DESC
    ) AS g
  FROM (SELECT DISTINCT dedup_key, prompt_version, model, refined_at FROM refinement)
)
SELECT r.dedup_key, r.pair_index, r.action, r.complaint_verbatim, r.complaint
FROM refinement r
JOIN gens ON gens.dedup_key = r.dedup_key AND gens.prompt_version = r.prompt_version
         AND gens.model = r.model AND gens.refined_at = r.refined_at AND gens.g = 1
ORDER BY r.dedup_key, r.pair_index
"""

CATEGORIES = {
    "wrong_approach": "rejects the assistant's plan, strategy, or design",
    "incorrect_change": "the code or content the assistant produced is wrong or broken",
    "unwanted_action": "the assistant did something the user did not ask for or want",
    "style_violation": "the work violates the user's conventions or stated preferences",
    "premature": "the assistant stopped early, skipped work, or claimed completion when work remains",
    "operational_directive": "a forward instruction that does not criticize prior work",
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

# cc-pushback traces

One developer's pushback on a coding agent: every judged moment the user corrected
Claude Code, mined from their own transcripts by
[cc-pushback](https://github.com/yasyf/cc-pushback), judged pushback-vs-noise by an
LLM triage judge, refined into atomic {{action, complaint}} pairs, and grounded in
the code edits the complaints fault. Built to train models that push back on — and
steer — a coding agent the way this user does.

## Configs

| Config | Rows (train/test) | Intended use |
|---|---|---|
{config_table}

`traces` is the canonical superset; the other configs are deterministic projections
carrying exactly their TRL column contract plus `id` and `category`, where `id` joins
every derived row back to its `traces` parent.

## Categories

The judge classifies every event into one of ten categories; the first five are
pushback, the rest noise.

{category_list}

## Splits

`train`/`test` is a group split on the session: an event lands in `test` iff
`int(sha256(session_id), 16) % 10 == 0`. Split membership is computed once on
`traces` and inherited by every derived config through `id`, so no session's rows
straddle the split. `traces` lands at {train_count} train / {test_count} test.

## Class balance

{pushback_count} pushback vs {noise_count} noise ({pushback_share:.0%} pushback) at
judge v{judge_version}. The natural imbalance is preserved; balance at train time
(e.g. KTO's `desirable_weight`), not in the data. `sft` keeps only the pushback rows;
`dpo` keeps one row per fully-grounded ledger correction, deduplicated across
dual-detected sibling events (the pushback-judged parent wins); `kto` keeps everything —
`label = True` means the user would NOT have pushed back on the action.

## Privacy

A single consenting user's own Claude Code sessions. Session ids, project labels,
and message text are raw and unredacted, so the repo stays private.

## Provenance

detector → judge v{judge_version} (auditor v{audit_version}) → refine
v{refine_version} → enrich (code evidence from the shared cc-transcript
`corrections` ledger, joined by pushback anchor). Context turns are the previews
captured at mining time; transcripts are never re-hydrated at export.
"""

CONFIG_USES = {
    "traces": "The canonical superset: context, verdicts, refined pairs, code evidence, and the aftermath.",
    "sft": "TRL conversational prompt-completion: context + agent action → the user's verbatim pushback.",
    "dpo": "TRL explicit-prompt preference: the correcting edit (`chosen`) over the faulted edit (`rejected`).",
    "kto": "TRL unpaired preference over every judged event; the only view that uses the noise negatives.",
}


@dataclass(frozen=True, slots=True)
class ExportReport:
    """The outcome of one export pass.

    Attributes:
        counts: Row counts keyed by config name, then split name.
        out: The directory the per-config parquet files and dataset card landed in.
        pushed: Whether the configs were pushed to the HuggingFace repo.
    """

    counts: Mapping[str, Mapping[str, int]]
    out: Path
    pushed: bool


def split_of(session_id: str) -> str:
    return "test" if int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 10 == 0 else "train"


def messages(turns: Sequence[TurnRef]) -> list[dict[str, str]]:
    return [{"role": turn.role, "content": turn.preview} for turn in turns]


def assistant_message(content: str) -> list[dict[str, str]]:
    return [{"role": "assistant", "content": content}]


def agent_action_of(window: ContextWindow) -> str | None:
    return next((turn.preview for turn in reversed(window.before) if turn.role == "assistant"), None)


def render_edit(file: str, old: str, new: str) -> str:
    return f"{file}\n```old\n{old}\n```\n```new\n{new}\n```"


def evidence_entry(correction: Correction) -> dict[str, object]:
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


def meta_of(row: Mapping[str, object]) -> str:
    payload = json.loads(str(row["payload_json"]))
    return json.dumps(
        {"signal": payload["signal"]}
        | {key: payload[key] for key in REVIEW_META_KEYS if key in payload}
        | {
            "prompt_version": PROMPT_VERSION,
            "audit_version": AUDIT_VERSION,
            "origin_path": row["origin_path"],
        }
    )


def trace_row(row: Mapping[str, object], pairs: list[dict[str, object]], log: CorrectionLog) -> dict[str, object]:
    window = ContextWindow.from_json(str(row["context_json"]))
    session_id = str(row["session_id"])
    is_pushback = bool(row["is_pushback"])
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
        "is_pushback": is_pushback,
        "category": row["category"],
        "confidence": row["confidence"],
        "judge_rationale": row["judge_rationale"],
        "judge_model": row["judge_model"],
        "fidelity": row["fidelity"],
        "auditor_category": row["auditor_category"],
        "pairs": pairs if is_pushback else [],
        "evidence": [
            evidence_entry(correction)
            for correction in log.for_anchor(SessionId(session_id), EventUuid(str(row["event_uuid"])))
            if correction.source == SOURCE
        ],
        "split": split_of(session_id),
        "meta": meta_of(row),
    }


def sft_row(trace: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": [*trace["context"], *assistant_message(str(trace["agent_action"] or trace["what_claude_did"]))],
        "completion": assistant_message(str(trace["user_message"])),
        "id": trace["id"],
        "category": trace["category"],
    }


def kto_row(trace: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": trace["context"],
        "completion": assistant_message(str(trace["agent_action"] or trace["what_claude_did"])),
        "label": not trace["is_pushback"],
        "id": trace["id"],
        "category": trace["category"],
    }


def dpo_row(trace: Mapping[str, object], entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "prompt": trace["context"],
        "chosen": assistant_message(
            render_edit(str(entry["correction_file"]), str(entry["correcting_old"]), str(entry["correcting_new"]))
        ),
        "rejected": assistant_message(
            render_edit(str(entry["file"]), str(entry["faulted_old"]), str(entry["faulted_new"]))
        ),
        "id": trace["id"],
        "category": trace["category"],
    }


def dpo_split(traces: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return list(
        {
            (trace["session_id"], trace["event_uuid"], entry["digest"]): dpo_row(trace, entry)
            for trace in sorted(traces, key=lambda trace: bool(trace["is_pushback"]))
            for entry in trace["evidence"]
            if all(entry[side] is not None for side in DPO_SIDES)
        }.values()
    )


def config_rows(traces: list[dict[str, object]]) -> dict[str, dict[str, list[dict[str, object]]]]:
    by_split = {split: [trace for trace in traces if trace["split"] == split] for split in SPLITS}
    return {
        "traces": by_split,
        "sft": {split: [sft_row(t) for t in ts if t["is_pushback"]] for split, ts in by_split.items()},
        "dpo": {split: dpo_split(ts) for split, ts in by_split.items()},
        "kto": {split: [kto_row(t) for t in ts] for split, ts in by_split.items()},
    }


def config_features() -> dict[str, Features]:
    from datasets import Features, Value

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
                "is_pushback": Value("bool"),
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
                        "complaint_verbatim": Value("string"),
                        "complaint": Value("string"),
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
    }


def dataset_card(counts: Mapping[str, Mapping[str, int]], *, pushback_count: int, noise_count: int) -> str:
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
            f"- `{category}` — {definition}{' *(pushback)*' if category in PUSHBACK_CATEGORIES else ''}"
            for category, definition in CATEGORIES.items()
        ),
        train_count=counts["traces"]["train"],
        test_count=counts["traces"]["test"],
        pushback_count=pushback_count,
        noise_count=noise_count,
        pushback_share=pushback_count / (pushback_count + noise_count),
        judge_version=PROMPT_VERSION,
        audit_version=AUDIT_VERSION,
        refine_version=REFINE_VERSION,
    )


async def load_traces(store: FeedbackStore) -> list[dict[str, object]]:
    log = CorrectionLog.open()
    pair_cur = await store.store.conn.execute(LATEST_PAIRS_QUERY)
    pairs_by_key: dict[str, list[dict[str, object]]] = {}
    async for pair in pair_cur:
        pairs_by_key.setdefault(str(pair["dedup_key"]), []).append(
            {key: pair[key] for key in ("pair_index", "action", "complaint_verbatim", "complaint")}
        )
    cur = await store.store.conn.execute(TRACES_QUERY, (PROMPT_VERSION, AUDIT_VERSION))
    return [trace_row(row, pairs_by_key.get(str(row["dedup_key"]), []), log) async for row in cur]


async def export(
    store: FeedbackStore, *, out: Path, repo_id: str = "yasyf/cc-pushback-traces", push: bool = False
) -> ExportReport:
    """Exports the judged corpus as a HuggingFace dataset: ``traces`` plus TRL views.

    Reads the feedback store and the shared ``corrections`` ledger (both read-only)
    and builds one row per judged event at the current judge prompt version — the
    canonical ``traces`` config — then projects the TRL-ready ``sft``, ``dpo``, and
    ``kto`` configs from the same rows. Every config is written as per-split parquet
    under ``out/<config>/<split>.parquet`` next to a generated dataset card at
    ``out/README.md``; with ``push``, every config is also pushed to the private
    HuggingFace repo ``repo_id`` and the card uploaded. Splits are a deterministic
    group split on the session hash, computed once on ``traces`` and inherited by
    every derived row.

    Args:
        store: The open feedback store.
        out: The directory to write the parquet files and dataset card into.
        repo_id: The HuggingFace dataset repo to push to.
        push: When True, push every config to ``repo_id`` as a private dataset.

    Returns:
        The export's per-config, per-split row counts.
    """
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    traces = await load_traces(store)
    features = config_features()
    built = {
        config: {
            split: Dataset.from_dict(
                {name: [row[name] for row in rows] for name in features[config]}, features=features[config]
            )
            for split, rows in splits.items()
        }
        for config, splits in config_rows(traces).items()
    }
    counts = {config: {split: len(rows) for split, rows in splits.items()} for config, splits in built.items()}
    for config, splits in built.items():
        (out / config).mkdir(parents=True, exist_ok=True)
        for split, dataset in splits.items():
            dataset.to_parquet(out / config / f"{split}.parquet")
    card = out / "README.md"
    card.write_text(
        dataset_card(
            counts,
            pushback_count=sum(trace["is_pushback"] is True for trace in traces),
            noise_count=sum(trace["is_pushback"] is False for trace in traces),
        )
    )
    if push:
        for config, splits in built.items():
            DatasetDict(splits).push_to_hub(repo_id, config_name=config, private=True)
        HfApi().upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo_id, repo_type="dataset")
    return ExportReport(counts=counts, out=out, pushed=push)
