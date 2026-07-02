# cc-pushback

![cc-pushback banner](https://github.com/yasyf/cc-pushback/raw/main/docs/assets/readme-banner.webp)

[![PyPI](https://img.shields.io/pypi/v/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Python](https://img.shields.io/pypi/pyversions/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/cc-pushback/docs.yml?branch=main&label=docs)](https://yasyf.github.io/cc-pushback/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/cc-pushback/blob/main/LICENSE)

Mine your Claude Code transcripts for the moments you pushed back into a local database.

cc-pushback collects your corrections, interrupts, rejected plans, and code-review comments — with the surrounding conversational context — into a feedback database, then judges each candidate, refines the accepted ones into atomic training pairs, and grounds them in the code they complain about. Your taste is mostly tacit (you notice a rule when it's violated), and that signal sits unused in transcript files; this turns it into a training dataset.

## Install

Run with [uvx](https://docs.astral.sh/uv/): `uvx cc-pushback --help`.

## Quickstart

Scan your transcripts and accumulate the pushback into a local feedback database:

```bash
uvx cc-pushback scan
```

```
scanned 412 files, 1473 new rows
```

`scan` is incremental and idempotent: each transcript is parsed only when new or changed, every candidate is keyed by a content digest, and recording a file commits in one transaction, so an interrupted scan never leaves the database half-written. The database lives at `~/.cc-pushback/feedback.db` by default (override with `--db`).

## Commands

| Command | What it does |
| --- | --- |
| `scan` | Scan transcripts for feedback, incrementally. `--full` re-mines every transcript; `--transcripts DIR` (repeatable) scans other directories. |
| `stats` | Print ingestion counts by source kind and triage coverage. |
| `list` | List recent feedback events, newest first. `--source KIND` and `--limit N`. |
| `triage` | Judge every stored candidate lacking a verdict at the current prompt version. |
| `audit` | Audit a seeded stratified sample of the current prompt version's verdicts. |
| `eval` | Compute the mechanical metrics for the current prompt version. No LLM calls. |
| `refine` | Refine every accepted pushback event into atomic training pairs. |
| `enrich` | Ground every refined pair in the code it complains about. |
| `export` | Export the pushback lineage as a HuggingFace dataset. `--push` uploads every config to the private HF repo. |
| `pairs` | Print the refined training pairs — the pipeline's deliverable. |
| `view-samples` | Serve the training-pairs dashboard: refined pairs and their full lineage. |

`scan`, `triage`, `audit`, `refine`, and `enrich` sync the dataset to HuggingFace whenever a pass changes data; `--no-sync` skips it. Run `uvx cc-pushback COMMAND --help` for the full flag list.

## What gets collected

`scan` runs four detectors over each transcript, each tagged with a source kind: **transcript messages** (`transcript_message`, the pushback you typed mid-session), **plan reviews** (`plan_review`, rejected `ExitPlanMode` plans and plan-mode re-entries), **interrupts and rejections** (`interrupt_rejection`, permission denials and user interrupts), and **review comments** (`review_comment`, one row per inline code-review comment). Each row carries the conversational window around the feedback, captured at collection time because transcripts are ephemeral.

## Exporting a training dataset

Once the corpus is judged, `export` turns it into a HuggingFace dataset:

```bash
uvx cc-pushback export
```

```
traces: train 1156  test 115
sft: train 499  test 67
dpo: train 363  test 44
kto: train 1156  test 115
```

One canonical `traces` config — one row per judged event, carrying the context, verdicts, refined pairs, and code evidence — plus three TRL-ready projections (`sft`, `dpo`, `kto`) land as per-split parquet under `~/.cc-pushback/dataset` (override with `--out`), next to a generated dataset card. Splits are a deterministic group split on the session hash, so a session never straddles train and test. `--push` uploads every config to a private HF repo (`--repo-id`, default `yasyf/cc-pushback-traces`).

You rarely run `export` by hand. Every mutating pass that changed data rebuilds the dataset and pushes all four configs to that repo. A failed push exits nonzero with the local writes already committed, and the next sync picks them up; `export --push` is the manual catch-up.

## Mining from another machine

Transcripts live under `~/.claude/projects`. Mirror a remote machine's history locally, then point `scan` at it — `--transcripts` is repeatable, so several mirrors fold into one scan:

```bash
rsync -az yasyf@yasyf:.claude/projects/ ~/.cc-pushback/mirrors/yasyf/
uvx cc-pushback scan --transcripts ~/.cc-pushback/mirrors/yasyf/
```

## Docs

[Read the docs](https://yasyf.github.io/cc-pushback/) for the full guide and API reference.
