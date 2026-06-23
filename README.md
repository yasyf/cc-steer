# cc-pushback

![cc-pushback banner](https://github.com/yasyf/cc-pushback/raw/main/docs/assets/readme-banner.webp)

[![PyPI](https://img.shields.io/pypi/v/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Python](https://img.shields.io/pypi/pyversions/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/cc-pushback/docs.yml?branch=main&label=docs)](https://yasyf.github.io/cc-pushback/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/cc-pushback/blob/main/LICENSE)

Mine your Claude Code transcripts for the moments you pushed back into a local database.

cc-pushback collects your corrections, interrupts, rejected plans, and code-review comments — with the surrounding conversational context — into a feedback database. That corpus is the raw material for learning your pushback style; this first release builds it. Your taste is mostly tacit (you notice a rule when it's violated), and that signal sits unused in transcript files; this turns it into a structured dataset.

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
| `scan` | Scan transcripts for pushback, incrementally. `--full` re-mines every transcript; `--transcripts DIR` (repeatable) scans other directories. |
| `stats` | Counts by source kind and the scanned-file count. |
| `list` | Recent feedback, newest first. `--source KIND` (repeatable) and `--limit N`. |

Run `uvx cc-pushback COMMAND --help` for the full flag list.

## What gets collected

`scan` runs four detectors over each transcript, each tagged with a source kind: **transcript messages** (`transcript_message`, the pushback you typed mid-session), **plan reviews** (`plan_review`, rejected `ExitPlanMode` plans and plan-mode re-entries), **interrupts and rejections** (`interrupt_rejection`, permission denials and user interrupts), and **review comments** (`review_comment`, one row per inline code-review comment). Each row carries the conversational window around the feedback, captured at collection time because transcripts are ephemeral.

## Mining from another machine

Transcripts live under `~/.claude/projects`. Mirror a remote machine's history locally, then point `scan` at it — `--transcripts` is repeatable, so several mirrors fold into one scan:

```bash
rsync -az yasyf@yasyf:.claude/projects/ ~/.cc-pushback/mirrors/yasyf/
uvx cc-pushback scan --transcripts ~/.cc-pushback/mirrors/yasyf/
```

## Docs

[Read the docs](https://yasyf.github.io/cc-pushback/) for the full guide and API reference.
