# cc-pushback

![cc-pushback banner](https://github.com/yasyf/cc-pushback/raw/main/docs/assets/readme-banner.png)

[![PyPI](https://img.shields.io/pypi/v/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Python](https://img.shields.io/pypi/pyversions/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/cc-pushback/docs.yml?branch=main&label=docs)](https://yasyf.github.io/cc-pushback/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/cc-pushback/blob/main/LICENSE)

cc-pushback mines your Claude Code transcripts for the moments you pushed back — corrections, interrupts, rejected plans, code-review comments, "no, do it this way" — and collects them, with the surrounding conversational context, into a local database. That corpus is the raw material for learning your pushback style; this first release builds it.

## Install

No install needed — run everything through [uvx](https://docs.astral.sh/uv/):

```bash
uvx cc-pushback --help
```

`uvx` fetches cc-pushback into a throwaway environment and runs it. To add it
to a project instead:

```bash
uv add cc-pushback
```

## Quickstart

Scan your transcripts for the moments you pushed back and accumulate them into a
local feedback database:

```bash
uvx cc-pushback scan
```

```
scanned 412 files, 1473 new rows
```

`scan` is incremental and idempotent. Each transcript is parsed only when it is
new or has changed since the last scan, and every candidate is keyed by a content
digest, so re-running over unchanged inputs adds nothing. Recording a file and
inserting its candidates commit in one transaction — interrupt a scan and the
database is never left half-written. A transcript that fails to parse (one Claude
Code is still writing, say) is skipped and retried next time, never aborting the run.

The database lives at `~/.cc-pushback/feedback.db` by default (override with
`--db`). Inspect what has been collected:

```bash
uvx cc-pushback stats          # counts by source kind, and the scanned-file count
uvx cc-pushback list           # recent feedback, newest first
uvx cc-pushback list --source plan_review --limit 50
```

### What gets collected

`scan` runs four detectors over each transcript:

- **Transcript messages** (`transcript_message`) — the pushback you typed
  mid-session, after trivial acknowledgements and structural noise are filtered out.
- **Plan reviews** (`plan_review`) — rejected `ExitPlanMode` plans (with the
  feedback you gave) and plan-mode re-entries right after an edit cycle, i.e.
  "let's rethink this."
- **Interrupts and rejections** (`interrupt_rejection`) — permission denials and
  `[Request interrupted by user]` corrections, with the denied tool and your
  follow-up captured.
- **Review comments** (`review_comment`) — structured code-review messages
  exploded into one row per inline comment.

Each row carries the conversational window around the feedback — the assistant
action it responded to, plus a few turns either side — captured at collection
time, because transcripts are ephemeral.

Restrict to specific kinds with `--source` (repeatable), or force a full re-mine
of every transcript (after a detector change, say) with `--full`:

```bash
uvx cc-pushback list --source transcript_message --source plan_review
uvx cc-pushback scan --full
```

### Mining transcripts from another machine

Transcripts live under `~/.claude/projects`. To mine a remote machine's history,
mirror its projects directory locally with `rsync`, then scan that directory:

```bash
rsync -az yasyf@yasyf:.claude/projects/ ~/.cc-pushback/mirrors/yasyf/
uvx cc-pushback scan --transcripts ~/.cc-pushback/mirrors/yasyf/
```

`--transcripts` is repeatable, so you can fold several mirrors into one scan.
Because discovery is mtime-keyed, repeating the `rsync` and re-scanning only
ingests what changed.

## What problems does this solve?

- **Your corrections evaporate.** Every "don't do it that way" you've typed into Claude Code is sitting unused in transcript files. cc-pushback turns that history into a structured dataset.
- **CLAUDE.md only captures what you remember to write down.** Most of your taste is tacit — you only notice a rule when it's violated. Collecting real pushbacks recovers the rules you never articulated.
- **The signal is buried in noise.** Trivial acknowledgements, structural reminders, and tool chatter drown out the moments that matter; cc-pushback keeps the pushback and discards the rest.

## Docs

[Read the docs](https://yasyf.github.io/cc-pushback/) for the full guide and API reference.
