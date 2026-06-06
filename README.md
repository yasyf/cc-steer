# cc-pushback

[![PyPI](https://img.shields.io/pypi/v/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Python](https://img.shields.io/pypi/pyversions/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/cc-pushback/docs.yml?branch=main&label=docs)](https://yasyf.github.io/cc-pushback/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/cc-pushback/blob/main/LICENSE)

cc-pushback mines your Claude Code transcripts for the moments you pushed back — corrections, code review comments, "no, do it this way" — and trains a classifier on them so a language model can replicate your pushbacks. Instead of hand-writing rules into CLAUDE.md, your accumulated feedback history becomes the spec.

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

Scan your transcripts, code reviews, and issue files for the moments you pushed
back, and accumulate them into a local feedback database:

```bash
uvx cc-pushback scan
```

```
transcripts=412, github_review=37
```

`scan` is incremental and idempotent. Each transcript is parsed only when it is
new or has changed since the last scan, and every candidate is keyed by a content
digest, so re-running over unchanged inputs adds nothing. Recording a file and
inserting its candidates commit in one transaction — interrupt a scan and the
database is never left half-written.

The database lives at `~/.cc-pushback/feedback.db` by default (override with
`--db`). Inspect what has been ingested:

```bash
uvx cc-pushback stats          # counts by source kind, file count, cursors
uvx cc-pushback list           # recent feedback, newest first
uvx cc-pushback list --source plan_review --limit 50
```

### What gets mined

`scan` runs several sources over your history:

- **Transcript messages** — the pushback you typed mid-session.
- **Plan reviews** — rejected `ExitPlanMode` plans and plan-mode re-entries
  after an edit cycle, i.e. "let's rethink this."
- **Interrupts and rejections** — permission denials and `[Request interrupted
  by user]` corrections, with the denied tool and your follow-up captured.
- **GitHub reviews** — your own review comments on pull requests authored by
  Claude Code, paginated incrementally per repository.
- **Superset issues** — `.context/cleanup/issues.jsonl` cleanup findings.

Restrict to specific sources with `--source` (repeatable), and skip the GitHub
source with `--no-github`:

```bash
uvx cc-pushback scan --source transcript_message --source plan_review --no-github
```

### Scanning issue files

Point `--issues` (repeatable) at one or more roots; each is searched recursively
for `.context/cleanup/issues.jsonl`:

```bash
uvx cc-pushback scan --issues ~/Code/my-project --issues ~/Code/other-project
```

## Classifying feedback

Once feedback is ingested, `classify` labels each event against a fixed taxonomy
of pushback patterns — `no-defensive-coding`, `ask-before-assuming`,
`minimal-scope`, `match-surrounding-code`, and so on:

```bash
uvx cc-pushback classify
```

```
events: 63  matcher rows: 41  llm rows: 71  new: 112
novel proposals: prefer-named-constants, reuse-test-builders
```

Classification runs in two passes:

- **Cheap matcher pass** — every event is tested against the taxonomy's regex
  and structural matchers (e.g. a denied `Edit`/`Write` is `denied-edit`). This
  is free, runs offline, and needs no language model.
- **Language-model pass** — every loaded event is then sent to a backend, which
  assigns a severity (`nit`, `minor`, `major`, `blocking`), restates the rule in
  your voice, names every taxonomy pattern that fits, and proposes a kebab-case
  `novel` pattern when nothing in the taxonomy applies.

Both passes write with `INSERT OR IGNORE` keyed by the taxonomy and prompt
versions, so re-running `classify` only labels events that are new or whose
taxonomy version has been bumped — the language model is never re-invoked on an
event it has already classified.

Pick a backend and model size, cap the batch, or skip the language model
entirely:

```bash
uvx cc-pushback classify --backend codex --model medium   # default: claude / small
uvx cc-pushback classify --limit 200                      # classify at most 200 events
uvx cc-pushback classify --no-llm                         # cheap matcher pass only, fully offline
```

`--no-llm` is useful offline or in CI: it records every regex and structural
match without spending a single model call.

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

- **Your corrections evaporate.** Every "don't do it that way" you've typed into Claude Code is sitting unused in transcript files. cc-pushback turns that history into a training dataset.
- **CLAUDE.md only captures what you remember to write down.** Most of your taste is tacit — you only notice a rule when it's violated. Mining real pushbacks recovers the rules you never articulated.
- **You repeat the same code review feedback.** A classifier trained on your past pushbacks can flag the same issues before you have to — your review style, applied preemptively.

## Docs

[Read the docs](https://yasyf.github.io/cc-pushback/) for the full guide and API reference.
