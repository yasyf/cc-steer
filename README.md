# ![cc-pushback](https://github.com/yasyf/cc-pushback/raw/main/docs/assets/readme-banner.webp)

**Your best training data is rotting in ~/.claude.** cc-pushback mines every correction, interrupt, and rejected plan from your transcripts into judge-refined, TRL-ready SFT/DPO/KTO pairs on HuggingFace.

[![CI](https://github.com/yasyf/cc-pushback/actions/workflows/ci.yml/badge.svg)](https://github.com/yasyf/cc-pushback/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cc-pushback.svg)](https://pypi.org/project/cc-pushback/)
[![License: PolyForm Noncommercial](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-blue)](https://github.com/yasyf/cc-pushback/blob/main/LICENSE)

## Get started

```bash
uvx cc-pushback scan
```

One pass over `~/.claude/projects` fills `~/.cc-pushback/feedback.db` with every correction you typed, every plan you rejected, and every inline review comment — conversational context included. `stats` shows what landed:

<img src="https://github.com/yasyf/cc-pushback/raw/main/docs/assets/demo.png" alt="Terminal running 'uvx cc-pushback stats' — 980 pushback events counted across four source kinds" width="700">

Driving with an agent? Paste this:

```text
Run `uvx cc-pushback scan` to mine my Claude Code transcripts into ~/.cc-pushback/feedback.db.
Then run `uvx cc-pushback stats` and report how much pushback was collected per source kind.
Docs: https://yasyf.github.io/cc-pushback/
```

---

## Use cases

### Build a training set from feedback you already gave

You've spent months telling Claude "no, not like that", and that signal evaporates as transcripts rotate out. Judge the corpus, then distill the accepted events into atomic pairs grounded in the code they complain about:

```bash
uvx cc-pushback triage
uvx cc-pushback refine
uvx cc-pushback enrich
```

`uvx cc-pushback pairs` prints the deliverable: training pairs distilled from your own pushback, each carrying the conversational window and code evidence behind it.

### See what you actually push back on, across every project

Your taste is mostly tacit — you notice a rule when it's violated. The corpus makes it legible:

```bash
uvx cc-pushback list --source plan_review
```

On my machine the split is 698 mid-session corrections, 219 rejected plans, 41 review comments, and 22 interrupts. Another machine's history folds in too: mirror it with rsync and point `scan --transcripts` at it (repeatable, so several mirrors fold into one scan).

### Push a private SFT/DPO/KTO dataset to your HuggingFace namespace

A judged corpus in SQLite trains nothing. Export projects it into TRL-ready configs:

```bash
uvx cc-pushback export --push
```

```
traces: train 1156  test 115
sft: train 499  test 67
dpo: train 363  test 44
kto: train 1156  test 115
```

Four configs land as per-split parquet in a private `<hf-user>/cc-pushback-traces`, next to a generated dataset card. Splits group on the session hash, so a session never straddles train and test.

## More in the docs

- **Incremental scanning** — content digests and one-transaction commits make re-scans cheap and interrupt-safe — [scan](https://yasyf.github.io/cc-pushback/reference/cli/scan.html)
- **Judge, audit, eval** — prompt-versioned triage, a seeded audit sample, and mechanical metrics with no LLM calls — [triage](https://yasyf.github.io/cc-pushback/reference/cli/triage.html)
- **Pair dashboard** — browse refined pairs and their full lineage in a local web UI — [view-samples](https://yasyf.github.io/cc-pushback/reference/cli/view_samples.html)
- **Python API** — drive the scanner and the feedback store from your own code — [reference](https://yasyf.github.io/cc-pushback/reference/)

Status: alpha — the pipeline runs end to end; the judge prompt still iterates (v5 today).

Read the [docs](https://yasyf.github.io/cc-pushback/) for the full guide. Licensed under [PolyForm Noncommercial 1.0.0](LICENSE).
