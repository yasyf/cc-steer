# ![cc-steer](https://github.com/yasyf/cc-steer/raw/main/docs/assets/readme-banner.webp)

**Your best training data is rotting in ~/.claude.** cc-steer mines every correction, interrupt, and rejected plan from your transcripts into judge-refined, TRL-ready SFT/DPO/KTO pairs on HuggingFace.

[![CI](https://github.com/yasyf/cc-steer/actions/workflows/ci.yml/badge.svg)](https://github.com/yasyf/cc-steer/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cc-steer.svg)](https://pypi.org/project/cc-steer/)
[![PolyForm Noncommercial license](https://img.shields.io/badge/license-PolyForm--Noncommercial--1.0.0-blue)](https://github.com/yasyf/cc-steer/blob/main/LICENSE)

## Get started

```bash
uvx cc-steer scan
```

One pass over `~/.claude/projects` fills `~/.cc-steer/feedback.db` with every correction you typed, every plan you rejected, every question you answered, and every inline review comment, conversational context included. `stats` shows what landed:

<img src="https://github.com/yasyf/cc-steer/raw/main/docs/assets/demo.png" alt="Terminal running 'uvx cc-steer stats' — 2,792 steering events counted across five source kinds" width="700">

Driving with an agent? Paste this:

```text
Run `uvx cc-steer scan` to mine my Claude Code transcripts into ~/.cc-steer/feedback.db.
Then run `uvx cc-steer stats` and report how much steering was collected per source kind.
Docs: https://yasyf.github.io/cc-steer/
```

---

## Use cases

### Build a training set from feedback you already gave

You've spent months telling Claude "no, not like that", and that signal evaporates as transcripts rotate out. Judge the corpus, then distill the accepted events into atomic pairs, the corrective ones grounded in the code they fault:

```bash
uvx cc-steer triage
uvx cc-steer refine
uvx cc-steer enrich
```

`uvx cc-steer pairs` prints the deliverable. Each training pair is distilled from your own steering and carries the conversational window and code evidence behind it.

### See how you actually steer, across every project

Your taste is mostly tacit, and you notice a rule only when it's violated. The corpus makes it legible:

```bash
uvx cc-steer list --source plan_review
```

On my machine the split is 1,840 answered questions, 714 mid-session corrections, 212 rejected plans, 21 interrupts, and 5 review comments. Another machine's history folds in too. Mirror it with rsync and point `scan --transcripts` at it; scans are repeatable, so several mirrors fold into one corpus.

### Push a private SFT/DPO/KTO dataset to your HuggingFace namespace

A judged corpus in SQLite trains nothing. Export projects it into TRL-ready configs:

```bash
uvx cc-steer export --push
```

```
traces: train 1156  test 115
sft: train 499  test 67
dpo: train 363  test 44
kto: train 1156  test 115
```

Four configs land as per-split parquet in a private `<hf-user>/cc-steer-traces`, next to a generated dataset card. Splits group on the session hash, so a session never straddles train and test.

## More in the docs

- [Incremental scanning](https://yasyf.github.io/cc-steer/reference/cli/scan.html) explains how content digests and one-transaction commits keep re-scans cheap and interrupt-safe.
- [Triage, audit, and eval](https://yasyf.github.io/cc-steer/reference/cli/triage.html) cover prompt-versioned judging, a seeded audit sample, and mechanical metrics that need no LLM calls.
- [view-samples](https://yasyf.github.io/cc-steer/reference/cli/view_samples.html) serves a local dashboard for browsing refined pairs and their full lineage.
- [The Python API](https://yasyf.github.io/cc-steer/reference/) drives the scanner and the feedback store from your own code.

cc-steer is alpha. The pipeline runs end to end, and the judge prompt still iterates (v6 today).

Read the [docs](https://yasyf.github.io/cc-steer/) for the full guide. Licensed under [PolyForm Noncommercial 1.0.0](LICENSE).
