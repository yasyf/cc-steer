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

Extract every message you've ever typed into Claude Code into a JSONL dataset:

```bash
uvx cc-pushback extract --output messages.jsonl
```

```
Extracted 4127 messages to messages.jsonl
```

Each line is one `{"session", "text"}` record — the raw material for pushback
labeling and classifier training.

## What problems does this solve?

- **Your corrections evaporate.** Every "don't do it that way" you've typed into Claude Code is sitting unused in transcript files. cc-pushback turns that history into a training dataset.
- **CLAUDE.md only captures what you remember to write down.** Most of your taste is tacit — you only notice a rule when it's violated. Mining real pushbacks recovers the rules you never articulated.
- **You repeat the same code review feedback.** A classifier trained on your past pushbacks can flag the same issues before you have to — your review style, applied preemptively.

## Docs

[Read the docs](https://yasyf.github.io/cc-pushback/) for the full guide and API reference.
