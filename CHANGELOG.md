# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scan` command: idempotent, incremental ingestion of pushback into a local
  SQLite feedback DB (`~/.cc-pushback/feedback.db`) from transcript user
  messages, plan-mode reviews, interrupts and permission denials,
  review-format comments (superset inline cites, conductor findings and
  workstreams), the current repo's GitHub PR review comments, and superset
  `issues.jsonl` findings. Multiple `--transcripts`/`--issues` roots supported,
  including rsync mirrors of remote corpora.
- `classify` command: declarative pattern taxonomy (regex + structural
  matchers) plus LLM extraction via the `claude` and `codex` CLIs with
  structured output, versioned by taxonomy and prompt for safe re-runs.
- `stats` and `list` inspection commands.
- `cc_pushback.llm`: claude/codex CLI backends, prompt builder, and a
  concurrency-bounded batch runner, ported from captain-hook.

### Removed
- The `extract` command and its flat JSONL export; `scan` subsumes it.

[Unreleased]: https://github.com/yasyf/cc-pushback/commits/main
