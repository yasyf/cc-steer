# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `scan` command: idempotent, incremental collection of developer pushback into a
  local SQLite feedback DB (`~/.cc-pushback/feedback.db`) from existing Claude Code
  transcripts. Four detectors — transcript messages, plan reviews (rejected
  `ExitPlanMode` plans and post-edit plan re-entries), interrupts and permission
  denials, and review-format comments (superset inline cites, conductor findings
  and workstreams) — each row capturing the surrounding conversational window.
  Multiple `--transcripts` roots (including rsync mirrors of remote corpora) and a
  `--full` re-mine are supported.
- `stats` and `list` inspection commands, plus `view-samples`, which renders the
  whole corpus into one HTML page (with an optional `claude`-CLI narrative) and
  serves it over a transient local `aiohttp` server.
- Built on `cc-transcript` 0.5 for transcript discovery, parsing, the declarative
  noise-filter spec, and the file-state store.
- Async-native throughout: the store (`aiosqlite`), discovery, transcript parsing,
  and the `claude` shell-out all run on `anyio`; Click commands bridge to the async
  core via `anyio.run`.

[Unreleased]: https://github.com/yasyf/cc-pushback/commits/main
