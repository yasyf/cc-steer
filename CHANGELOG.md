# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Changed
- Rebuilt on the cc-transcript 2.0 platform (`cc_transcript.mining` / `judge` /
  `context`): candidates now persist durable `cc-transcript.context/1` windows
  (refs plus labeled previews) captured over `SessionActivity`, instead of
  bake-truncated `ContextSnapshot` prose. Triage/audit/refine prompts hydrate
  each window and render at full fidelity while the transcript lives (a generous
  trigger-turn budget, so e.g. a >1500-char edit is no longer truncated), and
  fall back to the labeled summary previews once it expires; every verdict
  records the fidelity it was judged at. Prompt versions bumped: triage v4,
  audit v3, refine v2.
- `feedback_events.origin_path` is now a display hint only (the dashboard's
  project labels); transcript resolution goes through cc-transcript discovery by
  session UUID. The legacy `origin_uuid` column is replaced by the platform's
  `event_uuid`.
- Removed the `cc_pushback.context` / `nav` / `markers` / shim modules; their
  contents live in `cc_transcript.mining` and `cc_transcript.context`.

### Added
- `enrich` command — the pipeline's new final stage (scan → triage → audit →
  refine → enrich): grounds each refined pair in the code it complains about.
  Candidate incorrect edits and the corrections that later overwrote them are
  harvested deterministically around the pushback anchor
  (`cc_transcript.evidence`: session corrections ranked by hunk overlap, with a
  read-only git-pickaxe fallback), and an LLM picks the one edit per complaint,
  copied verbatim. Expired transcripts and editless lookback windows persist
  free `no_code` sentinel rows (`pair_index=-1`) with no LLM call. Evidence
  lands in the `pair_evidence` table keyed to the refine generation it
  annotates — `UNIQUE(dedup_key, refine_version, refine_model, pair_index,
  enrich_version, enrich_model, extractor_version)` — so a refine re-run or an
  `EXTRACTOR_VERSION` bump re-derives automatically; the `refined_pairs` view
  and the lineage detail carry each pair's latest evidence.
- Evidence surfaces in the UI: dashboard pair cards gain a collapsible
  before/after diff for evidence-grounded pairs — compact incorrect/correct
  panes (old lines `del`-tinted, new lines `ins`-tinted), a file chip, and a
  `git` chip when the correction came from git history — omitted entirely for
  pairs without code evidence. The lineage detail's refiner stage renders the
  same panes with the full, untruncated diff.
- `migrate-corpus` command: one-time, idempotent conversion of a pre-2.0
  database — legacy `context_json` snapshots become `cc-transcript.context/1`
  documents (previews only, summary fidelity, `origin='migrated'`), and the
  `event_uuid` / `triage.fidelity` columns are added.
- `triage --refresh-summary`: re-judges rows whose verdict was recorded at
  summary fidelity; a full-fidelity verdict replaces the summary one once the
  row's window hydrates again.
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
