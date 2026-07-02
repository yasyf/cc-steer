# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `export` command ships the pushback lineage as a HuggingFace dataset.
  The canonical `traces` config carries one row per judged event, and three
  TRL-ready projections derive from it. `sft` maps context and action to the
  user's verbatim pushback; `dpo` prefers the correcting edit over the faulted
  one, one row per both-sided ledger correction, deduplicated across
  dual-detected sibling events; `kto` scores unpaired desirability over every
  event. A deterministic session-hash 90/10 group split is computed once on
  `traces` and inherited by every derived row. Per-config parquet files and a
  generated dataset card land locally; `--push` uploads every config to a
  private dataset in the authenticated user's HuggingFace namespace,
  `<hf-user>/cc-pushback-traces`, with `--repo-id` overriding the target. A
  corpus with zero judged events exports cleanly; empty configs still write
  and push, and the dataset card renders the unjudged corpus honestly.
- Audit prompt v4 closes the judge-v5 boundary gap. The v3 auditor predated
  the judge's v5 boundary rules and missed exactly the four tuned cases; v4
  restates the correction boundary as six contrastive edges and scores 49/49
  on the golden set while keeping the auditor's independent quality-gate voice.
- Auto-sync to the authenticated user's HuggingFace repo after every mutating
  pass. A `scan`, `triage`, `audit`, `refine`, or `enrich` run that changed
  data rebuilds the derived dataset and pushes it; `--sync/--no-sync` controls
  this per command and defaults to on. CLI-level tests cover the push wiring
  and the sync seam, and run without a live LLM backend.

### Changed
- `datasets` and `huggingface-hub` moved from the `[export]` extra into the
  core dependencies, so a bare install can build and push the dataset.
- The ty type check now blocks CI; its `continue-on-error` escape is gone.
- The latest-verdict and latest-refinement window SQL is consolidated into
  shared `latest_judge` / `latest_auditor` / `latest_refinement` views,
  replacing the per-query CTEs the store and the export each carried.
- Enrich worker failures propagate as an `ExceptionGroup` instead of a `failed`
  counter. A failing pair aborts the pass loudly; corrections already appended
  to the ledger persist, so a re-run resumes idempotently.
- Dashboard summaries require the `claude` CLI and raise on any subprocess or
  parse failure.

### Removed
- The `[export]` extra, superseded by the core dependencies above.
- `view-samples --llm/--no-llm` and the heuristic summary path.
- `EnrichReport.failed`; `enriched` is now derived as `corrections + skipped`.
- `Lineage.status` and the `GoldenFailure`/`GoldenResult` re-exports.
- `lan_ip`'s loopback fallback; a routeless host now raises.
- The package-level `__init__` re-exports and `__all__`. The facade existed
  only to steer the great-docs API walker, and the `scan` re-export shadowed
  its own submodule; the API reference is now curated in `great-docs.yml`.

### Fixed
- Detectors crash on unknown review provenance instead of silently returning
  `None` from the survival gate.
- Sidecar `Finding.parse` narrows `line` with a typed check instead of a
  `type: ignore`.
- `view-samples` refuses an empty corpus with a clear error; an all-noise
  corpus now serves with an empty highlights set instead of crashing on its
  empty candidate pool.

## [0.7.2]

### Changed
- The release pipeline adopts the shared `release-pypi.yml@pypi-v1` reusable
  workflows. The build runs via `release-pypi-build.yml@pypi-v1`, while the
  OIDC publish to PyPI and the GitHub release run in-repo.

## [0.7.1]

### Changed
- Requires spawnllm `>=0.5.1,<0.6` and adopts its structured `Response`. Text
  reads from `result.raw`, errors from `error.msg`, and a timeout arrives as a
  plain `error` instead of being mapped to `subprocess.TimeoutExpired`.

## [0.7.0]

### Changed
- Requires cc-transcript `>=7,<8` and spawnllm `>=0.5.0`, migrating to the
  spawnllm 0.5 run/call/extract API. `run()` returns a `Response` carrying
  error, result, and parsed fields, and `parse_result_envelope` is gone.
- Judge prompt v5 fixes four golden boundary misses. Removal orders undoing
  the assistant's own additions, mixed directives carrying a
  constraint-violation clause, terse review-comment prohibitions, and
  over-reading a "no/nope" that merely answers the assistant's question are
  now judged correctly; the golden set confirms 49/49 at repeat-3 majority
  without regressing the other 45.

## [0.6.0]

### Changed
- Requires cc-transcript `>=6,<7`, adopting its declarative mining API. The six
  `iter_*_signals` detector entrypoints and `extract_all` are replaced by a single
  spec-driven `mine(events, spec)` over a `MiningSpec`. cc-pushback's review policy
  is now a `ReviewSpec` carrying its formats and `surfaces={"typed","surfaced"}`,
  and `detect` runs every detector through one `mine` pass. Mined output (confidence
  and reason tuples) is byte-identical — the spec defaults reproduce the historical
  scoring.

### Removed
- The `extract_all` extraction wrapper and the dedicated `extract_conductor_finding`
  callable. The `conductor-finding` format is now a portable `RegexReviewFormat`
  whose comment-join the platform's `regex_review_comments` performs;
  `superset-inline` and `conductor-workstream` remain `CallableReviewFormat`s
  (lookahead / multi-pass), keeping the review detector in Python by design.

## [0.5.0]

### Added
- Capture human-surfaced code-review findings. The review-comment detector now
  scans `typed` + `surfaced` provenances (human-typed inline cites plus findings
  surfaced via tool-result output), gated to exclude Claude-authored (`claude`)
  self/subagent reviews, and extracts structured `StructuredOutput`-style payloads
  via a field-map.
- A `--findings <dir>` source ingests superset `.context/cleanup/*issues.jsonl`
  findings, anchoring each to the closest session by timestamp.

### Changed
- Requires cc-transcript `>=5,<6` (the 5.0.0 review-scan API: required
  `surfaces`/`structured_formats`).

## [0.4.0]

### Changed
- Requires cc-transcript `>=4,<5` and spawnllm `>=0.2.0`. The `enrich` stage now
  delegates to cc-transcript's shared correction extractor
  (`cc_transcript.extract.extract_correction`): per refined pair it harvests the
  candidate edits around the pushback anchor, picks the one the complaint faults —
  an LLM call when a backend is ready (`usable_backend()`), the best-overlap
  candidate otherwise — and appends it to the shared `corrections` ledger. The
  extractor is idempotent per anchor, so pairs sharing one anchor produce a single
  row.

### Removed
- The local `pair_evidence` table, `pair_evidence_latest` view, and the evidence
  columns on the `refined_pairs` view, along with `FeedbackStore.record_evidence`
  and the `EXTRACTOR_VERSION`/`ENRICH_VERSION`-keyed evidence generation. Code
  evidence now lives only in the shared ledger; the dashboard reads it by anchor
  via `CorrectionLog.for_anchor`. `FeedbackStore.unenriched` now takes a
  `CorrectionLog` and returns refined pairs whose anchor carries no ledger row.
  `EvidenceRow` is built from a ledger `Correction` and no longer carries a `note`.

## [0.3.0]

### Changed
- Requires cc-transcript `>=3.0,<4`; a candidate's de-noising signal is always
  present (`Sample.signal` is required).

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
