# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.13.0] - 2026-07-17

### Changed
- **The watcher retrain now runs on `experiment-at-home` 0.6's pipelined Tinker
  engine** instead of cc-steer's own training loop. The old loop was the Tinker
  SDK's documented anti-pattern — it awaited every forward-backward and optim
  step round trip (~3 clock cycles per step where 1 suffices) and scored each
  checkpoint with hundreds of serial per-row calls; the new lane submits each
  step's pair together under a submit-ahead queue and scores every checkpoint
  with one batched forward riding the same stream. `retrain_watcher` hands
  `athome.train.retrain` the whole pipeline shape and supplies only domain
  callables: sentinel eval rows, checkpoint selection by sentinel AUC, local
  MLX scoring of the exact artifact that will serve, and the corrected gate.
  Registration, promotion, pruning, and the journal stay in cc-steer,
  post-return, unchanged in shape — metadata keys, `journal.jsonl` records,
  and the `serving_diagnostic.json` sidecar are byte-compatible.
- **`batch_size` 16 (was 4)** in the packaged watcher recipe — ~4x fewer
  optimizer steps at the same token budget and cost. The learning rate
  deliberately stays `1e-4` (not silently retuned); the AUC checkpoint pick
  and the corrected gate revalidate the new dynamics on the next run, and a
  reject is the designed containment. The step count now derives from the
  curated pool size rather than the post-length-filter survivor count —
  over-length rows drop inside athome after the count is fixed, a
  near-nil drift for short chat turns.
- The LoRA scale served after conversion is now read from Tinker's own PEFT
  `adapter_config.json` (athome's converter) instead of a hardcoded
  `alpha=32`. Nothing ever served at the wrong scale — both values agreed —
  but the agreement was coincidence, not an invariant; now it is structural.
- The gate statistics (`sign_test_p`, `threshold_for_budget`,
  `matched_fire_mask`, `sentinel_auc`, `corrected_gate`, `GateResult`) moved
  to `athome.train.gate` and are re-exported from `cc_steer.retrain.promotion`
  unchanged. The port makes fire orientation explicit (higher-is-fire) and
  fixes a latent float-floor defect that could drop one fire from an integer
  budget; a field-for-field golden replay pinned numerical equivalence.
- Sentinel-row construction moved to `cc_steer.retrain.sentinel`, expressed as
  pre-tokenized `EvalRow`s over `athome.train.data`'s tokenizer helpers — the
  NO_STEER-divergence trick is unchanged (verified byte-identical datum
  construction over 300 real pool rows before the cutover).

### Removed
- `cc_steer/retrain/tinker.py` — the bespoke Tinker client, trainer, scorers,
  and PEFT→MLX converter (464 lines) are gone; `experiment-at-home[train]`
  provides all of it. The `tinker` dependency left `pyproject.toml` with it
  (athome's `train` extra carries the SDK), and the core dependency is now
  `experiment-at-home[gate]>=0.6,<0.7`. `ConversionDroppedError` is deleted:
  athome's converter crashes before registration on any unfusable tensor, so
  the guarded state is unrepresentable.

## [0.12.1] - 2026-07-14

### Fixed
- `install_watch` now rewrites the bare default prefix to resolve the `gate`
  and `mlx` extras (`uvx --from 'cc-steer[gate,mlx]' cc-steer`), mirroring the
  retrain agent's prefix rewrite. The v0.12.0 default installed a watch daemon
  that crash-looped on startup because the base dist cannot import the lexical
  gate's scikit-learn or the mlx drafter's mlx-lm.

## [0.12.0] - 2026-07-14

### Added
- The production retrain loop, consolidated into the package from the lab:
  `cc-steer retrain --component gate|watcher` retrains the stage-1 lexical gate
  locally with sklearn and the stage-2 watcher LoRA on Tinker's managed
  training API under a hard spend cap, scores each candidate server-side
  against a frozen eval (`cc-steer freeze-eval`), gates it against the
  incumbent, and auto-promotes only on a strict beat — restarting the watch
  daemon on a watcher promote. The `retrain` extra carries the heavy deps; the
  weekly launchd agent (`cc-steer pipeline install-launchd`) now runs both
  lanes with no lab checkout, and the watch agent serves each promoted gate at
  its fitted threshold instead of a hardcoded `--gate-threshold 0.5`.

## [0.11.0] - 2026-07-13

### Added
- The live steering watcher: `cc-steer watch --shadow` tails open Claude Code
  sessions and runs a staged cascade — a cheap stage-1 gate over the flattened
  context window, a stage-2 drafting model, and an optional stage-3
  exemplar-conditioned refiner — on every turn a session completes and goes
  quiet. Proposals (abstentions included) land in a local shadow ledger at
  `~/.cc-steer/shadow.db`; no session is ever touched. `cc-steer shadow report`
  joins the ledger against the interventions users actually made
  (time-within-session, tunable window) and reports hit/nuisance rates,
  per-stage abstention, per-category hit counts, and the drafter's
  sentinel-probability distribution; `--journal-repo` appends the summary to a
  cc-notes log.
- A file-based model registry under `~/.cc-steer/models` (immutable version
  dirs, atomic `current` promotion, rollback, pruning) with `cc-steer models
  list/promote/rollback`, serving two components: the lab-trained lexical
  `gate` (TF-IDF + calibrated logistic regression behind the `gate` extra) and
  the lab-trained LoRA `watcher`.
- The two-stage local watcher, E2's winner, behind the new `mlx` extra:
  `MlxDrafter` serves the registered QLoRA adapter over the 4-bit base with
  score-based sentinel abstention — abstain iff first-token P(NO_STEER) ≥ the
  promoted threshold (precision-first budget point by default), generate with
  the sentinel banned otherwise — never greedy string-match. `watch` gains
  `--drafter auto|spawn|mlx`, `--stage2-threshold`, and `--refiner
  auto|spawn|none`; with the mlx drafter, stage 3 defaults to none and a fired
  draft ships as the steer. Proposals carry the new `sentinel_prob` column.
  The drafter's input reproduces the training rendering byte-for-byte
  (`tail_messages` under `DRAFT_CHAR_CAP`, shared with the lab by identity
  import), and every promoted watcher version pins its `render_version`.
- Exemplar retrieval (`embed` extra): a Voyage-embedded index over accepted
  steering moments with MMR selection, feeding the stage-3 refiner; capture
  hooks, launchd wiring, negatives sampling, and the nightly pipeline runner
  that journals each pass via cc-notes.

### Changed
- The cascade's `Drafter` protocol returns a `Draft` (text plus optional
  `sentinel_prob`) instead of a bare string, and `Cascade.refiner` is optional
  (`None` ships fired drafts as-is) — the seam E9's rewrite-only refiner slots
  into.

## [0.10.0] - 2026-07-05

### Changed
- Raised the `cc-transcript` floor to 9.0.0. Verdict identity drops `model`: a
  triage verdict is now unique per `(dedup_key, role, prompt_version)`, with
  `model` kept as provenance only, so switching the judge backend no longer
  re-judges the whole corpus. The `triage` table gains a `canonical_key`
  column; steering triage names no durable rule, so it stays null.

## [0.9.0]

### Added
- New `export` command ships the steering lineage as a HuggingFace dataset.
  The canonical `traces` config carries one row per judged event, and three
  TRL-ready projections derive from it. `sft` maps context and action to the
  user's verbatim steering; `dpo` prefers the correcting edit over the faulted
  one, one row per both-sided ledger correction, deduplicated across
  dual-detected sibling events; `kto` scores unpaired desirability over every
  event. A deterministic session-hash 90/10 group split is computed once on
  `traces` and inherited by every derived row. Per-config parquet files and a
  generated dataset card land locally; `--push` uploads every config to a
  private dataset in the authenticated user's HuggingFace namespace,
  `<hf-user>/cc-steer-traces`, with `--repo-id` overriding the target. A
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
- The judge and pipeline reframe from pushback to steering: any moment a human
  shapes a decision the assistant faced or raised, covering corrections plus
  the forward direction that resolves an open choice, such as an answered
  question, a picked option, or a directive. Judge prompt v6 adds a `direction` category and
  captures answered `AskUserQuestion` rounds as `question_answer` steering,
  resolving ordinal picks like "2, but…" to the concrete option chosen.
- The package is renamed from `cc-pushback` to `cc-steer`: the PyPI
  distribution, the `cc-steer` CLI, the `cc_steer` module, the `~/.cc-steer/`
  data directory, and the HuggingFace dataset `<hf-user>/cc-steer-traces`.
  Breaking, with no backwards-compat layer.
- The refiner distills each accepted message into `{action, direction}` pairs,
  replacing `{action, complaint}`.
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
  spec-driven `mine(events, spec)` over a `MiningSpec`. cc-steer's review policy
  is now a `ReviewSpec` carrying its formats and `surfaces={"typed","surfaced"}`,
  and `detect` runs every detector through one `mine` pass. The mined confidence
  and reason tuples are byte-identical. The spec defaults reproduce the
  historical scoring.

### Removed
- The `extract_all` extraction wrapper and the dedicated `extract_conductor_finding`
  callable. The `conductor-finding` format is now a portable `RegexReviewFormat`
  whose comment-join the platform's `regex_review_comments` performs;
  `superset-inline` and `conductor-workstream` remain `CallableReviewFormat`s,
  their lookahead and multi-pass logic keeping the review detector in Python
  by design.

## [0.5.0]

### Added
- Capture human-surfaced code-review findings. The review-comment detector now
  scans `typed` + `surfaced` provenances, covering human-typed inline cites
  plus findings surfaced via tool-result output. It is gated to exclude
  Claude-authored (`claude`) self/subagent reviews and extracts structured
  `StructuredOutput`-style payloads via a field-map.
- A `--findings <dir>` source ingests superset `.context/cleanup/*issues.jsonl`
  findings, anchoring each to the closest session by timestamp.

### Changed
- Requires cc-transcript `>=5,<6`, whose 5.0.0 review-scan API makes
  `surfaces` and `structured_formats` required.

## [0.4.0]

### Changed
- Requires cc-transcript `>=4,<5` and spawnllm `>=0.2.0`. The `enrich` stage now
  delegates to cc-transcript's shared correction extractor,
  `cc_transcript.extract.extract_correction`. Per refined pair it harvests the
  candidate edits around the steering anchor, picks the one the complaint
  faults, and appends it to the shared `corrections` ledger. The pick is an LLM
  call when a backend is ready (`usable_backend()`) and the best-overlap
  candidate otherwise. The extractor is idempotent per anchor, so pairs sharing
  one anchor produce a single row.

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
- Requires cc-transcript `>=3.0,<4`; `Sample.signal` is required, so a
  candidate's de-noising signal is always present.

## [0.2.0]

### Changed
- Rebuilt on the cc-transcript 2.0 platform of `cc_transcript.mining`,
  `cc_transcript.judge`, and `cc_transcript.context`. Candidates now persist
  durable `cc-transcript.context/1` windows of refs plus labeled previews,
  captured over `SessionActivity`, instead of bake-truncated `ContextSnapshot`
  prose. Triage/audit/refine prompts hydrate each window and render at full
  fidelity while the transcript lives, under a generous trigger-turn budget
  that no longer truncates e.g. a >1500-char edit, and fall back to the labeled
  summary previews once it expires; every verdict records the fidelity it was
  judged at. Prompt versions bumped to triage v4, audit v3, and refine v2.
- `feedback_events.origin_path` is now a display hint only, feeding the
  dashboard's project labels; transcript resolution goes through cc-transcript
  discovery by session UUID. The legacy `origin_uuid` column is replaced by the
  platform's `event_uuid`.
- Removed the `cc_steer.context` / `nav` / `markers` / shim modules; their
  contents live in `cc_transcript.mining` and `cc_transcript.context`.

### Added
- `enrich` command, the new final stage of the scan -> triage -> audit ->
  refine -> enrich pipeline. It grounds each refined pair in the code it
  complains about. Candidate incorrect edits and the corrections that later
  overwrote them are harvested deterministically around the steering anchor by
  `cc_transcript.evidence`, which ranks session corrections by hunk overlap
  with a read-only git-pickaxe fallback, and an LLM picks the one edit per
  complaint, copied verbatim. Expired transcripts and editless lookback windows
  persist free `no_code` sentinel rows (`pair_index=-1`) with no LLM call.
  Evidence lands in the `pair_evidence` table keyed to the refine generation it
  annotates, `UNIQUE(dedup_key, refine_version, refine_model, pair_index,
  enrich_version, enrich_model, extractor_version)`, so a refine re-run or an
  `EXTRACTOR_VERSION` bump re-derives automatically; the `refined_pairs` view
  and the lineage detail carry each pair's latest evidence.
- Evidence surfaces in the UI. Dashboard pair cards gain a collapsible
  before/after diff for evidence-grounded pairs, with compact incorrect/correct
  panes that tint old lines `del` and new lines `ins`, a file chip, and a `git`
  chip when the correction came from git history; all of it is omitted entirely
  for pairs without code evidence. The lineage detail's refiner stage renders
  the same panes with the full, untruncated diff.
- `migrate-corpus` command for one-time, idempotent conversion of a pre-2.0
  database. Legacy `context_json` snapshots become `cc-transcript.context/1`
  documents carrying previews only, summary fidelity, and `origin='migrated'`,
  and the `event_uuid` / `triage.fidelity` columns are added.
- `triage --refresh-summary` re-judges rows whose verdict was recorded at
  summary fidelity; a full-fidelity verdict replaces the summary one once the
  row's window hydrates again.
- `scan` command for idempotent, incremental collection of developer steering
  into a local SQLite feedback DB at `~/.cc-steer/feedback.db` from existing
  Claude Code transcripts. Four detectors cover transcript messages; plan
  reviews, both rejected `ExitPlanMode` plans and post-edit plan re-entries;
  interrupts and permission denials; and review-format comments spanning
  superset inline cites, conductor findings, and workstreams. Each row captures
  the surrounding conversational window. Multiple `--transcripts` roots,
  including rsync mirrors of remote corpora, and a `--full` re-mine are
  supported.
- `stats` and `list` inspection commands, plus `view-samples`, which renders the
  whole corpus into one HTML page with an optional `claude`-CLI narrative and
  serves it over a transient local `aiohttp` server.
- Built on `cc-transcript` 0.5 for transcript discovery, parsing, the declarative
  noise-filter spec, and the file-state store.
- Async-native throughout. The store (`aiosqlite`), discovery, transcript
  parsing, and the `claude` shell-out all run on `anyio`; Click commands bridge
  to the async core via `anyio.run`.

[Unreleased]: https://github.com/yasyf/cc-steer/commits/main
