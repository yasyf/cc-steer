import { esc, diffPane } from "./dom.js";

// Renders one candidate's pipeline trail as a five-stage rail from the JSON the
// `/api/lineage/{key}` endpoint returns. The markup and class names mirror the
// shared stylesheet (base.css) exactly.

function goldenLabel(isPushback) {
  return isPushback ? "pushback" : "noise";
}

function metaChips(meta) {
  return meta.map((m) => `<span class="chip">${esc(m)}</span>`).join("");
}

function turnHtml(t) {
  const cls = "turn turn-" + t.role + (t.is_trigger ? " turn-trigger" : "");
  const tools = t.tool_calls ? `<span class="tools">${t.tool_calls} tool calls</span>` : "";
  return `<div class="${cls}"><span class="role">${esc(t.role)}</span>${tools}<pre>${esc(t.preview)}</pre></div>`;
}

function contextHtml(ctx) {
  if (!ctx) return "";
  return `<details class="ctx"><summary>context (${ctx.turns.length} turns)</summary>${ctx.turns.map(turnHtml).join("")}</details>`;
}

function verdictHtml(v) {
  const flag = v.flipped ? '<span class="flip">flipped across versions</span>' : "";
  return `<div class="verdict stage-${esc(v.role)}"><div class="vhead">` +
    `<span class="badge cat-${esc(v.category)}">${esc(v.category)}</span>` +
    `<span class="chip">${esc(v.role)} v${v.prompt_version} · ${esc(v.model)}</span>` +
    `<span class="chip">conf ${v.confidence.toFixed(2)}</span>` +
    `<span class="chip">${goldenLabel(v.is_pushback)}</span>${flag}</div>` +
    `<pre class="vsum">${esc(v.what_claude_did)}</pre>` +
    `<pre class="vrat">${esc(v.rationale)}</pre></div>`;
}

function auditorHtml(a) {
  if (!a) return '<p class="muted">not audited</p>';
  return verdictHtml(a) + `<span class="${esc(a.agreement)}">${esc(a.agreement)} with judge</span>`;
}

function evidenceHtml(ev) {
  const git = ev.source === "git" ? '<span class="chip chip-git">git</span>' : "";
  const correct = ev.correct ? diffPane("correct", ev.correct) : "";
  return `<div class="evidence"><div class="vhead"><span class="chip">${esc(ev.file_path)}</span>${git}</div>` +
    `<div class="panes">${diffPane("incorrect", ev.incorrect)}${correct}</div></div>`;
}

function highlightSpans(text, spans) {
  let out = esc(text);
  for (const span of spans) out = out.split(esc(span)).join(`<mark>${esc(span)}</mark>`);
  return out;
}

function refinerHtml(refiner) {
  if (!refiner.pairs.length) return '<p class="muted">not yet refined</p>';
  const cards = refiner.pairs.map((p) =>
    `<div class="pair"><div class="vhead"><span class="chip">pair ${p.pair_index}</span>` +
    `<span class="chip">v${p.prompt_version} · ${esc(p.model)}</span></div>` +
    `<pre class="paction">${esc(p.action)}</pre>` +
    `<blockquote class="pverbatim">${esc(p.complaint_verbatim)}</blockquote>` +
    `<pre class="pcomplaint">${esc(p.complaint)}</pre>` +
    `${p.evidence ? evidenceHtml(p.evidence) : ""}</div>`).join("");
  return `<div class="orig"><pre>${highlightSpans(refiner.original, refiner.spans)}</pre></div>${cards}`;
}

function goldenHtml(g) {
  if (!g) return '<p class="muted">not in golden set</p>';
  return `<span class="badge ${esc(g.verdict)}">golden ${esc(g.verdict)} · expected ${esc(goldenLabel(g.expected))}</span>`;
}

export function lineageHtml(d) {
  const det = d.detector;
  return `<div class="lineage">` +
    `<section class="stage stage-detector"><h3>1 · detector</h3>` +
    `<header class="card-head"><span class="badge badge-${esc(det.source_kind)}">${esc(det.source_kind)}</span>` +
    `<time>${esc(det.occurred_at)}</time>${metaChips(det.meta)}</header>` +
    `<div class="text"><pre>${esc(det.text)}</pre></div>${contextHtml(det.context)}</section>` +
    `<section class="stage stage-judge"><h3>2 · judge</h3>${d.judge.map(verdictHtml).join("") || '<p class="muted">unjudged</p>'}</section>` +
    `<section class="stage stage-auditor"><h3>3 · auditor</h3>${auditorHtml(d.auditor)}</section>` +
    `<section class="stage stage-refiner"><h3>4 · refiner — atomic pairs</h3>${refinerHtml(d.refiner)}</section>` +
    `<section class="stage stage-golden"><h3>5 · golden gate</h3>${goldenHtml(d.golden)}</section>` +
    `</div>`;
}
