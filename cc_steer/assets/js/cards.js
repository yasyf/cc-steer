import { state } from "./state.js";
import { esc, chip, badge, diffPane } from "./dom.js";

export function evidenceHtml(ev) {
  if (!ev) return "";
  const git = ev.source === "git" ? '<span class="chip chip-git">git</span>' : "";
  const correct = ev.correct ? diffPane("correct", ev.correct) : "";
  return `<details class="diff"><summary>code evidence</summary>` +
    `<div class="vhead"><span class="chip">${esc(ev.file_path)}</span>${git}</div>` +
    `<div class="panes">${diffPane("incorrect", ev.incorrect)}${correct}</div></details>`;
}

function pairRow(r) {
  const file = r.evidence ? `<span class="chip">${esc(r.evidence.file_path)}</span>` : "";
  const lang = r.language ? `<span class="chip chip-lang">${esc(r.language)}</span>` : "";
  return `<article class="card" data-key="${esc(r.dedup_key)}"><header class="card-head">` +
    badge("cat-" + (r.category || "other"), r.category || "—") + badge("badge-" + r.source_kind, r.source_kind) +
    `${chip(r.project)}<span class="chip">pair ${r.pair_index}</span>${file}${lang}</header>` +
    `<div class="text"><pre>${esc(r.action)}</pre></div>` +
    `<blockquote class="pverbatim">${esc(r.direction_verbatim)}</blockquote>` +
    `<div class="direction">↳ ${esc(r.direction)}</div>${evidenceHtml(r.evidence)}</article>`;
}

function candRow(r) {
  const cat = r.category ? badge("cat-" + r.category, r.category) : "";
  const pc = r.pair_count ? `<span class="chip">${r.pair_count} pairs</span>` : "";
  const flip = r.flipped ? '<span class="flip">flip</span>' : "";
  const agree = r.agreement ? `<span class="${esc(r.agreement)}">${esc(r.agreement)}</span>` : "";
  const gold = r.golden ? badge(r.golden, "golden " + r.golden) : "";
  return `<article class="card" data-key="${esc(r.dedup_key)}"><header class="card-head">` +
    badge("st-" + r.status, r.status) + cat + badge("badge-" + r.source_kind, r.source_kind) +
    `${chip(r.project)}${pc}${flip}${agree}${gold}</header>` +
    `<div class="text"><pre>${esc(r.text)}</pre></div></article>`;
}

export function rowHtml(r) {
  return state.view === "pairs" ? pairRow(r) : candRow(r);
}
