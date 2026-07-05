import { state } from "./state.js";
import { rowHtml } from "./cards.js";
import { apply, rowText } from "./filters.js";
import { loadStats } from "./stats.js";
import { openDetail, closeDetail } from "./detail.js";

function render() {
  const listEl = document.getElementById("list");
  listEl.innerHTML = state.rows.map(rowHtml).join("") || '<p class="muted">none</p>';
  state.cards = [...listEl.querySelectorAll(".card")];
  for (const c of state.cards) c.addEventListener("click", () => openDetail(c.dataset.key));
  for (const s of listEl.querySelectorAll("details.diff summary")) s.addEventListener("click", (e) => e.stopPropagation());
  apply();
}

async function load() {
  const data = await (await fetch(state.view === "pairs" ? "/api/pairs" : "/api/candidates")).json();
  state.rows = state.view === "pairs" ? data.pairs : data.candidates;
  state.rows.forEach((r) => { r._text = rowText(r); });
  state.picks = {};
  state.flags = {};
  document.getElementById("search").value = "";
  render();
}

for (const b of document.querySelectorAll(".view-btn")) b.addEventListener("click", () => {
  state.view = b.dataset.view;
  for (const x of document.querySelectorAll(".view-btn")) x.classList.toggle("active", x === b);
  closeDetail();
  load();
});

document.getElementById("search").addEventListener("input", apply);

document.getElementById("filters").addEventListener("click", (e) => {
  const row = e.target.closest(".facet-row");
  if (!row) return;
  if (row.dataset.facet) {
    const set = state.picks[row.dataset.facet] || (state.picks[row.dataset.facet] = new Set());
    set.has(row.dataset.value) ? set.delete(row.dataset.value) : set.add(row.dataset.value);
  } else {
    state.flags[row.dataset.toggle] = !state.flags[row.dataset.toggle];
  }
  apply();
});

document.getElementById("active").addEventListener("click", (e) => {
  const c = e.target.closest(".achip");
  if (!c) return;
  if (c.dataset.clear) { state.picks = {}; state.flags = {}; }
  else if (c.dataset.k[0] === "@") state.flags[c.dataset.k.slice(1)] = false;
  else state.picks[c.dataset.k].delete(c.dataset.v);
  apply();
});

document.getElementById("backdrop").addEventListener("click", closeDetail);
document.getElementById("detail-close").addEventListener("click", closeDetail);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDetail(); });
document.getElementById("stats-toggle").addEventListener("click", (e) => {
  e.target.textContent = document.getElementById("stats").classList.toggle("hidden") ? "stats ▾" : "stats ▴";
});

loadStats();
load();
