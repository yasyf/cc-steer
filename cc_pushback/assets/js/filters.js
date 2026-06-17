import { state } from "./state.js";
import { esc, badge } from "./dom.js";

// Facet groups, ordered as they render in the sidebar. `views` gates each group
// to the view where it means something; list facets are multi-select with
// drill-down counts, toggle facets are booleans.
export const GROUPS = [
  { label: "Status", views: ["candidates"], list: { key: "status", get: (r) => r.status } },
  { label: "Category", views: ["pairs", "candidates"], list: { key: "category", get: (r) => r.category, badge: "cat" } },
  { label: "Kind", views: ["pairs", "candidates"], list: { key: "source_kind", get: (r) => r.source_kind, badge: "kind" } },
  { label: "Project", views: ["pairs", "candidates"], list: { key: "project", get: (r) => r.project } },
  { label: "Language", views: ["pairs"], list: { key: "language", get: (r) => r.language } },
  { label: "Evidence", views: ["pairs"], toggles: [{ key: "evidence", text: "has code", match: (r) => !!r.evidence }] },
  { label: "Quality", views: ["candidates"], toggles: [
    { key: "golden", text: "golden", match: (r) => !!r.golden },
    { key: "flipped", text: "flipped", match: (r) => !!r.flipped },
    { key: "disagree", text: "disagreements", match: (r) => r.agreement === "disagree" },
  ] },
];

export function rowText(r) {
  const parts = [];
  const walk = (v) => {
    if (v == null) return;
    if (typeof v === "object") Object.values(v).forEach(walk);
    else parts.push(String(v));
  };
  walk(r);
  return parts.join(" ").toLowerCase();
}

function groupsFor(view) {
  return GROUPS.filter((g) => g.views.includes(view));
}

function matchRow(r, except) {
  for (const g of groupsFor(state.view)) {
    if (g.list) {
      if (g.list.key !== except) {
        const sel = state.picks[g.list.key];
        if (sel && sel.size && !sel.has(g.list.get(r))) return false;
      }
    } else {
      for (const t of g.toggles) {
        if (t.key !== except && state.flags[t.key] && !t.match(r)) return false;
      }
    }
  }
  return !state.q || r._text.includes(state.q);
}

function listCounts(facet) {
  const m = new Map();
  for (const r of state.rows) {
    if (matchRow(r, facet.key)) {
      const v = facet.get(r);
      if (v != null) m.set(v, (m.get(v) || 0) + 1);
    }
  }
  return m;
}

function facetGroupHtml(g) {
  if (g.list) {
    const counts = listCounts(g.list);
    const sel = state.picks[g.list.key] || new Set();
    const vals = [...new Set(state.rows.map(g.list.get).filter((v) => v != null))].sort();
    if (!vals.length) return "";
    const body = vals.map((v) => {
      const n = counts.get(v) || 0;
      const on = sel.has(v);
      const label = g.list.badge === "cat" ? badge("cat-" + v, v)
        : g.list.badge === "kind" ? badge("badge-" + v, v) : `<span class="fv">${esc(v)}</span>`;
      return `<button class="facet-row${on ? " on" : ""}${n ? "" : " empty"}" data-facet="${esc(g.list.key)}" ` +
        `data-value="${esc(v)}"><span class="fcheck">${on ? "✓" : ""}</span>${label}` +
        `<span class="facet-count">${n}</span></button>`;
    }).join("");
    return `<div class="facet-group"><h3>${esc(g.label)}</h3>${body}</div>`;
  }
  const body = g.toggles.map((t) => {
    const n = state.rows.filter((r) => matchRow(r, t.key) && t.match(r)).length;
    const on = !!state.flags[t.key];
    return `<button class="facet-row${on ? " on" : ""}${n ? "" : " empty"}" data-toggle="${esc(t.key)}">` +
      `<span class="fcheck">${on ? "✓" : ""}</span><span class="fv">${esc(t.text)}</span>` +
      `<span class="facet-count">${n}</span></button>`;
  }).join("");
  return `<div class="facet-group"><h3>${esc(g.label)}</h3>${body}</div>`;
}

function renderFacets() {
  document.getElementById("filters").innerHTML = groupsFor(state.view).map(facetGroupHtml).join("");
}

function chipsHtml() {
  const items = [];
  for (const g of groupsFor(state.view)) {
    if (g.list) for (const v of state.picks[g.list.key] || []) items.push([g.list.key, v, v]);
    else for (const t of g.toggles) if (state.flags[t.key]) items.push(["@" + t.key, t.key, t.text]);
  }
  if (!items.length) return "";
  return items.map(([k, v, label]) => `<button class="achip" data-k="${esc(k)}" data-v="${esc(v)}">${esc(label)} ✕</button>`).join("") +
    '<button class="achip clear" data-clear="1">clear all ✕</button>';
}

export function apply() {
  state.q = document.getElementById("search").value.trim().toLowerCase();
  let shown = 0;
  state.rows.forEach((r, i) => {
    const ok = matchRow(r);
    state.cards[i].style.display = ok ? "" : "none";
    if (ok) shown++;
  });
  document.getElementById("count").textContent = shown + " / " + state.rows.length;
  renderFacets();
  document.getElementById("active").innerHTML = chipsHtml();
}
