import { esc } from "./dom.js";

function statHtml(label, val) {
  return `<div class="stat"><div class="n">${esc(val)}</div><div class="l">${esc(label)}</div></div>`;
}

function compHtml(comp) {
  const cats = Object.keys(comp);
  if (!cats.length) return "";
  const kinds = [...new Set(cats.flatMap((c) => Object.keys(comp[c])))].sort();
  const max = Math.max(1, ...cats.flatMap((c) => kinds.map((k) => comp[c][k] || 0)));
  const head = `<tr><td></td>${kinds.map((k) => `<td>${esc(k)}</td>`).join("")}<td>total</td></tr>`;
  const body = cats.map((c) => {
    const tot = kinds.reduce((a, k) => a + (comp[c][k] || 0), 0);
    const cells = kinds.map((k) => {
      const n = comp[c][k] || 0;
      return `<td>${n ? `<span class="bar" style="width:${Math.round(40 * n / max)}px"></span> ${n}` : "·"}</td>`;
    }).join("");
    return `<tr><td><span class="badge cat-${esc(c)}">${esc(c)}</span></td>${cells}<td>${tot}</td></tr>`;
  }).join("");
  return `<div class="comp"><h2>composition · accepted by category × kind</h2>` +
    `<table class="dist"><tbody>${head}${body}</tbody></table></div>`;
}

export async function loadStats() {
  const s = await (await fetch("/api/stats")).json();
  const p = s.pipeline, c = s.corpus;
  const statCards = [["events", c.total], ["accepted", p.accepted], ["refined", p.refined], ["pending", p.pending],
    ["atomic pairs", p.total_pairs], ["pairs/event", p.pairs_per_event.toFixed(2)], ["noise", p.noise_judged],
    ["unjudged", p.unjudged], ["audited", p.audited], ["disagree", p.disagree], ["flips", p.flips],
    ["golden", p.golden_pass + "/" + p.golden_total]];
  document.getElementById("stats").innerHTML = `<div class="stat-cards">${statCards.map((x) => statHtml(x[0], x[1])).join("")}</div>` +
    compHtml(p.by_category_kind) +
    (s.narrative ? `<div class="narrative">${esc(s.narrative)}</div>` : "");
  document.getElementById("stat-strip").textContent =
    `${c.total} events · ${p.accepted} accepted · ${p.refined} refined · ${p.total_pairs} pairs`;
}
