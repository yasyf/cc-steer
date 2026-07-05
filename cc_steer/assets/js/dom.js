// Small HTML-building helpers shared across the renderers. esc() is the single
// escape point — every value interpolated into markup must pass through it.
export function esc(s) {
  return (s == null ? "" : String(s))
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#x27;");
}

export function chip(t) {
  return t ? `<span class="chip">${esc(t)}</span>` : "";
}

export function badge(cls, t) {
  return `<span class="badge ${cls}">${esc(t)}</span>`;
}

function diffLines(cls, text) {
  return text.split("\n").map((l) => `<div class="${cls}">${esc(l)}</div>`).join("");
}

export function diffPane(label, side) {
  return `<div class="pane"><div class="plabel">${esc(label)}</div>` +
    diffLines("del", side.old) + diffLines("ins", side.new) + "</div>";
}
