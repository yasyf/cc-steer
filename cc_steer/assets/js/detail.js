import { state } from "./state.js";
import { lineageHtml } from "./lineage.js";

export function openDetail(key) {
  for (const c of state.cards) c.classList.toggle("sel", c.dataset.key === key);
  document.getElementById("detail").classList.add("open");
  document.getElementById("backdrop").classList.add("open");
  const body = document.getElementById("detail-body");
  body.innerHTML = '<p class="muted">loading…</p>';
  body.scrollTop = 0;
  fetch("/api/lineage/" + encodeURIComponent(key))
    .then((res) => res.ok ? res.json().then(lineageHtml) : '<p class="muted">no lineage</p>')
    .then((html) => { body.innerHTML = html; });
}

export function closeDetail() {
  document.getElementById("detail").classList.remove("open");
  document.getElementById("backdrop").classList.remove("open");
  for (const c of state.cards) c.classList.remove("sel");
}
