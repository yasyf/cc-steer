"""The FastAPI dashboard: a thin shell over a JSON API for exploring training pairs.

Serves the refined pairs (the pipeline's deliverable) and every candidate behind
them, and on demand renders one candidate's full lineage — detector hit, judge
verdicts across versions, the auditor's agreement, the refiner's atomic split, and
the golden gate. The data model and lineage HTML live in :mod:`cc_pushback.report`;
this module owns the routes, the JSON shapes, and the client shell.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from cc_pushback import report
from cc_pushback.evaluate import load_golden

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cc_pushback.evaluate import GoldenRow
    from cc_pushback.report import Summary
    from cc_pushback.store import FeedbackStore

LIST_TEXT_LIMIT = 280

DASHBOARD_CSS = """
.layout{display:flex;gap:16px;align-items:flex-start;padding:0 24px 24px}
#list{flex:1;min-width:0}
#detail{flex:1;min-width:0;position:sticky;top:60px;max-height:calc(100vh - 76px);overflow:auto}
.views{display:flex;gap:6px}
.view-btn{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:14px;
padding:4px 12px;cursor:pointer;font:inherit}
.view-btn.active{background:var(--accent);color:#0d1117;border-color:var(--accent)}
#status-filter{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:6px;
padding:5px 8px;font:inherit}
.card{cursor:pointer}
.card-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.complaint{color:var(--accent);margin-top:6px}
.st-refined{color:#7ee787}.st-accepted{color:#58a6ff}.st-noise{color:#8b949e}.st-unjudged{color:#6e7681}
"""

DASHBOARD_JS = """
const listEl=document.getElementById('list');
const detailEl=document.getElementById('detail');
const statsEl=document.getElementById('stats');
const searchEl=document.getElementById('search');
const countEl=document.getElementById('count');
const statusEl=document.getElementById('status-filter');
let view='pairs';
let rows=[];

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}
function chip(t){return t?`<span class="chip">${esc(t)}</span>`:'';}
function badge(cls,t){return `<span class="badge ${cls}">${esc(t)}</span>`;}

function attrs(r){
  return `data-kind="${esc(r.source_kind)}" data-status="${esc(r.status||'refined')}" `
    +`data-cat="${esc(r.category||'')}" data-flip="${r.flipped?'1':'0'}" `
    +`data-agree="${esc(r.agreement||'')}" data-golden="${esc(r.golden||'')}" data-key="${esc(r.dedup_key)}"`;
}

function pairRow(r){
  return `<article class="card" ${attrs(r)}><header class="card-head">`
    +badge('cat-'+(r.category||'other'),r.category||'—')+badge('badge-'+r.source_kind,r.source_kind)
    +`${chip(r.project)}<span class="chip">pair ${r.pair_index}</span></header>`
    +`<div class="text"><pre>${esc(r.action)}</pre></div>`
    +`<div class="complaint">↳ ${esc(r.complaint)}</div></article>`;
}

function candRow(r){
  const cat=r.category?badge('cat-'+r.category,r.category):'';
  const pc=r.pair_count?`<span class="chip">${r.pair_count} pairs</span>`:'';
  const flip=r.flipped?'<span class="flip">flip</span>':'';
  const agree=r.agreement?`<span class="${esc(r.agreement)}">${esc(r.agreement)}</span>`:'';
  const gold=r.golden?badge(r.golden,'golden '+r.golden):'';
  return `<article class="card" ${attrs(r)}><header class="card-head">`
    +badge('st-'+r.status,r.status)+cat+badge('badge-'+r.source_kind,r.source_kind)
    +`${chip(r.project)}${pc}${flip}${agree}${gold}</header>`
    +`<div class="text"><pre>${esc(r.text)}</pre></div></article>`;
}

function rowHtml(r){return view==='pairs'?pairRow(r):candRow(r);}

function apply(){
  const q=searchEl.value.trim().toLowerCase();
  const status=statusEl.value;
  const fFlip=document.getElementById('f-flip').checked;
  const fDis=document.getElementById('f-dis').checked;
  const fGold=document.getElementById('f-gold').checked;
  let shown=0;
  for(const c of listEl.querySelectorAll('.card')){
    const ok=(status==='all'||c.dataset.status===status)
      &&(!fFlip||c.dataset.flip==='1')
      &&(!fDis||c.dataset.agree==='disagree')
      &&(!fGold||c.dataset.golden!=='')
      &&(!q||c.textContent.toLowerCase().includes(q));
    c.style.display=ok?'':'none';if(ok)shown++;
  }
  countEl.textContent=shown+' / '+rows.length;
}

function render(){
  listEl.innerHTML=rows.map(rowHtml).join('')||'<p class="muted">none</p>';
  for(const c of listEl.querySelectorAll('.card'))c.addEventListener('click',()=>openDetail(c.dataset.key));
  apply();
}

async function openDetail(key){
  detailEl.innerHTML='<p class="muted">loading…</p>';
  const res=await fetch('/api/lineage/'+encodeURIComponent(key));
  detailEl.innerHTML=res.ok?(await res.json()).detail_html:'<p class="muted">no lineage</p>';
  detailEl.scrollIntoView({behavior:'smooth',block:'start'});
}

async function load(){
  const data=await (await fetch(view==='pairs'?'/api/pairs':'/api/candidates')).json();
  rows=view==='pairs'?data.pairs:data.candidates;
  render();
}

function statHtml(label,val){
  return `<div class="stat"><div class="n">${esc(val)}</div><div class="l">${esc(label)}</div></div>`;
}
async function loadStats(){
  const s=await (await fetch('/api/stats')).json();
  const p=s.pipeline,c=s.corpus;
  const cards=[['events',c.total],['accepted',p.accepted],['refined',p.refined],['pending',p.pending],
    ['atomic pairs',p.total_pairs],['pairs/event',p.pairs_per_event.toFixed(2)],['noise',p.noise_judged],
    ['unjudged',p.unjudged],['audited',p.audited],['disagree',p.disagree],['flips',p.flips],
    ['golden',p.golden_pass+'/'+p.golden_total]];
  statsEl.innerHTML=`<div class="stat-cards">${cards.map(x=>statHtml(x[0],x[1])).join('')}</div>`
    +(s.narrative?`<div class="narrative">${esc(s.narrative)}</div>`:'');
}

for(const b of document.querySelectorAll('.view-btn'))b.addEventListener('click',()=>{
  view=b.dataset.view;
  for(const x of document.querySelectorAll('.view-btn'))x.classList.toggle('active',x===b);
  load();
});
searchEl.addEventListener('input',apply);
statusEl.addEventListener('change',apply);
for(const id of ['f-flip','f-dis','f-gold'])document.getElementById(id).addEventListener('change',apply);
loadStats();load();
"""

SHELL = "".join(
    [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>cc-pushback dashboard</title><style>",
        report.CSS,
        DASHBOARD_CSS,
        "</style></head><body>",
        '<header class="top"><h1>cc-pushback — training pairs</h1>',
        '<div class="sub">refined pairs &amp; their lineage</div></header>',
        '<section id="stats"></section>',
        '<section id="controls"><div class="views">',
        '<button class="view-btn active" data-view="pairs">refined pairs</button>',
        '<button class="view-btn" data-view="candidates">all candidates</button></div>',
        '<select id="status-filter"><option value="all">all status</option>',
        '<option value="refined">refined</option><option value="accepted">accepted</option>',
        '<option value="noise">noise</option><option value="unjudged">unjudged</option></select>',
        '<input id="search" type="search" placeholder="search…">',
        '<label class="noise"><input type="checkbox" id="f-flip"> flipped</label>',
        '<label class="noise"><input type="checkbox" id="f-dis"> disagreements</label>',
        '<label class="noise"><input type="checkbox" id="f-gold"> golden</label>',
        '<span id="count"></span></section>',
        '<div class="layout"><section id="list"></section>',
        '<aside id="detail"><p class="muted">select a row to see its lineage</p></aside></div>',
        "<script>",
        DASHBOARD_JS,
        "</script></body></html>",
    ]
)


def project_of(origin_path: object) -> str | None:
    return report.project_label(str(origin_path)) if origin_path else None


def serialize_pair(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "dedup_key": row["dedup_key"],
        "pair_index": row["pair_index"],
        "action": report.truncate(str(row["action"]), LIST_TEXT_LIMIT),
        "complaint": row["complaint"],
        "category": row["category"],
        "source_kind": row["source_kind"],
        "project": project_of(row["origin_path"]),
        "occurred_at": str(row["occurred_at"])[:19],
    }


def serialize_candidate(row: Mapping[str, object], golden_map: Mapping[str, GoldenRow]) -> dict[str, object]:
    key = str(row["dedup_key"])
    audited = row["auditor_is_pushback"] is not None and row["is_pushback"] is not None
    in_golden = key in golden_map and row["is_pushback"] is not None
    return {
        "dedup_key": key,
        "source_kind": row["source_kind"],
        "occurred_at": str(row["occurred_at"])[:19],
        "project": project_of(row["origin_path"]),
        "status": report.candidate_status(row),
        "category": row["category"],
        "confidence": row["confidence"],
        "pair_count": row["pair_count"],
        "flipped": bool(row["flipped"]),
        "agreement": ("agree" if bool(row["auditor_is_pushback"]) == bool(row["is_pushback"]) else "disagree")
        if audited
        else None,
        "golden": ("pass" if report.golden_label(row["is_pushback"]) == golden_map[key].expected else "fail")
        if in_golden
        else None,
        "text": report.truncate(str(row["text"]), LIST_TEXT_LIMIT),
    }


def build_app(store: FeedbackStore, *, summary: Summary) -> FastAPI:
    """Builds the dashboard app over an open store, serving the shell and JSON API.

    Args:
        store: The open feedback store the routes query live.
        summary: The corpus summary, built once; its narrative and highlights are
            cached and served at ``/api/stats``.

    Returns:
        The configured :class:`fastapi.FastAPI` application.
    """
    golden_map = {row.dedup_key: row for row in load_golden()}
    app = FastAPI(title="cc-pushback")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return SHELL

    @app.get("/api/pairs")
    async def api_pairs() -> dict[str, object]:
        return {"pairs": [serialize_pair(row) for row in await store.pairs()]}

    @app.get("/api/candidates")
    async def api_candidates() -> dict[str, object]:
        return {"candidates": [serialize_candidate(row, golden_map) for row in await store.candidates()]}

    @app.get("/api/lineage/{dedup_key}")
    async def api_lineage(dedup_key: str) -> dict[str, object]:
        if not (data := await store.lineage(dedup_key)):
            raise HTTPException(status_code=404, detail="unknown dedup_key")
        return {"detail_html": report.render_lineage_detail(report.Lineage.from_lineage(data), golden_map)}

    @app.get("/api/stats")
    async def api_stats() -> dict[str, object]:
        candidates = await store.candidates()
        return {
            "corpus": asdict(report.corpus_stats([report.Sample.from_row(row) for row in candidates])),
            "pipeline": asdict(report.pipeline_stats(candidates, golden_map=golden_map)),
            "narrative": summary.narrative,
            "highlights": [asdict(highlight) for highlight in summary.highlights],
        }

    return app
