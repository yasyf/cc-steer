"""The FastAPI dashboard: a thin shell over a JSON API for exploring training pairs.

Serves the refined pairs (the pipeline's deliverable) and every candidate behind
them, and on demand renders one candidate's full lineage — detector hit, judge
verdicts across versions, the auditor's agreement, the refiner's atomic split, and
the golden gate. The data model and lineage HTML live in :mod:`cc_pushback.report`;
this module owns the routes, the JSON shapes, and the client shell.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
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
header.top{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}
.head-right{display:flex;align-items:center;gap:12px}
#stat-strip{color:var(--muted);font-size:12px;text-align:right}
#stats-toggle{background:var(--panel);color:var(--fg);border:1px solid var(--border);
border-radius:8px;padding:4px 12px;cursor:pointer;font:inherit}
#stats.hidden{display:none}
.comp h2{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:18px 0 0}
.comp table.dist td{padding:3px 12px 3px 0}
.app{display:flex;align-items:flex-start}
#filters{width:236px;flex:none;position:sticky;top:0;max-height:100vh;overflow:auto;
padding:16px;border-right:1px solid var(--border)}
.facet-group{margin-bottom:18px}
.facet-group h3{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:0 0 8px}
.facet-row{display:flex;align-items:center;gap:8px;width:100%;text-align:left;background:none;
border:0;color:var(--fg);font:inherit;padding:4px 6px;border-radius:6px;cursor:pointer}
.facet-row:hover{background:var(--panel)}
.facet-row.on{background:var(--panel);box-shadow:inset 2px 0 0 var(--accent)}
.facet-row.empty{opacity:.4}
.facet-row .fcheck{width:12px;color:var(--accent)}
.facet-row .fv{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:130px}
.facet-count{margin-left:auto;color:var(--muted);font-size:11px}
main{flex:1;min-width:0;padding:0 24px 48px}
#toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;position:sticky;top:0;
background:var(--bg);padding:14px 0;z-index:2;border-bottom:1px solid var(--border)}
.views{display:flex;gap:6px}
.view-btn{background:var(--panel);color:var(--fg);border:1px solid var(--border);border-radius:14px;
padding:4px 12px;cursor:pointer;font:inherit}
.view-btn.active{background:var(--accent);color:#0d1117;border-color:var(--accent)}
#count{color:var(--muted)}
#active{display:flex;flex-wrap:wrap;gap:6px;padding:10px 0}
#active:empty{display:none}
.achip{background:var(--panel);border:1px solid var(--border);border-radius:12px;color:var(--fg);
font:inherit;font-size:12px;padding:2px 10px;cursor:pointer}
.achip:hover{border-color:var(--accent)}
.achip.clear{color:var(--muted)}
.card{cursor:pointer}
.card:hover{border-color:var(--accent)}
.card.sel{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.card-head{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
.complaint{color:var(--accent);margin-top:6px}
.chip-lang{color:var(--accent);border:1px solid var(--accent)}
.st-refined{color:#7ee787}.st-accepted{color:#58a6ff}.st-noise{color:#8b949e}.st-unjudged{color:#6e7681}
details.diff{margin-top:8px}
details.diff summary{color:var(--accent);cursor:pointer}
#backdrop{position:fixed;inset:0;background:#00000066;opacity:0;pointer-events:none;
transition:opacity .22s ease;z-index:10}
#backdrop.open{opacity:1;pointer-events:auto}
#detail{position:fixed;top:0;right:0;height:100vh;width:min(760px,92vw);background:var(--bg);
border-left:1px solid var(--border);transform:translateX(100%);transition:transform .22s ease;
z-index:20;overflow:auto;box-shadow:-16px 0 40px #00000066}
#detail.open{transform:translateX(0)}
#detail-bar{position:sticky;top:0;display:flex;justify-content:flex-end;padding:8px;
background:var(--bg);border-bottom:1px solid var(--border)}
#detail-close{background:var(--panel);color:var(--fg);border:1px solid var(--border);
border-radius:8px;width:30px;height:30px;cursor:pointer;font:inherit}
#detail-body{padding:16px 24px}
"""

DASHBOARD_JS = """
const filtersEl=document.getElementById('filters');
const listEl=document.getElementById('list');
const detailEl=document.getElementById('detail');
const detailBodyEl=document.getElementById('detail-body');
const backdropEl=document.getElementById('backdrop');
const statsEl=document.getElementById('stats');
const statStripEl=document.getElementById('stat-strip');
const searchEl=document.getElementById('search');
const countEl=document.getElementById('count');
const activeEl=document.getElementById('active');
let view='pairs';
let rows=[];
let cards=[];
let picks={};
let flags={};
let q='';

const GROUPS=[
  {label:'Status',views:['candidates'],list:{key:'status',get:r=>r.status}},
  {label:'Category',views:['pairs','candidates'],list:{key:'category',get:r=>r.category,badge:'cat'}},
  {label:'Kind',views:['pairs','candidates'],list:{key:'source_kind',get:r=>r.source_kind,badge:'kind'}},
  {label:'Project',views:['pairs','candidates'],list:{key:'project',get:r=>r.project}},
  {label:'Language',views:['pairs'],list:{key:'language',get:r=>r.language}},
  {label:'Evidence',views:['pairs'],toggles:[{key:'evidence',text:'has code',match:r=>!!r.evidence}]},
  {label:'Quality',views:['candidates'],toggles:[
    {key:'golden',text:'golden',match:r=>!!r.golden},
    {key:'flipped',text:'flipped',match:r=>!!r.flipped},
    {key:'disagree',text:'disagreements',match:r=>r.agreement==='disagree'},
  ]},
];

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML;}
function chip(t){return t?`<span class="chip">${esc(t)}</span>`:'';}
function badge(cls,t){return `<span class="badge ${cls}">${esc(t)}</span>`;}

function diffLines(cls,text){
  return text.split('\\n').map(l=>`<div class="${cls}">${esc(l)}</div>`).join('');
}
function diffPane(label,side){
  return `<div class="pane"><div class="plabel">${esc(label)}</div>`
    +diffLines('del',side.old)+diffLines('ins',side.new)+'</div>';
}
function evidenceHtml(ev){
  if(!ev)return '';
  const git=ev.source==='git'?'<span class="chip chip-git">git</span>':'';
  const correct=ev.correct?diffPane('correct',ev.correct):'';
  return `<details class="diff"><summary>code evidence</summary>`
    +`<div class="vhead"><span class="chip">${esc(ev.file_path)}</span>${git}</div>`
    +`<div class="panes">${diffPane('incorrect',ev.incorrect)}${correct}</div></details>`;
}

function pairRow(r){
  const file=r.evidence?`<span class="chip">${esc(r.evidence.file_path)}</span>`:'';
  const lang=r.language?`<span class="chip chip-lang">${esc(r.language)}</span>`:'';
  return `<article class="card" data-key="${esc(r.dedup_key)}"><header class="card-head">`
    +badge('cat-'+(r.category||'other'),r.category||'—')+badge('badge-'+r.source_kind,r.source_kind)
    +`${chip(r.project)}<span class="chip">pair ${r.pair_index}</span>${file}${lang}</header>`
    +`<div class="text"><pre>${esc(r.action)}</pre></div>`
    +`<blockquote class="pverbatim">${esc(r.complaint_verbatim)}</blockquote>`
    +`<div class="complaint">↳ ${esc(r.complaint)}</div>${evidenceHtml(r.evidence)}</article>`;
}

function candRow(r){
  const cat=r.category?badge('cat-'+r.category,r.category):'';
  const pc=r.pair_count?`<span class="chip">${r.pair_count} pairs</span>`:'';
  const flip=r.flipped?'<span class="flip">flip</span>':'';
  const agree=r.agreement?`<span class="${esc(r.agreement)}">${esc(r.agreement)}</span>`:'';
  const gold=r.golden?badge(r.golden,'golden '+r.golden):'';
  return `<article class="card" data-key="${esc(r.dedup_key)}"><header class="card-head">`
    +badge('st-'+r.status,r.status)+cat+badge('badge-'+r.source_kind,r.source_kind)
    +`${chip(r.project)}${pc}${flip}${agree}${gold}</header>`
    +`<div class="text"><pre>${esc(r.text)}</pre></div></article>`;
}

function rowHtml(r){return view==='pairs'?pairRow(r):candRow(r);}

function rowText(r){
  const parts=[];
  const walk=v=>{if(v==null)return;if(typeof v==='object')Object.values(v).forEach(walk);else parts.push(String(v));};
  walk(r);
  return parts.join(' ').toLowerCase();
}

function groupsFor(v){return GROUPS.filter(g=>g.views.includes(v));}

function matchRow(r,except){
  for(const g of groupsFor(view)){
    if(g.list){
      if(g.list.key!==except){
        const sel=picks[g.list.key];
        if(sel&&sel.size&&!sel.has(g.list.get(r)))return false;
      }
    }else for(const t of g.toggles){
      if(t.key!==except&&flags[t.key]&&!t.match(r))return false;
    }
  }
  return !q||r._text.includes(q);
}

function listCounts(facet){
  const m=new Map();
  for(const r of rows)if(matchRow(r,facet.key)){const v=facet.get(r);if(v!=null)m.set(v,(m.get(v)||0)+1);}
  return m;
}

function facetGroupHtml(g){
  if(g.list){
    const counts=listCounts(g.list);
    const sel=picks[g.list.key]||new Set();
    const vals=[...new Set(rows.map(g.list.get).filter(v=>v!=null))].sort();
    if(!vals.length)return '';
    const body=vals.map(v=>{
      const n=counts.get(v)||0,on=sel.has(v);
      const label=g.list.badge==='cat'?badge('cat-'+v,v)
        :g.list.badge==='kind'?badge('badge-'+v,v):`<span class="fv">${esc(v)}</span>`;
      return `<button class="facet-row${on?' on':''}${n?'':' empty'}" data-facet="${esc(g.list.key)}" `
        +`data-value="${esc(v)}"><span class="fcheck">${on?'✓':''}</span>${label}`
        +`<span class="facet-count">${n}</span></button>`;
    }).join('');
    return `<div class="facet-group"><h3>${esc(g.label)}</h3>${body}</div>`;
  }
  const body=g.toggles.map(t=>{
    const n=rows.filter(r=>matchRow(r,t.key)&&t.match(r)).length,on=!!flags[t.key];
    return `<button class="facet-row${on?' on':''}${n?'':' empty'}" data-toggle="${esc(t.key)}">`
      +`<span class="fcheck">${on?'✓':''}</span><span class="fv">${esc(t.text)}</span>`
      +`<span class="facet-count">${n}</span></button>`;
  }).join('');
  return `<div class="facet-group"><h3>${esc(g.label)}</h3>${body}</div>`;
}

function renderFacets(){
  filtersEl.innerHTML=groupsFor(view).map(facetGroupHtml).join('');
}

function chipsHtml(){
  const items=[];
  for(const g of groupsFor(view)){
    if(g.list)for(const v of picks[g.list.key]||[])items.push([g.list.key,v,v]);
    else for(const t of g.toggles)if(flags[t.key])items.push(['@'+t.key,t.key,t.text]);
  }
  if(!items.length)return '';
  return items.map(([k,v,label])=>`<button class="achip" data-k="${esc(k)}" data-v="${esc(v)}">${esc(label)} ✕</button>`).join('')
    +'<button class="achip clear" data-clear="1">clear all ✕</button>';
}

function apply(){
  q=searchEl.value.trim().toLowerCase();
  let shown=0;
  rows.forEach((r,i)=>{const ok=matchRow(r);cards[i].style.display=ok?'':'none';if(ok)shown++;});
  countEl.textContent=shown+' / '+rows.length;
  renderFacets();
  activeEl.innerHTML=chipsHtml();
}

function render(){
  listEl.innerHTML=rows.map(rowHtml).join('')||'<p class="muted">none</p>';
  cards=[...listEl.querySelectorAll('.card')];
  for(const c of cards)c.addEventListener('click',()=>openDetail(c.dataset.key));
  for(const s of listEl.querySelectorAll('details.diff summary'))s.addEventListener('click',e=>e.stopPropagation());
  apply();
}

function openDetail(key){
  for(const c of cards)c.classList.toggle('sel',c.dataset.key===key);
  detailEl.classList.add('open');backdropEl.classList.add('open');
  detailBodyEl.innerHTML='<p class="muted">loading…</p>';
  detailBodyEl.scrollTop=0;
  fetch('/api/lineage/'+encodeURIComponent(key))
    .then(res=>res.ok?res.json().then(d=>d.detail_html):'<p class="muted">no lineage</p>')
    .then(html=>{detailBodyEl.innerHTML=html;});
}

function closeDetail(){
  detailEl.classList.remove('open');backdropEl.classList.remove('open');
  for(const c of cards)c.classList.remove('sel');
}

async function load(){
  const data=await (await fetch(view==='pairs'?'/api/pairs':'/api/candidates')).json();
  rows=view==='pairs'?data.pairs:data.candidates;
  rows.forEach(r=>{r._text=rowText(r);});
  picks={};flags={};searchEl.value='';
  render();
}

function statHtml(label,val){
  return `<div class="stat"><div class="n">${esc(val)}</div><div class="l">${esc(label)}</div></div>`;
}
function compHtml(comp){
  const cats=Object.keys(comp);
  if(!cats.length)return '';
  const kinds=[...new Set(cats.flatMap(c=>Object.keys(comp[c])))].sort();
  const max=Math.max(1,...cats.flatMap(c=>kinds.map(k=>comp[c][k]||0)));
  const head=`<tr><td></td>${kinds.map(k=>`<td>${esc(k)}</td>`).join('')}<td>total</td></tr>`;
  const body=cats.map(c=>{
    const tot=kinds.reduce((a,k)=>a+(comp[c][k]||0),0);
    const cells=kinds.map(k=>{const n=comp[c][k]||0;
      return `<td>${n?`<span class="bar" style="width:${Math.round(40*n/max)}px"></span> ${n}`:'·'}</td>`;}).join('');
    return `<tr><td><span class="badge cat-${esc(c)}">${esc(c)}</span></td>${cells}<td>${tot}</td></tr>`;
  }).join('');
  return `<div class="comp"><h2>composition · accepted by category × kind</h2>`
    +`<table class="dist"><tbody>${head}${body}</tbody></table></div>`;
}
async function loadStats(){
  const s=await (await fetch('/api/stats')).json();
  const p=s.pipeline,c=s.corpus;
  const statCards=[['events',c.total],['accepted',p.accepted],['refined',p.refined],['pending',p.pending],
    ['atomic pairs',p.total_pairs],['pairs/event',p.pairs_per_event.toFixed(2)],['noise',p.noise_judged],
    ['unjudged',p.unjudged],['audited',p.audited],['disagree',p.disagree],['flips',p.flips],
    ['golden',p.golden_pass+'/'+p.golden_total]];
  statsEl.innerHTML=`<div class="stat-cards">${statCards.map(x=>statHtml(x[0],x[1])).join('')}</div>`
    +compHtml(p.by_category_kind)
    +(s.narrative?`<div class="narrative">${esc(s.narrative)}</div>`:'');
  statStripEl.textContent=`${c.total} events · ${p.accepted} accepted · ${p.refined} refined · ${p.total_pairs} pairs`;
}

for(const b of document.querySelectorAll('.view-btn'))b.addEventListener('click',()=>{
  view=b.dataset.view;
  for(const x of document.querySelectorAll('.view-btn'))x.classList.toggle('active',x===b);
  closeDetail();load();
});
searchEl.addEventListener('input',apply);
filtersEl.addEventListener('click',e=>{
  const row=e.target.closest('.facet-row');
  if(!row)return;
  if(row.dataset.facet){
    const set=picks[row.dataset.facet]||(picks[row.dataset.facet]=new Set());
    set.has(row.dataset.value)?set.delete(row.dataset.value):set.add(row.dataset.value);
  }else flags[row.dataset.toggle]=!flags[row.dataset.toggle];
  apply();
});
activeEl.addEventListener('click',e=>{
  const c=e.target.closest('.achip');
  if(!c)return;
  if(c.dataset.clear){picks={};flags={};}
  else if(c.dataset.k[0]==='@')flags[c.dataset.k.slice(1)]=false;
  else picks[c.dataset.k].delete(c.dataset.v);
  apply();
});
backdropEl.addEventListener('click',closeDetail);
document.getElementById('detail-close').addEventListener('click',closeDetail);
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeDetail();});
document.getElementById('stats-toggle').addEventListener('click',e=>{
  e.target.textContent=statsEl.classList.toggle('hidden')?'stats ▾':'stats ▴';
});
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
        '<header class="top"><div><h1>cc-pushback — training pairs</h1>',
        '<div class="sub">refined pairs &amp; their lineage</div></div>',
        '<div class="head-right"><span id="stat-strip"></span>',
        '<button id="stats-toggle">stats ▾</button></div></header>',
        '<section id="stats" class="hidden"></section>',
        '<div class="app"><aside id="filters"></aside><main>',
        '<div id="toolbar"><div class="views">',
        '<button class="view-btn active" data-view="pairs">refined pairs</button>',
        '<button class="view-btn" data-view="candidates">all candidates</button></div>',
        '<input id="search" type="search" placeholder="search…"><span id="count"></span></div>',
        '<div id="active"></div><div id="list"></div></main></div>',
        '<div id="backdrop"></div>',
        '<aside id="detail"><div id="detail-bar"><button id="detail-close">✕</button></div>',
        '<div id="detail-body"><p class="muted">select a row to see its lineage</p></div></aside>',
        "<script>",
        DASHBOARD_JS,
        "</script></body></html>",
    ]
)


def project_of(origin_path: object) -> str | None:
    return report.project_label(str(origin_path)) if origin_path else None


def language_of(file_path: str | None) -> str | None:
    if not file_path:
        return None
    return (Path(file_path).suffix.lstrip(".") or Path(file_path).name).lower()


def edit_json(old: str, new: str) -> dict[str, str]:
    return {"old": report.truncate(old, LIST_TEXT_LIMIT), "new": report.truncate(new, LIST_TEXT_LIMIT)}


def serialize_evidence(row: Mapping[str, object]) -> dict[str, object] | None:
    if (evidence := report.EvidenceRow.from_row(row)) is None:
        return None
    return {
        "file_path": evidence.file_path,
        "source": evidence.source,
        "incorrect": edit_json(*evidence.incorrect),
        "correct": None if evidence.correct is None else edit_json(*evidence.correct),
    }


def serialize_pair(row: Mapping[str, object]) -> dict[str, object]:
    evidence = serialize_evidence(row)
    return {
        "dedup_key": row["dedup_key"],
        "pair_index": row["pair_index"],
        "action": report.truncate(str(row["action"]), LIST_TEXT_LIMIT),
        "complaint_verbatim": report.truncate(str(row["complaint_verbatim"]), LIST_TEXT_LIMIT),
        "complaint": row["complaint"],
        "category": row["category"],
        "source_kind": row["source_kind"],
        "project": project_of(row["origin_path"]),
        "occurred_at": str(row["occurred_at"])[:19],
        "evidence": evidence,
        "language": language_of(str(evidence["file_path"])) if evidence else None,
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
        "golden": ("pass" if bool(row["is_pushback"]) == golden_map[key].expected else "fail") if in_golden else None,
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
