"""Offline review-page JavaScript and lossless repair helpers."""
from __future__ import annotations

import re
import json

CSV_FIELDS = [
    "source_id", "source_image_sha256", "group_id", "near_duplicate_group_id",
    "classes_present", "polygon_ids", "selected_polygon_ids", "image_width",
    "image_height", "historical_cloud_exposure", "repair_priority",
    "user_decision", "user_notes", "trusted_training_pair_status",
]


def render_script(source_count: int, storage_key: str = "exp13-g3b1-review") -> str:
    return f'''const KEY={json.dumps(storage_key)}, cards=[...document.querySelectorAll('.source')], classFilter=document.querySelector('#classFilter'), search=document.querySelector('#sourceSearch'), count=document.querySelector('#count'), message=document.querySelector('#message');
const counts=document.querySelector('#reviewCounts'), decisions=['UNREVIEWED','SELECT','MAYBE','REJECT'];
function values(card){{return {{repair_priority:card.querySelector('.priority').value,user_decision:card.querySelector('.decision').value,user_notes:card.querySelector('.notes').value,selected_polygon_ids:[...card.querySelectorAll('.target-poly:checked')].map(x=>x.value)}}}}
function apply(card,state){{const decision=decisions.includes(state.user_decision)?state.user_decision:'UNREVIEWED';card.querySelector('.priority').value=['','A','B','C'].includes(state.repair_priority)?state.repair_priority:'';card.querySelector('.decision').value=decision;card.querySelector('.notes').value=state.user_notes||'';const selected=new Set(decision==='REJECT'||decision==='UNREVIEWED'?[]:(state.selected_polygon_ids||[]));for(const input of card.querySelectorAll('.target-poly'))input.checked=selected.has(input.value);card.dataset.decision=decision}}
function save(card){{const state=values(card);try{{const all=JSON.parse(localStorage.getItem(KEY)||'{{}}');all[card.dataset.source]=state;localStorage.setItem(KEY,JSON.stringify(all));message.textContent='Saved in this browser; download a CSV or JSON backup for a portable copy';}}catch(error){{message.textContent='State is active in this page; download a CSV or JSON backup for a durable copy';}}updateCounts()}}
function updateCounts(){{const totals={{UNREVIEWED:0,SELECT:0,MAYBE:0,REJECT:0}};for(const card of cards)totals[values(card).user_decision]++;const reviewed={source_count}-totals.UNREVIEWED;counts.textContent=`Reviewed ${{reviewed}} / {source_count} | SELECT ${{totals.SELECT}} | MAYBE ${{totals.MAYBE}} | REJECT ${{totals.REJECT}} | UNREVIEWED ${{totals.UNREVIEWED}}`;}}
function filter(){{const c=classFilter.value.toLowerCase(),s=search.value.toLowerCase();let n=0;for(const card of cards){{const show=(!c||card.dataset.classes.toLowerCase().split('|').includes(c))&&(!s||card.dataset.source.toLowerCase().includes(s));card.hidden=!show;if(show)n++}}count.textContent=`${{n}} / {source_count} sources`;}}
let saved={{}};try{{saved=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(error){{}}for(const card of cards){{apply(card,saved[card.dataset.source]||{{}});card.addEventListener('input',()=>save(card));card.addEventListener('change',()=>{{if(card.querySelector('.decision').value==='REJECT'){{const checked=card.querySelectorAll('.target-poly:checked');if(checked.length){{for(const input of checked)input.checked=false;message.textContent='REJECT clears previously selected Polygon IDs.'}}}}if(card.querySelector('.decision').value==='UNREVIEWED')for(const input of card.querySelectorAll('.target-poly:checked'))input.checked=false;save(card)}})}}
classFilter.onchange=search.oninput=filter;filter();updateCounts();
function parseCsv(text){{let rows=[],row=[],cell='',quoted=false;for(let i=0;i<text.length;i++){{const c=text[i];if(quoted){{if(c==='"'&&text[i+1]==='"'){{cell+='"';i++}}else if(c==='"')quoted=false;else cell+=c}}else if(c==='"')quoted=true;else if(c===','){{row.push(cell);cell=''}}else if(c==='\\n'){{row.push(cell.replace(/\\r$/,''));rows.push(row);row=[];cell=''}}else cell+=c}}if(cell.length||row.length){{row.push(cell.replace(/\\r$/,''));rows.push(row)}}const head=rows.shift()||[];if(head.length&&head[0].charCodeAt(0)===0xfeff)head[0]=head[0].slice(1);return rows.map(r=>Object.fromEntries(head.map((h,i)=>[h,r[i]||''])))}}
function importRows(imported){{const byId=new Map(),pending=[];for(const row of imported){{if(byId.has(row.source_id))throw Error('Duplicate source_id in import: '+row.source_id);byId.set(row.source_id,row)}}for(const card of cards){{const row=byId.get(card.dataset.source);if(!row)continue;const valid=new Set([...card.querySelectorAll('.target-poly')].map(x=>x.value)),selection=(row.selected_polygon_ids||'').split('|').filter(x=>x),decision=row.user_decision||'UNREVIEWED';if(!decisions.includes(decision))throw Error('Unknown decision for source '+card.dataset.source);if(selection.some(x=>!valid.has(x)))throw Error('Unknown Polygon ID for source '+card.dataset.source);if(decision==='SELECT'&&!selection.length)throw Error('SELECT requires a Polygon ID for source '+card.dataset.source);const state={{repair_priority:row.repair_priority||'',user_decision:decision,user_notes:row.user_notes||'',selected_polygon_ids:decision==='REJECT'||decision==='UNREVIEWED'?[]:selection}};pending.push([card,state])}}for(const [card,state] of pending){{apply(card,state);save(card)}}updateCounts();return pending.length}}
document.querySelector('#upload').onchange=async e=>{{const file=e.target.files[0];if(!file)return;try{{const used=importRows(parseCsv(await file.text()));message.textContent=`Imported ${{used}} source rows; decisions and target Polygons restored`;}}catch(error){{message.textContent='Import rejected: '+error.message}}}};
function download(name,text,type){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([text],{{type}}));a.download=name;a.click();URL.revokeObjectURL(a.href)}}
document.querySelector('#download').onclick=()=>{{const rows=[];for(const card of cards){{const source=card.dataset.source,base=window.__rows.find(x=>x.source_id===source),state=values(card);if(state.user_decision==='SELECT'&&state.selected_polygon_ids.length===0){{message.textContent=`SELECT requires at least one checked Polygon for source ${{source}}`;card.scrollIntoView();return}}if(state.user_decision==='REJECT'||state.user_decision==='UNREVIEWED')state.selected_polygon_ids=[];rows.push({{...base,...state,selected_polygon_ids:state.selected_polygon_ids.join('|')}})}}const fields=__CSV_FIELDS__,quote=v=>'"'+String(v??'').replaceAll('"','""')+'"',csv=String.fromCharCode(0xfeff)+[fields.join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\\r\\n');download('source_review_decisions.csv',csv,'text/csv;charset=utf-8');message.textContent=`CSV exported for ${{rows.length}} sources; incomplete sources remain UNREVIEWED`;}};
document.querySelector('#backup').onclick=()=>{{const backup=cards.map(card=>({{source_id:card.dataset.source,...values(card)}}));download('g3b1_review_emergency_backup.json',JSON.stringify(backup,null,2)+'\\n','application/json');message.textContent='Emergency JSON backup downloaded';}};
window.__restoreReviewRows=importRows;'''.replace("__CSV_FIELDS__", json.dumps(CSV_FIELDS))


def repair_html_document(source: str, source_count: int) -> str:
    header_old = '<span id="count"></span><button id="download">Download review CSV</button><label>Import reviewed CSV'
    header_new = '<span id="count"></span><span id="reviewCounts"></span><button id="download">Download review CSV</button><button id="backup">Download emergency JSON</button><label>Import reviewed CSV'
    if header_old not in source:
        raise ValueError("source review HTML header marker is missing or already repaired")
    source = source.replace(header_old, header_new, 1)
    key_match = re.search(r"const KEY=(\"[^\"]+\")", source)
    storage_key = json.loads(key_match.group(1)) if key_match else "exp13-g3b1-review"
    script = render_script(source_count, storage_key)
    if "__CSV_FIELDS__" in script:
        raise ValueError("internal CSV field placeholder missing")
    source = re.sub(r"<script>[\s\S]*?</script>", lambda _: "<script>" + script + "</script>", source, count=1)
    if "<script>" not in source or "reviewCounts" not in source:
        raise ValueError("failed to write source review controls")
    return source
