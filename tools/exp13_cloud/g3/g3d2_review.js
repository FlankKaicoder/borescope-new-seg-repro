/* Offline mechanical-landmark review. No network, automatic matches, or fitting. */
(function (root) {
  'use strict';
  const SCHEMA = 'g3d2-mechanical-landmarks-v1';
  const qualityKeys = ['visual_repair','normal_structure','mask_complete','hard_seam','feather_halo','ui_non_target','multi_polygon','shortcut_risk'];
  const decisions = ['PENDING','PASS','FAIL','UNCERTAIN','NOT_APPLICABLE'];
  const pointStates = ['PENDING','CONFIRMED','UNIDENTIFIABLE','WRONG_MATCH','STRUCTURE_CHANGED','AMBIGUOUS'];
  const clone = x => JSON.parse(JSON.stringify(x));
  const exact = (obj, keys) => obj && typeof obj === 'object' && !Array.isArray(obj) && Object.keys(obj).sort().join('|') === [...keys].sort().join('|');
  function initial(bundle) {
    return {schema:SCHEMA,bundle_id:bundle.bundle_id,records:bundle.pairs.map(p=>({pair_id:p.pair_id,reviewer:'',notes:'',geometry_review:'PENDING',pair_suitability:'PENDING',quality:Object.fromEntries(qualityKeys.map(k=>[k,'PENDING'])),points:p.slots.map(s=>({id:s.id,status:'PENDING',label:'',source:null,output:null,uncertainty_px:null,notes:''}))}))};
  }
  function validate(bundle, state) {
    const fail = msg => {throw new Error(msg);};
    if (!exact(state,['schema','bundle_id','records']) || state.schema!==SCHEMA || state.bundle_id!==bundle.bundle_id) fail('审核 schema 或冻结基准身份不匹配');
    if (!Array.isArray(state.records) || state.records.length!==bundle.pairs.length) fail('需要完整 Pair 集合');
    const expected = new Map(bundle.pairs.map(p=>[p.pair_id,p])), seen = new Set();
    for (const r of state.records) {
      if (!exact(r,['pair_id','reviewer','notes','geometry_review','pair_suitability','quality','points'])) fail('Pair 字段不匹配');
      const p=expected.get(r.pair_id);
      if (!p || seen.has(r.pair_id)) fail('未知或重复 Pair'); seen.add(r.pair_id);
      if (typeof r.reviewer!=='string' || typeof r.notes!=='string' || !decisions.slice(0,4).includes(r.geometry_review) || !['PENDING','REJECT','CANDIDATE_FOR_GATE'].includes(r.pair_suitability)) fail('无效 Pair 判断');
      if (!exact(r.quality,qualityKeys) || Object.values(r.quality).some(v=>!decisions.includes(v))) fail('无效质量判断');
      if (!Array.isArray(r.points) || r.points.length!==p.slots.length) fail('点位槽数量错误');
      const slots=new Set(p.slots.map(s=>s.id)), used=new Set();
      for (const t of r.points) {
        if (!exact(t,['id','status','label','source','output','uncertainty_px','notes']) || !slots.has(t.id) || used.has(t.id)) fail('未知、重复或多余点位字段'); used.add(t.id);
        if (!pointStates.includes(t.status) || typeof t.label!=='string' || typeof t.notes!=='string') fail('无效点位状态');
        for (const side of ['source','output']) {
          const xy=t[side], size=p[side+'_size'];
          if (xy!==null && (!Array.isArray(xy) || xy.length!==2 || xy.some((v,i)=>typeof v!=='number' || !Number.isFinite(v) || v<0 || v>=size[i]))) fail('点位坐标超出图像或非有限数');
        }
        if (t.uncertainty_px!==null && (typeof t.uncertainty_px!=='number' || !Number.isFinite(t.uncertainty_px) || t.uncertainty_px<0)) fail('无效点击不确定度');
        if (t.status==='CONFIRMED' && (!r.reviewer.trim() || !t.label.trim() || t.source===null || t.output===null || t.uncertainty_px===null)) fail('确认点位前请填写审核者、身份、两侧坐标和不确定度');
      }
    }
    return clone(state);
  }
  function restore(bundle, text) { return validate(bundle, JSON.parse(text.replace(/^\uFEFF/,''))); }
  function csv(state,bundle) {
    const q=v=>'"'+String(v??'').replaceAll('"','""')+'"';
    const fields=['pair_id','reviewer','point_id','group','polygon_id','status','label','source_x','source_y','output_x','output_y','uncertainty_px','notes'];
    const rows=[];
    for(const r of state.records){const p=bundle.pairs.find(p=>p.pair_id===r.pair_id);for(const t of r.points){const s=p.slots.find(s=>s.id===t.id);rows.push([r.pair_id,r.reviewer,t.id,s.group,s.polygon_id,t.status,t.label,t.source?.[0],t.source?.[1],t.output?.[0],t.output?.[1],t.uncertainty_px,t.notes]);}}
    return '\uFEFF'+[fields,...rows].map(row=>row.map(q).join(',')).join('\r\n');
  }
  function imageCoordinate(clientX,clientY,rect,size) {
    const xy=[(clientX-rect.left)/rect.width*size[0],(clientY-rect.top)/rect.height*size[1]];
    return xy.every((v,i)=>Number.isFinite(v)&&v>=0&&v<size[i])?xy.map(v=>Math.round(v*100)/100):null;
  }
  const api={initial,validate,restore,csv,imageCoordinate};
  if(typeof module!=='undefined'&&module.exports){module.exports=api;return;}
  root.G3D2=api;
  const $=id=>document.getElementById(id), bundle=JSON.parse($('bundle').textContent);
  let state=initial(bundle), index=0, slotIndex=0, epoch=0;
  const storageKey='exp13-g3d2-'+bundle.bundle_id;
  const images={};
  const pair=()=>bundle.pairs[index], record=()=>state.records.find(r=>r.pair_id===pair().pair_id), point=()=>record().points.find(t=>t.id===pair().slots[slotIndex].id);
  const notice=message=>{$('notice').textContent=message;};
  function persist(){try{localStorage.setItem(storageKey,JSON.stringify(state));$('status').textContent='已自动保存；结束时请导出 JSON';}catch(e){$('status').textContent='自动保存失败，请导出或复制备份';notice('浏览器存储不可用：'+e.message);} }
  function transaction(change){const previous=clone(state);try{change();state=validate(bundle,state);persist();notice('');}catch(e){state=previous;notice(e.message);} }
  function backup(){const text=JSON.stringify(state,null,2);$('backupText').value=text;return text;}
  function download(text,type,name){backup();try{const url=URL.createObjectURL(new Blob([text],{type})),a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();setTimeout(()=>{a.remove();URL.revokeObjectURL(url);},1500);$('status').textContent='已发起下载，请确认文件已保存；下方有备份文本';}catch(e){notice('下载失败，请复制下方备份文本：'+e.message);} }
  function draw(side){const canvas=$(side),img=images[side];if(!img)return;const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(img,0,0);const scale=canvas.width/canvas.getBoundingClientRect().width;for(const t of record().points){if(!t[side])continue;const [x,y]=t[side];ctx.strokeStyle=t.id===point().id?'#ffdd00':'#00e8ff';ctx.lineWidth=2*scale;ctx.beginPath();ctx.arc(x,y,6*scale,0,Math.PI*2);ctx.moveTo(x-10*scale,y);ctx.lineTo(x+10*scale,y);ctx.moveTo(x,y-10*scale);ctx.lineTo(x,y+10*scale);ctx.stroke();ctx.font=`${13*scale}px system-ui`;ctx.fillStyle=ctx.strokeStyle;ctx.fillText(t.id,x+10*scale,y-9*scale);} }
  function zoom(){for(const side of ['source','output']){const c=$(side),z=Number($('zoom').value);c.style.width=(z?c.width*z:c.parentElement.clientWidth)+'px';c.style.height='auto';draw(side);} }
  function renderPoint(){const t=point(),s=pair().slots[slotIndex];$('slot').value=String(slotIndex);$('label').value=t.label;$('pointStatus').value=t.status;$('uncertainty').value=t.uncertainty_px??'';$('pointNotes').value=t.notes;$('region').textContent=s.group==='NEAR'?`近目标 ${s.polygon_id}：Polygon 外 2–${pair().near_radius_px} 源像素，选择可确认的正常机械点。`:`远目标：距所有 Polygon > ${pair().near_radius_px} 源像素。`;$('coordinates').textContent=`源坐标 ${JSON.stringify(t.source)} ｜ 输出坐标 ${JSON.stringify(t.output)}`;draw('source');draw('output');}
  function render(){epoch++;const token=epoch,p=pair(),r=record();$('pair').value=String(index);$('title').textContent=`${p.pair_id} · ${p.class}`;$('stratum').textContent=p.selection_reason;$('risk').textContent=`原几何：${p.historical_geometry}；${p.polygon_ids.length} 个 Polygon。历史视觉判断仅为参考，当前所有审核需独立完成。`;$('reviewer').value=r.reviewer;$('geometry').value=r.geometry_review;$('suitability').value=r.pair_suitability;$('notes').value=r.notes;$('evidence').open=false;$('slot').replaceChildren();p.slots.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${s.id} · ${s.group}`;$('slot').append(o);});slotIndex=0;
    $('quality').replaceChildren();const labels={visual_repair:'视觉：目标缺陷完全消失',normal_structure:'视觉：正常机械结构保持／合理恢复',mask_complete:'合成：完整 Repair Mask 覆盖目标',hard_seam:'合成：Hard 无残边／接缝',feather_halo:'合成：Feather 无光晕／接缝',ui_non_target:'非目标：UI/Logo/时间戳及其他缺陷安全',multi_polygon:'多 Polygon：逐区正确处理',shortcut_risk:'训练：无明显人工捷径'};for(const key of qualityKeys){const label=document.createElement('label');label.textContent=labels[key]+' ';const sel=document.createElement('select');sel.id='q_'+key;decisions.forEach(d=>{const o=document.createElement('option');o.textContent=d;sel.append(o);});sel.value=r.quality[key];sel.onchange=()=>transaction(()=>{record().quality[key]=sel.value;});label.append(sel);$('quality').append(label);}
    $('gallery').replaceChildren();for(const [file,title] of [['mask_overlay.png','完整冻结 Repair Mask（非 valid_target）'],['aligned_output.png','历史保存的 Aligned Output'],['hard_composite.png','历史 Hard Composite'],['feather_composite.png','历史 Feather Composite']]){const f=document.createElement('figure'),cap=document.createElement('figcaption');cap.textContent=title+'（点击原尺寸查看）';f.append(cap);if(p.assets[file]){const a=document.createElement('a'),img=document.createElement('img');a.href=p.pair_id+'/'+file;a.target='_blank';a.rel='noopener';img.src=a.href;img.loading='lazy';a.append(img);f.append(a);}else{f.append(document.createTextNode('无历史有效变换／合成，保持 UNVERIFIED'));}$('gallery').append(f);}
    $('provenance').textContent=JSON.stringify({prior_visual:p.prior_visual,methods:p.methods,composite_invariant:p.composite_invariant},null,2);
    for(const side of ['source','output']){delete images[side];const c=$(side),img=new Image();c.width=p[side+'_size'][0];c.height=p[side+'_size'][1];c.getContext('2d').clearRect(0,0,c.width,c.height);img.onload=()=>{if(token!==epoch)return;images[side]=img;zoom();};img.onerror=()=>notice('图像加载失败：'+img.src);img.src=p.pair_id+'/'+(side==='source'?'original.png':'raw_output.png');}
    renderPoint();}
  function importText(text){try{const candidate=restore(bundle,text);state=candidate;persist();render();notice('完整审核数据恢复成功');}catch(e){notice('导入失败，原有审核未改变：'+e.message);} }
  bundle.pairs.forEach((p,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${p.pair_id} · ${p.class}`;$('pair').append(o);});
  $('pair').onchange=()=>{index=Number($('pair').value);render();};$('prev').onclick=()=>{index=(index-1+bundle.pairs.length)%bundle.pairs.length;render();};$('next').onclick=()=>{index=(index+1)%bundle.pairs.length;render();};$('slot').onchange=()=>{slotIndex=Number($('slot').value);renderPoint();};$('zoom').onchange=zoom;
  for(const side of ['source','output']){$(side).onclick=e=>{const xy=imageCoordinate(e.clientX,e.clientY,$(side).getBoundingClientRect(),pair()[side+'_size']);if(xy){transaction(()=>{point()[side]=xy;});renderPoint();}};}
  $('clear').onclick=()=>{transaction(()=>{point().source=null;point().output=null;point().status='PENDING';});renderPoint();};
  for(const [id,field] of [['label','label'],['pointStatus','status'],['pointNotes','notes'],['uncertainty','uncertainty_px']]){$(id).onchange=()=>{transaction(()=>{point()[field]=field==='uncertainty_px'?($('uncertainty').value===''?null:Number($('uncertainty').value)):$(id).value;});renderPoint();};}
  for(const [id,field] of [['reviewer','reviewer'],['notes','notes'],['geometry','geometry_review'],['suitability','pair_suitability']]){$(id).onchange=()=>{transaction(()=>{record()[field]=$(id).value;});$(id).value=record()[field];};}
  $('export').onclick=()=>download(backup(),'application/json;charset=utf-8','g3d2_review_'+bundle.bundle_id.slice(0,10)+'.json');$('csv').onclick=()=>download(csv(state,bundle),'text/csv;charset=utf-8','g3d2_landmarks.csv');$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{const f=$('file').files[0];if(f){try{importText(await f.text());}catch(e){notice('无法读取文件：'+e.message);}finally{$('file').value='';}}};$('backup').onclick=backup;$('restoreText').onclick=()=>importText($('backupText').value);$('copy').onclick=async()=>{try{await navigator.clipboard.writeText(backup());notice('JSON 已复制');}catch{notice('剪贴板不可用，请手动选择下方 JSON 复制');}};
  try{const text=localStorage.getItem(storageKey);if(text)state=restore(bundle,text);}catch(e){notice('自动恢复失败，已保留初始状态；请从 JSON 恢复：'+e.message);}render();
  window.addEventListener('resize',zoom);
  // Commit focused text before a backup if the page is closed immediately after editing.
  window.addEventListener('beforeunload',()=>{document.activeElement?.blur();persist();});
})(typeof globalThis!=='undefined'?globalThis:this);
