"""Batch03 historical TRAIN review packaging. No models or quality decisions."""
import base64,collections,csv,difflib,hashlib,io,json,os,re,socket,subprocess,sys,tempfile,textwrap,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/root/autodl-tmp/borescope-new-seg-repro');os.chdir(ROOT)
HOST='autodl-container-m6ahgmrgv8-57f0f00d'
BRANCH='exp13/vlm-semantic-audit';HEAD='75dda701c054fb9ea6efb4ffbf4332fbed797f4e'
ORIGIN='git@github-borescope:FlankKaicoder/borescope-new-seg-repro.git'
B='results/exp13_restoration_feasibility/'
A0=B+'a0_20261021_081812/'
SCREEN=B+'qa_20261021_090000/source_visual_screening.csv'
APP=['docs/PROJECT_STATE.md','docs/DECISION_LOG.md','docs/experiment_index.md','results/experiment_registry.csv']
GATE='BATCH03_REVIEW_PACKET_PREPARED_AWAITING_HUMAN_REVIEW'
FLAGS={k:'NO' for k in ['FINAL_TRUSTED_PAIR_FREEZE','RESTORATION_QUALITY_GATE_PASS','EXP13_2_B_AUTHORIZATION','SSL_AUTHORIZATION']}
HUMAN=['defect_removed','normal_surface_plausible','background_preserved','geometry_preserved','hallucination_present','editing_artifact_present','restoration_quality','reviewer_comment','final_accept']
ENV=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
def now():return datetime.now(timezone.utc).isoformat()
def jb(o):return (json.dumps(o,ensure_ascii=False,indent=2)+'\n').encode()
def dig(b):return hashlib.sha256(b).hexdigest()
def sha(p):return dig(p.read_bytes())
def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],env=ENV)
def names(b):return [x.decode() for x in b.split(b'\0') if x]
def meta(paths):
 result=[]
 for n in paths:
  s=(ROOT/n).lstat();result.append([n,s.st_size,s.st_mtime_ns,s.st_mode])
 return result
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);r=list(rd)
  assert len(rd.fieldnames)==len(set(rd.fieldnames))
  assert all(None not in x and all(v is not None for v in x.values()) for x in r),str(p)
  return r
def source(r):return (r.get('class') or r['class_name']).casefold(),r['source_id']
def key(r):return (*source(r),r['run_id'],r['variant_id'])
def gate(head=HEAD,remote=HEAD):
 assert socket.gethostname()==HOST and Path.cwd()==ROOT and ROOT.resolve()==ROOT
 assert git('branch','--show-current').decode().strip()==BRANCH
 assert git('rev-parse','HEAD').decode().strip()==head
 assert git('remote','get-url','origin').decode().strip()==ORIGIN
 assert git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0]==remote
def external(name,data):
 with (X/name).open('xb') as f:f.write(jb(data))
def safe_history_path(n):
 p=(ROOT/n).resolve();base=ROOT/'results/exp13_diffusion_pilot/run_20260921_124440'
 assert p.is_relative_to(base) and p.is_file(),n
 return p
def selection():
 idx=rows(ROOT/(A0+'source_variant_review_index.csv'));ledger=rows(ROOT/SCREEN)
 by={source(r):r for r in ledger};allkeys=set(source(r) for r in idx)
 assert len(by)==len(ledger)==20 and allkeys==set(by)
 covered=set();batch_evidence={}
 for batch in [1,2]:
  mp=B+f'qa_20261023_batch0{batch}_manifest.csv';op=B+f'qa_20261023_batch0{batch}_user_review_overlay.csv'
  mm=rows(ROOT/mp);oo=rows(ROOT/op)
  assert len(mm)==len(oo)==8 and len({key(r) for r in mm})==8 and {key(r) for r in mm}=={key(r) for r in oo}
  covered.update(source(r) for r in mm)
  batch_evidence[mp]=sha(ROOT/mp);batch_evidence[op]=sha(ROOT/op)
 assert len(covered)==8 and covered<=allkeys
 remaining=allkeys-covered;assert len(remaining)==12
 sortkey=lambda k:(k[0],int(k[1]))
 chosen=[];seen=set();steps=[]
 for cls,quota in [('corrosion',3),('crack',1)]:
  pool={k for k in remaining if k[0]==cls};assert len(pool)>=quota,'HARD_STOP: remaining class quota not met'
  for step in range(quota):
   k=min(pool,key=lambda k:(by[k]['source_level_review_status'] in seen,int(k[1])))
   steps.append(dict(class_name=by[k]['class'],source_id=k[1],historical_status=by[k]['source_level_review_status'],
    status_not_previously_selected=by[k]['source_level_review_status'] not in seen,
    rule='Prefer status not yet represented in this batch, then numeric source_id ascending; corrosion quota first'))
   chosen.append(k);seen.add(by[k]['source_level_review_status']);pool.remove(k)
 selected=[]
 for k in chosen:
  rr=[r for r in idx if source(r)==k and r['historical_stage']=='R1']
  assert len(rr)==2 and {r['variant_id'].rsplit('_',1)[1] for r in rr}=={'v1','v2'},('HARD_STOP: R1 pair incomplete',k)
  assert all(r['split']=='train' and r['run_id']=='run_20260921_124440' for r in rr)
  selected.extend(sorted(rr,key=lambda r:r['variant_id']))
 return selected,dict(rule='Quota3corrosion+1Crack; greedily cover new historical screening status, ties numeric ID ascending; no image-quality or model-output selection',
  independent_key=['class','source_id'],all_source_count=len(allkeys),previously_covered_source_count=len(covered),remaining_before_count=len(remaining),
  all_sources=[dict(class_name=by[k]['class'],source_id=k[1]) for k in sorted(allkeys,key=sortkey)],
  covered_sources=[dict(class_name=by[k]['class'],source_id=k[1]) for k in sorted(covered,key=sortkey)],
  remaining_before=[dict(class_name=by[k]['class'],source_id=k[1],historical_screening=by[k]) for k in sorted(remaining,key=sortkey)],
  selected_sources=steps,unselected_after=[dict(class_name=by[k]['class'],source_id=k[1]) for k in sorted(remaining-set(chosen),key=sortkey)],
  input_sha256={A0+'source_variant_review_index.csv':sha(ROOT/(A0+'source_variant_review_index.csv')),SCREEN:sha(ROOT/SCREEN),**batch_evidence})
def preserve(index=False):
 for n,size,mtime,mode in S['untracked_metadata']+S['tracked_metadata']:
  if n in APP:continue
  t=(ROOT/n).lstat();assert [t.st_size,t.st_mtime_ns,t.st_mode]==[size,mtime,mode],n
 for n,h in S['protected_sha256'].items():assert sha(ROOT/n)==h,n
 a=rows(ROOT/(A0+'human_quality_review_template.csv'))
 assert len(a)==88 and all(r[f]=='' for r in a for f in HUMAN)
 for batch in [1,2]:
  for suffix in ['manifest.csv','user_review_overlay.csv']:
   p=B+f'qa_20261023_batch0{batch}_'+suffix;assert sha(ROOT/p)==S['protected_sha256'][p]
 if index:assert sha(ROOT/'.git/index')==S['index_sha256'] and not git('diff','--cached','--name-only')
 return dict(legacy_untracked_paths=11258,paths_size_mtime_ns_mode_unchanged=True,protected_sha256_count=len(S['protected_sha256']),
  historical_unique_r1_r2_files=164,a0_blank_human_fields=792,batch01_batch02_records_unchanged=True)
def put(path,data):
 assert path.is_relative_to(P) and not path.exists(),str(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
def csvbytes(rr):
 o=io.StringIO(newline='');w=csv.DictWriter(o,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);return o.getvalue().encode()
def patch_files(contents):
 diff=[]
 for n,b in contents.items():
  before=(ROOT/n).read_bytes().decode() if (ROOT/n).exists() else ''
  diff.extend(difflib.unified_diff(before.splitlines(keepends=True),b.decode().splitlines(keepends=True),fromfile='a/'+n if before else '/dev/null',tofile='b/'+n))
 p=X/'records.patch'
 with p.open('x',encoding='utf-8',newline='') as f:f.write(''.join(diff))
 for dry in [True,False]:subprocess.run(['/usr/bin/patch',*(['--dry-run'] if dry else []),'--batch','--fuzz=0','-p1','-i',str(p)],cwd=ROOT,check=True)
 for n,b in contents.items():assert (ROOT/n).read_bytes()==b
def check_package():
 from PIL import Image
 import numpy as np
 mm=rows(P/'batch03_manifest.csv');tt=rows(P/'batch03_human_review_template.csv')
 assert len(mm)==len(tt)==8 and len({key(r) for r in mm})==8 and {key(r) for r in mm}=={key(r) for r in tt}
 assert len({source(r) for r in mm})==4
 assert all(all(r[k]=='' for k in HUMAN+['user_decision','decision_scope','user_comment','reviewer_identity','reviewed_utc']) for r in tt)
 assert all(r['trusted_pair_frozen']==r['restoration_quality_gate_pass']=='false' for r in tt)
 original=rows(ROOT/(A0+'source_variant_review_index.csv'));origin={key(r):r for r in original};assert len(origin)==88
 for m in mm:
  assert origin[key(m)]['historical_stage']=='R1' and origin[key(m)]['split']=='train'
  assert all(m[k]==v for k,v in origin[key(m)].items())
  for role in ['source_image','original_defect_mask','edit_mask','output_image']:
   p=safe_history_path(m[role]);assert sha(p)==m[role+'_sha256']
   with Image.open(p) as im:im.load();assert f'{im.width}x{im.height}'==m[role+'_dimensions']
 for p in P.rglob('*.png'):
  with Image.open(p) as im:im.verify()
 assert len(list(P.rglob('*.png')))==13
 for line in (P/'SHA256SUMS.txt').read_text().splitlines():
  h,n=line.split('  ',1);assert sha(P/n)==h,n
 for r in mm:
  for n in ['source_review_sheet','variant_review_sheet']:assert (P/r[n]).is_file()
 integ=json.loads((P/'package_integrity.json').read_text())
 assert integ['gate']==GATE and integ['flags']==FLAGS
 assert integ['manifest_sha256']==sha(P/'batch03_manifest.csv') and integ['template_sha256']==sha(P/'batch03_human_review_template.csv')
 for asset in integ['review_assets']:
  assert sha(P/asset['path'])==asset['sha256']
 assert integ['join_failures']==0 and integ['missing_files']==[]
 for n,b in [(x,x.read_bytes()) for x in P.rglob('*') if x.is_file() and x.suffix in ['.md','.csv','.json','.txt','.py']]:
  assert all(not x.endswith((b' ',b'\t',b'\r')) for x in b.split(b'\n')),str(n)
 return dict(sources=4,candidates=8,images_decodable=13,join_failures=0,human_decision_and_quality_fields_blank=True,sha256s_verified=True)

mode=sys.argv[1]
if mode=='init':
 gate();assert not git('status','--porcelain=v1','--untracked-files=no')
 prevdir=Path('/root/autodl-tmp/exp13_2_a2_cross_batch_eligibility_um5px7qd')
 prev=json.loads((prevdir/'starting_state_snapshot.json').read_text());last=json.loads((prevdir/'final_delivery_receipt.json').read_text())
 u=git('ls-files','--others','--exclude-standard','-z');nn=names(u)
 assert len(nn)==11258 and nn==prev['untracked_paths'] and meta(nn)==prev['untracked_metadata']
 assert sha(ROOT/'.git/index')==last['index_final_sha256']
 protected=prev['protected_sha256'].copy();protected.update({n:h for n,h in last['file_sha256'].items() if n not in APP})
 for n in [A0+'file_integrity.csv',A0+'historical_sha256_before.json',A0+'human_quality_review_template.csv']:protected[n]=sha(ROOT/n)
 for n,h in protected.items():assert sha(ROOT/n)==h,n
 rr,sel=selection()
 # Validate the selected files against A0 recorded integrity BEFORE any writes.
 filemap={(r['variant_id'],r['path']):r for r in rows(ROOT/(A0+'file_integrity.csv'))}
 from PIL import Image
 for r in rr:
  for role in ['source_image','original_defect_mask','edit_mask','output_image']:
   p=safe_history_path(r[role]);f=filemap[(r['variant_id'],r[role])];assert sha(p)==f['sha256']
   with Image.open(p) as im:im.load();assert [im.width,im.height]==[int(f['width']),int(f['height'])]
   if role=='output_image':assert im.size==(512,512)
 tracked=names(git('ls-files','-z'));S=dict(utc=now(),hostname=HOST,pwd=str(ROOT),branch=BRANCH,head=HEAD,remote_head=HEAD,origin=ORIGIN,
  index_sha256=sha(ROOT/'.git/index'),untracked_paths=nn,untracked_metadata=meta(nn),untracked_paths_sha256=dig(u),
  tracked_metadata=meta(tracked),protected_sha256=protected,status_base64=base64.b64encode(git('status','--porcelain=v1','--untracked-files=all','-z')).decode(),selection=sel)
 X=Path(tempfile.mkdtemp(prefix='exp13_2_a2_batch03_delivery_',dir='/root/autodl-tmp'))
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 S['package_relative']=B+'batch03_'+stamp+'_'+X.name.rsplit('_',1)[1]
 S['handoff_relative']='docs/exp13_2_a2_batch03_'+stamp+'_handoff.md'
 external('starting_state_snapshot.json',S)
 print(json.dumps(dict(snapshot=str(X),package=S['package_relative'],selection=sel),ensure_ascii=False,indent=2));sys.exit()
X=Path(sys.argv[2]);S=json.loads((X/'starting_state_snapshot.json').read_text());P=ROOT/S['package_relative'];DOC=S['handoff_relative']
if mode=='build':
 gate();safety=preserve(True);assert not git('diff','--name-only') and not P.exists() and not (ROOT/DOC).exists()
 assert names(git('ls-files','--others','--exclude-standard','-z'))==S['untracked_paths']
 from PIL import Image,ImageDraw,ImageFont,ImageOps,ImageChops,__version__ as pillow_version
 import numpy as np
 rr,sel=selection();assert sel==S['selection']
 spatial={key(r):r for r in rows(ROOT/(A0+'spatial_correspondence_report.csv'))}
 screens={source(r):r for r in rows(ROOT/SCREEN)}
 a0files={(r['variant_id'],r['path']):r for r in rows(ROOT/(A0+'file_integrity.csv'))}
 fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
 font=lambda size:ImageFont.truetype(fontpath,size)
 def overlay(raw,mask):
  rgba=raw.convert('RGBA');a=mask.point(lambda v:110 if v else 0)
  rgba.alpha_composite(Image.merge('RGBA',(Image.new('L',raw.size,255),Image.new('L',raw.size,40),Image.new('L',raw.size,40),a)))
  return rgba.convert('RGB')
 def textblock(draw,x,y,txt,width=123,size=19):
  for paragraph in txt.split('\n'):
   for line in textwrap.wrap(paragraph,width=width,replace_whitespace=False) or ['']:
    draw.text((x,y),line,font=font(size),fill='#17212b');y+=size+6
  return y
 def sheet(title,panels,footer,path):
  # Six equal512-square panels, with aspect-ratio-preserving letterboxing. Headers never overlap images.
  w=1608;top=100;cellw=532;cellh=565
  footerlines=sum(max(1,len(textwrap.wrap(x,125,replace_whitespace=False))) for x in footer.split('\n'))
  h=top+cellh*2+30+footerlines*25+30
  canvas=Image.new('RGB',(w,h),'#ffffff');d=ImageDraw.Draw(canvas)
  d.text((20,18),title,font=font(27),fill='#111827');d.text((20,56),'Historical TRAIN R1 only. Equal output display size. No quality verdict or variant preference.',font=font(18),fill='#334155')
  for i,(label,im,kind) in enumerate(panels):
   x=12+(i%3)*cellw;y=top+(i//3)*cellh
   d.text((x,y),label,font=font(19),fill='#111827')
   tile=Image.new('RGB',(512,512),'#e7e9ed')
   v=ImageOps.contain(im,(512,512),method=Image.Resampling.NEAREST if kind=='mask' else Image.Resampling.LANCZOS)
   tile.paste(v,((512-v.width)//2,(512-v.height)//2));canvas.paste(tile,(x,y+32))
  textblock(d,18,top+cellh*2+15,footer,width=125,size=19)
  path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();canvas.save(path,format='PNG')
  return canvas.size
 groups=collections.defaultdict(list)
 for r in rr:groups[source(r)].append(r)
 manifest=[];inventory={};checks=[];images=[];overview_rows=[]
 for k,variants in groups.items():
  r=variants[0];sc=screens[k];slug=k[0]+'_'+k[1]
  native=Image.open(safe_history_path(r['source_image'])).convert('RGB')
  mask=Image.open(safe_history_path(r['original_defect_mask'])).convert('L')
  edit=Image.open(safe_history_path(r['edit_mask'])).convert('L')
  assert native.size==mask.size==edit.size
  sp=spatial[key(r)];box=json.loads(sp['native_crop_xyxy']);w,h=native.size
  side=min(w,h);calculated=[(w-side)/2,(h-side)/2,(w+side)/2,(h+side)/2]
  assert box==calculated and all(v==int(v) for v in box),('HARD_STOP: crop transform mismatch',k)
  assert sp['transform']=='ImageOps.fit; center=(0.5,0.5); RGB=LANCZOS; masks=NEAREST'
  x0,y0,x1,y1=map(int,box);a=np.array(mask)>0;total=int(a.sum());inside=int(a[y0:y1,x0:x1].sum());outside=total-inside
  assert total>0;fraction=inside/total
  assert abs(fraction-float(sp['original_mask_retained_fraction_native_crop']))<1e-15
  # Exact documented deterministic historical view; derived only for display, not a new generated candidate.
  fit=ImageOps.fit(native,(512,512),method=Image.Resampling.LANCZOS,centering=(.5,.5))
  # ImageOps.fit uses resize(box=...), which preserves Lanczos support across
  # crop boundaries. Cropping the image first is not pixel-identical.
  assert np.array_equal(np.array(fit),np.array(native.resize((512,512),Image.Resampling.LANCZOS,box=tuple(box))))
  mf=ImageOps.fit(mask,(512,512),method=Image.Resampling.NEAREST,centering=(.5,.5))
  ef=ImageOps.fit(edit,(512,512),method=Image.Resampling.NEAREST,centering=(.5,.5))
  assert int((np.array(mf)>0).sum())==int(sp['original_defect_px_fit'])
  ann=overlay(native,mask);cropann=ann.copy();ImageDraw.Draw(cropann).rectangle((x0,y0,x1-1,y1-1),outline='#00d7ff',width=4)
  outputs=[];fileinfo=[]
  for v in variants:
   assert spatial[key(v)]['native_crop_xyxy']==sp['native_crop_xyxy']
   assert float(spatial[key(v)]['original_mask_retained_fraction_native_crop'])==fraction
   out=Image.open(safe_history_path(v['output_image'])).convert('RGB');assert out.size==(512,512);outputs.append(out)
   m=dict(v)
   for role in ['source_image','original_defect_mask','edit_mask','output_image']:
    p=safe_history_path(v[role]);f=a0files[(v['variant_id'],v[role])]
    assert sha(p)==f['sha256']
    with Image.open(p) as im:im.load();dims=f'{im.width}x{im.height}';assert dims==f"{f['width']}x{f['height']}"
    m[role+'_sha256']=sha(p);m[role+'_dimensions']=dims;m[role+'_absolute_path']=str(p)
    inventory[v[role]]=dict(path=v[role],sha256=sha(p),dimensions=dims,bytes=p.stat().st_size,a0_sha256_matches=True)
   m.update(native_crop_xyxy=json.dumps(box),transform=sp['transform'],original_mask_pixels=total,original_mask_pixels_inside_crop=inside,
    original_mask_pixels_outside_crop=outside,original_mask_retained_fraction=sp['original_mask_retained_fraction_native_crop'],
    independently_recomputed_retained_fraction=repr(fraction),retention_definition='Native annotated mask area inside historical crop; not restoration success',
    historical_screening_status=sc['source_level_review_status'],historical_screening_risk=sc['risk_and_evidence'],historical_screening_evidence=sc['evidence_source'],
    existing_registered_user_decision='',existing_decision_record_status='NO_BATCH01_OR_BATCH02_RECORD_FOR_THIS_SOURCE',
    source_review_sheet='contact_sheets/'+slug+'_review.png',variant_review_sheet='variant_review_sheets/'+v['variant_id']+'_review.png',
    a0_index_path=A0+'source_variant_review_index.csv',a0_spatial_path=A0+'spatial_correspondence_report.csv',historical_screening_ledger=SCREEN,
    technical_audit_status='FILES_CROP_AND_MASK_COUNTS_CONSISTENT_NOT_QUALITY_APPROVAL',trusted_pair_frozen='false',restoration_quality_gate_pass='false')
   manifest.append(m)
   fileinfo.append(v['variant_id']+' seed='+v['seed']+' strength='+v['strength']+' prompt='+v['prompt_id']+' mask='+v['mask_type'])
  common='\n'.join([f"Source {r['class']} {r['source_id']}. Native {w}x{h}. Crop xyxy={box}. Aligned view512x512.",
   f"Original annotation retained {inside}/{total} = {fraction:.16g}. Outside annotation pixels: {outside}.",
   'Retained fraction is annotation coverage, not repair success. Cropped-away annotation is not reviewed as repaired.',
   'Red overlay = original defect annotation. Cyan rectangle = historical crop. Mask panels use NEAREST.',
   'Native and candidate panels are letterboxed without stretching. Alignment view is deterministic ImageOps.fit.',*fileinfo,
   'Paths below are repository-relative; full SHA256 and exact provenance are in batch03_manifest.csv.',
   'Source: '+r['source_image'],'Mask: '+r['original_defect_mask'],'Edit: '+r['edit_mask'],
   'v1: '+variants[0]['output_image'],'v2: '+variants[1]['output_image']])
  spath=P/manifest[-1]['source_review_sheet']
  dims=sheet(r['class']+' '+r['source_id']+' - source comparison',[
   ('Native source (unannotated)',native,'rgb'),('Original mask + historical crop',cropann,'rgb'),('Historical aligned input (derived view)',fit,'rgb'),
   ('Historical edit mask (aligned)',ef,'mask'),(variants[0]['variant_id'],outputs[0],'rgb'),(variants[1]['variant_id'],outputs[1],'rgb')],common,spath)
  images.append(dict(path=str(spath.relative_to(P)),size=list(dims),sha256=sha(spath)))
  for v,out in zip(variants,outputs):
   diff=ImageChops.difference(fit,out)
   footer=common+'\nAbsolute RGB difference is unamplified; it does not establish defect removal or quality.'
   vp=P/('variant_review_sheets/'+v['variant_id']+'_review.png')
   dims=sheet(v['variant_id']+' - candidate review',[
    ('Native annotation + historical crop',cropann,'rgb'),('Historical aligned input (derived)',fit,'rgb'),('Original defect overlay (aligned)',overlay(fit,mf),'rgb'),
    ('Edit mask (aligned)',ef,'mask'),(v['variant_id']+' (historical output)',out,'rgb'),('Absolute RGB difference (unamplified)',diff,'rgb')],footer,vp)
   images.append(dict(path=str(vp.relative_to(P)),size=list(dims),sha256=sha(vp)))
  checks.append(dict(class_name=r['class'],source_id=r['source_id'],native_size=list(native.size),native_crop_xyxy=box,
   transform=sp['transform'],mask_pixels_total=total,mask_pixels_inside=inside,mask_pixels_outside=outside,retained_fraction=fraction,
   a0_spatial_match=True,fit_matches_explicit_resize_box=True,fit_original_mask_pixels=int((np.array(mf)>0).sum()),quality_not_assessed=True))
  overview_rows.append((r,fit,outputs,fraction))
 # Compact navigation overview, with the same display scale for v1/v2.
 overview=Image.new('RGB',(1216,1856),'white');od=ImageDraw.Draw(overview)
 od.text((16,12),'Batch03 navigation - input / v1 / v2',font=font(25),fill='black')
 for i,(r,fit,outs,fraction) in enumerate(overview_rows):
  y=60+i*444;od.text((16,y),f"{r['class']} {r['source_id']} | mask retained {fraction:.6f} | no quality verdict",font=font(20),fill='black')
  for j,(im,label) in enumerate([(fit,'Aligned input'),(outs[0],r['class'].lower()+'_'+r['source_id']+'_v1'),(outs[1],r['class'].lower()+'_'+r['source_id']+'_v2')]):
   x=16+j*400;od.text((x,y+28),label,font=font(18),fill='black');overview.paste(im.resize((384,384),Image.Resampling.LANCZOS),(x,y+54))
 op=P/'contact_sheets/batch03_overview.png';overview.save(op);images.append(dict(path=str(op.relative_to(P)),size=list(overview.size),sha256=sha(op)))
 put(P/'batch03_manifest.csv',csvbytes(manifest))
 template=[]
 for m in manifest:
  t={k:m[k] for k in ['source_id','class','run_id','variant_id','source_image','source_image_sha256','original_defect_mask','original_defect_mask_sha256','edit_mask','edit_mask_sha256','output_image','output_image_sha256','native_crop_xyxy','original_mask_retained_fraction','source_review_sheet','variant_review_sheet','technical_audit_status']}
  t.update({f:'' for f in ['user_decision','decision_scope','user_comment','reviewer_identity','reviewed_utc',*HUMAN]})
  t.update(trusted_pair_frozen='false',restoration_quality_gate_pass='false');template.append(t)
 put(P/'batch03_human_review_template.csv',csvbytes(template));put(P/'source_selection_audit.json',jb(sel))
 # Source-level historical opinions are separated from neutral image panels to reduce anchoring.
 history='# Historical screening ledger (not candidate decisions)\n\nOpen after neutral review if desired. ChatGPT qualitative source-level notes from uploaded A0 sheets, not independent expert scores. No registered Batch01/02 decisions exist for selected sources.\n\n'
 for k in groups:
  s=screens[k];history+=f"## {s['class']} {s['source_id']}\n\nStatus: `{s['source_level_review_status']}`.\n\n{s['risk_and_evidence']}\n\nEvidence: {s['evidence_source']}\n\nNext step recorded historically: {s['next_step']}\n\n"
 put(P/'historical_screening_context.md',(history.rstrip()+'\n').encode())
 readme=f'''# Exp13.2-A2 Batch03 historical review packet

Prepared UTC: {now()}. `{GATE}`. Evidence HEAD: `{HEAD}`.

Start with `contact_sheets/batch03_overview.png`, then the four source sheets and eight variant sheets. All are review composites of existing TRAIN-only R1 files. No model, generation, inference or quality assessment occurred. The derived aligned input uses the recorded deterministic ImageOps.fit transform; this is a display reconstruction, not a newly generated candidate or a claim that serialized historical input pixels were recovered.

## Selection

`source_selection_audit.json` lists all20 source keys,8 previously covered and12 remaining (Crack5/corrosion7), derived from the original88-row A0 index, historical source ledger and Batch01/02 manifest/overlay joins. Quota:3corrosion+1Crack. At each step prefer a historical screening status not yet represented in this batch, then numeric source ID ascending. Class corrosion first. This selects corrosion1,9,6 and Crack469; every ID and both R1variants exist in A0. Selection uses no current visual rating or generated output comparison. R2 is excluded. Four selected sources leave8historical sources uncovered.

## Review navigation and image semantics

- `contact_sheets/*_review.png`: native RAW copy in a panel, original mask/crop overlay, aligned input, aligned edit mask, v1/v2 at equal512display size.
- `variant_review_sheets/`: annotation/crop, input, aligned original defect overlay, edit mask, historical candidate and unamplified absolute RGB difference. Differences are descriptive, not quality scores.
- `batch03_manifest.csv`:8unique candidates; exact historical relative/absolute paths, original SHA256/dimensions, run/prompt/seed/mask provenance, crop/native counts and source-level historical context. Source opinions are never candidate decisions. Missing prior decisions remain blank with a separate no-record indicator.
- `batch03_human_review_template.csv`: independent blank template. All user decisions, scope/comments/reviewer fields and9quality fields are empty. Both freeze/quality booleans are false. Do not populate from screening labels or pixel differences.
- `historical_screening_context.md`: historical subjective source notes, separate from neutral image panels. Includes corrosion9 hallucination warning and Crack469 uncertainty. No new judgment is made.
- `package_integrity.json`: file inventory, crop cross-checks, image dimensions, joins and preservation validation.
- `SHA256SUMS.txt`: every package file except itself, including integrity JSON. The external ZIP checksum is in the delivery receipt.

Red tint marks the original native defect mask; cyan marks crop bounds. Panels preserve aspect ratio with letterboxing, RGB uses LANCZOS and masks NEAREST. Output v1/v2 are shown at equal size and without preference. Gray letterbox is a display-only margin. Edit mask grayscale includes historical feathering; no mask/prompt modification occurs. Original historical bytes remain in place and are linked rather than duplicated. Full paths/hashes are in manifest; display-only resizing/recompression affects composite pixels only.

## Spatial scope

Mask retained fraction = native nonzero original annotation area within historical center crop / all native annotated area. It is not repair success or a mask annotation completeness guarantee. Cropped-away defect pixels are outside the reviewed output. A fraction1.0 does not imply the entire native image is represented. Existing R1 rendering and Diffusion pixel changes, dilation/feather boundary artifacts and gray input borders require human review; the recorded deterministic grid does not certify semantic geometry.

'''
 for c in checks:readme+=f"- {c['class_name']} {c['source_id']}: crop{c['native_crop_xyxy']}; {c['mask_pixels_inside']}/{c['mask_pixels_total']} retained={c['retained_fraction']}; outside={c['mask_pixels_outside']} annotation pixels.\n"
 readme+='''
## Human review and authority

Inspect residual defect, normal metal texture, mechanical hallucinations, geometry, edit boundary/halo/smoothing, and changes outside the edit. Smoothness alone is not normality. Low-strength retained cracks are not successful repair. Each source's two variants are correlated and count as one independent source. User supplies final semantic decisions; ChatGPT may assist later and determine Gate under separate review. Polygon masks originate from annotations; future learning use must disclose supervision.

Current sufficient formal admission evidence remains0for previously reviewed sources; adding review assets creates no admissions. A1 proposed10trusted sources,5per class is unfrozen and not a statistical guarantee. User acceptance alone cannot certify Restoration Quality PASS. All historical templates, overlays, masks, outputs and Phase C1 files are protected. No VAL/TEST images, dataset image reads or paper repository access.

```text
FINAL_TRUSTED_PAIR_FREEZE = NO
RESTORATION_QUALITY_GATE_PASS = NO
EXP13_2_B_AUTHORIZATION = NO
SSL_AUTHORIZATION = NO
```

Stop for human/ChatGPT review. No next batch or experiments are authorized by this packet.
'''
 put(P/'README.md',readme.encode())
 script=base64.b64decode(os.environ['BATCH03_SCRIPT_BASE64']);put(P/'prepare_batch03.py',script)
 integ=dict(prepared_utc=now(),gate=GATE,flags=FLAGS,hostname=HOST,branch=BRANCH,starting_head=HEAD,starting_snapshot_path=str(X/'starting_state_snapshot.json'),
  selection=sel,source_count=4,candidate_count=8,join_failures=0,missing_files=[],historical_run='run_20260921_124440',split='train',
  source_spatial_checks=checks,input_file_inventory=list(inventory.values()),review_assets=images,
  manifest_sha256=sha(P/'batch03_manifest.csv'),template_sha256=sha(P/'batch03_human_review_template.csv'),
  human_fields_blank=True,protection=safety,pillow_version=pillow_version,numpy_version=np.__version__,
  display='PNG composites only; original bytes unchanged; deterministic aligned display reconstruction; no source/output stretching',
  no_models_or_new_candidates=True,no_new_visual_quality_judgments=True,no_val_test_images_read=True,no_dataset_images_read=True,
  checklist='Decode all input/output/masks and13review sheets; SHA matches A0; transform/crop/native-mask counts match A0; blank independent human template')
 put(P/'package_integrity.json',jb(integ))
 sums=''.join(sha(p)+'  '+str(p.relative_to(P))+'\n' for p in sorted(P.rglob('*')) if p.is_file())
 put(P/'SHA256SUMS.txt',sums.encode())
 result=check_package();preserve(True)
 hdr=f'''# Exp13.2-A2 Batch03 handoff

Prepared UTC {now()}. `{GATE}`.

Host `{HOST}`; repo `{ROOT}`; branch `{BRANCH}`; starting local/live remote HEAD `{HEAD}`; origin `{ORIGIN}`. Complete pre-write state: `{X}/starting_state_snapshot.json`.

Package: [{P.name}](../{S['package_relative']}/README.md). Manifest: [8candidates](../{S['package_relative']}/batch03_manifest.csv); [blank human template](../{S['package_relative']}/batch03_human_review_template.csv); [integrity](../{S['package_relative']}/package_integrity.json); [selection](../{S['package_relative']}/source_selection_audit.json).

Programmatic A0/ledger/Batch01/02 join:20historical independent sources,8covered,12remaining (Crack5/corrosion7). Quota3corrosion+1Crack; choose previously unrepresented status first, then numeric ID ascending, corrosion first. Selected corrosion1,9,6 and Crack469, each R1v1/v2. No reviewed source substituted, no R2 mixed, no output-effect selection. No registered Batch01/02 decisions for selected sources. Historical source notes remain separate from neutral panels.

4sources/8unique variants,0join failures,0missing files.4source sheets+8variant sheets+1overview all decode. Native source/mask/edit/output SHA256 and dimensions match A0 integrity. Deterministic ImageOps.fit display, crop counts and spatial coordinates match A0; semantic quality not certified. Human decision and quality fields all blank; booleans false.

'''
 for c in checks:hdr+=f"- {c['class_name']} {c['source_id']}: native{c['native_size']}, crop{c['native_crop_xyxy']}; mask{c['mask_pixels_inside']}/{c['mask_pixels_total']}={c['retained_fraction']}, outside{c['mask_pixels_outside']}.\n"
 hdr+=f'''
Retained fraction is annotation coverage, not restoration success. Cropped-away annotations cannot count as repaired. Normal texture, residual defects, geometry, hallucinations, smoothing and edit artifacts await users' semantic decisions and ChatGPT review. No candidate accepted, rejected, ranked or frozen here. Original pending decisions stay pending; A1 suggested10/5/5not frozen. Formal admission evidence remains0in the current cross-batch audit.

Only this independent package, this handoff and append-only PROJECT_STATE/DECISION_LOG(D030)/experiment_index/registry changes are staged. Registry retains11columns and records starting evidence HEAD, not self-referential commit. Snapshot/validation/cached/commit/push receipts and ZIP are external under `{X}`. Exact final local/remote HEAD, ZIP path/size/SHA256 and file whitelist are in `{X}/final_delivery_receipt.json` after delivery.

Before staging: protected historical hashes and11,258legacy path/size/mtime_ns/mode unchanged; A0 88rows/792human fields blank; Batch01/02 manifests/overlays unchanged; index unchanged until authorized whitelist staging. Validate CSV/JSON/Markdown links, complete one-to-one8variant join, all source/derived checksums and git diff --check/cached --check. After commit/push tracked worktree/index clean, live remote equals local; original untracked assets retained. A normal authorized commit changes index, not historical asset contents.

No models loaded/downloaded/executed, no new model images, inference, training, SSL, Exp13.2-B, VAL/TEST image access, dataset modification, paper access, A1 changes or next stage. Only Pillow/NumPy historical-image review assembly. Stop after Git delivery, await user and ChatGPT.

```text
{GATE}
{chr(10).join(k+' = '+v for k,v in FLAGS.items())}
```
'''
 contents={DOC:hdr.encode()}
 for n,label in [(APP[0],'PROJECT_STATE'),(APP[1],'DECISION_LOG'),(APP[2],'experiment_index')]:
  append=f"\n## Exp13.2-A2 Batch03 review preparation ({now()})\n\n"
  if label=='DECISION_LOG':append+='Decision D030: prepare only new historical review assets, without a semantic quality or admission decision. '
  append+=f"`{GATE}`. [Handoff]({Path(DOC).name}); [packet](../{S['package_relative']}/README.md). Recomputed20sources minus8covered leaves12; select corrosion1/9/6 and Crack469 by status diversity then numeric ID,3corrosion+1Crack.8R1variants/4sources;13review PNGs; no models or new decisions. Historical overlays and A0 unchanged; all freeze/quality/Exp13.2-B/SSL remain NO. Stop for human/ChatGPT review.\n"
  contents[n]=(ROOT/n).read_bytes()+append.encode()
 reg=rows(ROOT/APP[3]);assert len(reg)==50 and len(reg[0])==11
 row=['Exp13.2-A2-Batch03-ReviewPacket','Batch03 historical R1 review packet',GATE,S['utc'],now(),HEAD,'Existing historical TRAIN-only R1 source/mask/output copies; no raw dataset images',sha(P/'batch03_manifest.csv'),'CPU Pillow/NumPy deterministic review assembly; no models',S['package_relative'],'4sources/8variants;corrosion1/9/6+Crack469;status diversity/numeric ID selection;human fields blank;freeze/quality/Exp13.2-B/SSL NO;git_commit=starting evidence HEAD']
 o=io.StringIO(newline='');csv.writer(o,lineterminator='\n').writerow(row);contents[APP[3]]=(ROOT/APP[3]).read_bytes()+o.getvalue().encode()
 patch_files(contents)
 new=sorted(str(p.relative_to(ROOT)) for p in P.rglob('*') if p.is_file())+[DOC]
 white=new+APP
 plan=dict(whitelist=white,new_files=new,append_only=APP,expected_sha256={n:sha(ROOT/n) for n in white},validation=result)
 external('plan.json',plan)
 print(json.dumps(dict(package=str(P),handoff=DOC,files=len(new),validation=result,spatial=checks),indent=2))
elif mode=='validate':
 gate();safety=preserve(True);p=json.loads((X/'plan.json').read_text());validation=check_package()
 assert set(names(git('ls-files','--others','--exclude-standard','-z')))==set(S['untracked_paths'])|set(p['new_files'])
 assert set(git('diff','--name-only').decode().splitlines())==set(APP)
 for n,h in p['expected_sha256'].items():assert sha(ROOT/n)==h,n
 for n in APP:assert (ROOT/n).read_bytes().startswith(git('show',HEAD+':'+n)),n
 assert len(rows(ROOT/APP[3]))==51
 for n in [DOC,*APP[:3],*p['new_files']]:
  if not n.endswith('.md'):continue
  text=(ROOT/n).read_text()
  if n in APP:text=(ROOT/n).read_bytes()[len(git('show',HEAD+':'+n)):].decode()
  assert text.count('```')%2==0
  for dest in re.findall(r'\]\(([^)]+)\)',text):assert ((ROOT/n).parent/dest).is_file(),(n,dest)
 subprocess.run(['git','-C',str(ROOT),'diff','--check'],env=ENV,check=True)
 # External downloadable ZIP contains every package file and no historical source copies.
 z=X/(P.name+'.zip');assert not z.exists()
 with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for f in sorted(P.rglob('*')):
   if f.is_file():archive.write(f,arcname=P.name+'/'+str(f.relative_to(P)))
 with zipfile.ZipFile(z) as archive:
  assert archive.testzip() is None
  for f in P.rglob('*'):
   if f.is_file():assert dig(archive.read(P.name+'/'+str(f.relative_to(P))))==sha(f)
 external('pre_staging_validation.json',dict(utc=now(),validation=validation,preservation=safety,diff_check='PASS',csv_json_structure='PASS',
  append_prefixes_preserved=True,index_unchanged=True,zip_path=str(z),zip_bytes=z.stat().st_size,zip_sha256=sha(z),zip_members_byte_identical=True))
 print(json.dumps(dict(validation=validation,protection=safety,zip_path=str(z),zip_bytes=z.stat().st_size,zip_sha256=sha(z)),indent=2))
elif mode=='stage':
 gate();preserve(True);p=json.loads((X/'plan.json').read_text());check_package()
 assert (X/'pre_staging_validation.json').exists()
 for n,h in p['expected_sha256'].items():assert sha(ROOT/n)==h,n
 assert set(names(git('ls-files','--others','--exclude-standard','-z')))==set(S['untracked_paths'])|set(p['new_files'])
 assert set(git('diff','--name-only').decode().splitlines())==set(APP)
 subprocess.run(['git','-C',str(ROOT),'diff','--check'],env=ENV,check=True)
 subprocess.run(['git','-C',str(ROOT),'add','--',*p['whitelist']],check=True)
 assert set(names(git('diff','--cached','--name-only','-z')))==set(p['whitelist'])
 for n in p['whitelist']:assert git('show',':'+n)==(ROOT/n).read_bytes(),n
 assert not git('diff','--name-only') and dig(git('ls-files','--others','--exclude-standard','-z'))==S['untracked_paths_sha256']
 subprocess.run(['git','-C',str(ROOT),'diff','--cached','--check'],env=ENV,check=True)
 external('cached_audit.json',dict(utc=now(),whitelist=p['whitelist'],diff_sha256=dig(git('diff','--cached','--binary')),cached_check='PASS',stat=git('diff','--cached','--stat').decode()))
 print(git('diff','--cached','--stat').decode())
elif mode=='commit':
 gate();preserve();check_package();p=json.loads((X/'plan.json').read_text());c=json.loads((X/'cached_audit.json').read_text())
 assert dig(git('diff','--cached','--binary'))==c['diff_sha256'] and not git('diff','--name-only')
 subprocess.run(['git','-C',str(ROOT),'diff','--cached','--check'],env=ENV,check=True)
 subprocess.run(['git','-C',str(ROOT),'commit','-m','Prepare Exp13.2-A2 Batch03 historical review packet'],check=True)
 h=git('rev-parse','HEAD').decode().strip();assert git('rev-parse','HEAD^').decode().strip()==HEAD
 assert set(names(git('diff-tree','--no-commit-id','--name-only','-r','-z','HEAD')))==set(p['whitelist'])
 external('commit_receipt.json',dict(head=h,parent=HEAD,utc=now()));print(h)
elif mode=='push':
 h=json.loads((X/'commit_receipt.json').read_text())['head'];gate(h,HEAD);safety=preserve();validation=check_package()
 assert not git('status','--porcelain=v1','--untracked-files=no')
 subprocess.run(['git','-C',str(ROOT),'push','origin','HEAD:refs/heads/'+BRANCH],check=True)
 gate(h,h);safety=preserve();p=json.loads((X/'plan.json').read_text());check_package()
 assert not git('status','--porcelain=v1','--untracked-files=no')
 assert dig(git('ls-files','--others','--exclude-standard','-z'))==S['untracked_paths_sha256']
 assert not git('show','--format=','--check','HEAD')
 for n,hs in p['expected_sha256'].items():assert sha(ROOT/n)==hs
 external('final_delivery_receipt.json',dict(utc=now(),hostname=HOST,pwd=str(ROOT),branch=BRANCH,origin=ORIGIN,starting_head=HEAD,
  final_head=h,remote_head=h,commit_push='SUCCESS',gate=GATE,flags=FLAGS,package_path=str(P),handoff=DOC,validation=validation,
  preservation=safety,index_start_sha256=S['index_sha256'],index_final_sha256=sha(ROOT/'.git/index'),index_unchanged_before_staging=True,
  tracked_worktree_clean=True,staged_area_empty=True,legacy_untracked_paths_sha256=S['untracked_paths_sha256'],
  files=p['expected_sha256'],zip=json.loads((X/'pre_staging_validation.json').read_text()),unauthorized_operations=False))
 print(json.dumps(json.loads((X/'final_delivery_receipt.json').read_text()),indent=2))
else:raise ValueError(mode)
