// Exercise a private file:// package in an isolated Chrome profile; never approve an actual case.
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import path from "node:path";
import {createHash} from "node:crypto";
import {pathToFileURL,fileURLToPath} from "node:url";

const [directory,browserModule,chrome,qaDirectory]=process.argv.slice(2);
const {puppeteer}=await import(pathToFileURL(browserModule).href);
await fs.mkdir(qaDirectory);
const browser=await puppeteer.launch({executablePath:chrome,headless:true,args:["--disable-background-networking"]});
const errors=[],network=[],requests=[];
try{
  const page=await browser.newPage();await page.setViewport({width:1440,height:1100});
  page.on("pageerror",e=>errors.push(e.message));
  page.on("request",r=>{requests.push(r.url());if(/^https?:/.test(r.url()))network.push(r.url());});
  const url=pathToFileURL(path.join(directory,"index.html")).href;
  await page.goto(url);
  const bundle=await page.evaluate(()=>window.exp17Review.bundle);
  assert.equal(bundle.records.length,545);assert.equal(new Set(bundle.records.map(r=>r.candidate_id)).size,545);
  assert.equal(bundle.records.filter(r=>r.priority_six).length,6);assert.equal(bundle.records.filter(r=>r.class==="Tip curl").length,0);
  assert.equal(bundle.formal_trusted_pairs,0);assert.equal((await page.evaluate(()=>window.exp17Review.exportPayload())).records.length,0);
  await page.waitForFunction(()=>[...document.querySelectorAll("#images img")].every(i=>i.complete&&i.naturalWidth>0));
  const initialImageRequests=requests.filter(u=>/\.(?:png|jpe?g|bmp)$/i.test(u)).length;
  assert(initialImageRequests<=3,"Initial page must not preload the full image population");
  await page.click("#overlay");assert.equal(await page.$$eval("#images svg",s=>s.length),0);
  await page.click("#overlay");await page.click("#valid");await page.click("#valid");
  assert.equal(requests.filter(u=>/\.(?:png|jpe?g|bmp)$/i.test(u)).length,initialImageRequests,
               "Overlay/visibility changes must reuse already loaded media");
  const manifest=JSON.parse(await fs.readFile(path.join(directory,"file_manifest.json"),"utf8"));
  for(const relative of Object.keys(manifest))await fs.access(fileURLToPath(new URL(relative,url)));
  for(let i=0;i<6;i++){
    for(const view of ["full","roi"]){
      await page.select("#view",view);
      await page.waitForFunction(()=>[...document.querySelectorAll("#images img")].length===3&&[...document.querySelectorAll("#images img")].every(i=>i.complete&&i.naturalWidth>0));
      assert.equal(await page.$$eval("#images svg",s=>s.length),3);
    }
    assert.match(await page.$eval("#currentAttribution",e=>e.textContent),/UNREVIEWED/);
    if(i<5)await page.click("#next");
  }
  await page.select("#scope","all");
  for(const c of [...new Set(bundle.records.map(r=>r.class))]){
    await page.select("#class",c);assert.equal(await page.$eval("#position",e=>e.textContent),"1 / "+bundle.records.filter(r=>r.class===c).length);
  }
  await page.select("#class","");
  for(const status of [...new Set(bundle.records.map(r=>r.g1_status))]){
    await page.select("#status",status);assert.equal(await page.$eval("#position",e=>e.textContent),"1 / "+bundle.records.filter(r=>r.g1_status===status).length);
  }
  await page.select("#status","");
  await page.type("#source","NOT_A_SOURCE");assert.equal(await page.$eval("#empty",e=>e.hidden),false);
  await page.$eval("#source",e=>{e.value="";e.dispatchEvent(new Event("input",{bubbles:true}));});
  // An invalid PASS click on the real record is rejected and creates no record.
  await page.select("#decision","PASS");await page.click("#save");
  assert.equal((await page.evaluate(()=>window.exp17Review.exportPayload())).records.length,0);
  assert.match(await page.$eval("#message",e=>e.textContent),/PASS/);
  // Real-record interaction uses HOLD only, explicitly tagged as automated QA in an ephemeral profile.
  await page.select("#decision","HOLD");await page.type("#reviewer","AUTOMATED_QA_NOT_HUMAN");
  await page.type("#note",'AUTOMATED_QA_NOT_HUMAN: quoted "text", comma\nsecond line');await page.click("#save");
  const exported=await page.evaluate(()=>window.exp17Review.exportPayload());
  assert.equal(exported.records.length,1);assert.equal(exported.records[0].final_decision,"HOLD");
  assert.equal(exported.records[0].training_approved,false);assert.equal(exported.records[0].independent_data_gate,"NOT_GRANTED");
  assert.equal(exported.records[0].reviewer_id,"AUTOMATED_QA_NOT_HUMAN");
  assert(exported.records[0].asset_identity.valid_mask_sha256&&exported.records[0].asset_identity.matrix_output_to_source);
  const text=await page.evaluate(()=>window.exp17Review.csvText(window.exp17Review.exportPayload()));
  await fs.writeFile(path.join(qaDirectory,"synthetic_browser_hold_export.json"),JSON.stringify(exported,null,2),{flag:"wx"});
  await fs.writeFile(path.join(qaDirectory,"synthetic_browser_hold_export.csv"),text,{flag:"wx"});
  const downloadSession=await browser.target().createCDPSession();
  await downloadSession.send("Browser.setDownloadBehavior",{behavior:"allow",downloadPath:path.resolve(qaDirectory),eventsEnabled:true});
  await page.click("#json");await page.click("#csv");
  let downloads=[];
  for(let i=0;i<40;i++){downloads=(await fs.readdir(qaDirectory)).filter(n=>n.startsWith("exp17_user_review_"));if(downloads.filter(n=>n.endsWith(".json")).length===1&&downloads.filter(n=>n.endsWith(".csv")).length===1)break;await new Promise(r=>setTimeout(r,100));}
  assert.equal(downloads.filter(n=>n.endsWith(".json")).length,1);assert.equal(downloads.filter(n=>n.endsWith(".csv")).length,1);
  const actualDownload=JSON.parse(await fs.readFile(path.join(qaDirectory,downloads.find(n=>n.endsWith(".json"))),"utf8"));
  assert.deepEqual(actualDownload.records,exported.records);
  // PASS validation uses only an unpersisted synthetic identity, never an actual candidate.
  assert.equal(await page.evaluate(()=>{
    const api=window.exp17Review,r={candidate_id:"SYNTHETIC_QA_ONLY",binding_sha256:"synthetic",g0_status:"G0_PASS",asset_identity:{synthetic:true},
      source_id:"SYNTHETIC",group_id:"SYNTHETIC",polygon_id:"SYNTHETIC_000",class:"SYNTHETIC",review_id:"SYNTHETIC",variant_id:"SYNTHETIC",version:"SYNTHETIC"};
    const v={schema:api.bundle.schema,...Object.fromEntries(["source_id","group_id","polygon_id","class","review_id","variant_id","version"].map(k=>[k,r[k]])),
      candidate_id:r.candidate_id,binding_sha256:r.binding_sha256,asset_identity:r.asset_identity,package_id:api.bundle.package_id,
      review_role:"USER_LOCAL_REVIEW_UNVERIFIED",training_approved:false,independent_data_gate:"NOT_GRANTED",final_decision:"PASS",
      reviewer_id:"AUTOMATED_SYNTHETIC_FIXTURE",boundary_evidence_note:"synthetic edge",reviewed_at:"2026-11-22T00:00:00Z"};
    for(const k of api.bundle.positive)v[k]="YES";for(const k of api.bundle.negative)v[k]="NO";
    const valid=api.validate(r,v).length===0;v.asset_identity={tampered:true};
    return valid&&api.validate(r,v).length>0;
  }),true);
  await page.reload();assert.equal((await page.evaluate(()=>window.exp17Review.exportPayload())).records.length,1);
  const context=await browser.createBrowserContext(),fresh=await context.newPage();await fresh.goto(url);
  await fresh.waitForFunction(()=>[...document.querySelectorAll("#images img")].length===3&&[...document.querySelectorAll("#images img")].every(i=>i.complete&&i.naturalWidth>0));
  assert.equal((await fresh.evaluate(()=>window.exp17Review.exportPayload())).records.length,0);
  const template=JSON.parse(await fs.readFile(path.join(directory,"new_review_template.json"),"utf8"));
  assert.equal(template.records.filter(r=>r.final_decision!=="PENDING").length,0);
  assert.equal(template.records.filter(r=>r.review_role!=="UNREVIEWED").length,0);
  await fresh.screenshot({path:path.join(qaDirectory,"offline_review.png"),fullPage:true});
  await context.close();assert.deepEqual(network,[]);assert.deepEqual(errors,[]);
  const result={status:"PASS_OFFLINE_BROWSER_QA",browser:await browser.version(),package_id:bundle.package_id,
    index_sha256:createHash("sha256").update(await fs.readFile(path.join(directory,"index.html"))).digest("hex"),
    candidates:545,priority_cases_exercised:6,
    accessible_media_files:Object.keys(manifest).length,initial_image_requests:initialImageRequests,external_requests:0,page_errors:0,
    all_production_template_records_pending:true,actual_case_pass_records_generated:0,hold_export_is_automated_QA_only:true,
    pass_validation_fixture_synthetic_only:true,offline:true};
  result.actual_json_and_csv_downloads=2;
  await fs.writeFile(path.join(qaDirectory,"browser_qa.json"),JSON.stringify(result,null,2)+"\n",{flag:"wx"});
  console.log(JSON.stringify(result));
}finally{await browser.close();}
