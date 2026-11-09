/* Synthetic recovery/identity contract, no industrial assets. */
const assert=require('node:assert/strict');
const api=require('./g3d2_review.js');
const bundle={bundle_id:'synthetic',pairs:[{pair_id:'a',source_size:[100,80],output_size:[200,160],slots:[{id:'N1',group:'NEAR',polygon_id:'p0'}]}]};
let count=0;
function test(name,fn){fn();count++;console.log('PASS '+name);}
test('initial PENDING round trip',()=>{const r=api.initial(bundle);assert.deepEqual(api.restore(bundle,JSON.stringify(r)),r);assert.equal(r.records[0].points[0].status,'PENDING');});
test('identity mismatch atomic rejection',()=>{const original=api.initial(bundle),bad=structuredClone(original);bad.bundle_id='other';assert.throws(()=>api.restore(bundle,JSON.stringify(bad)));assert.equal(original.bundle_id,'synthetic');});
test('duplicate pair rejected',()=>{const r=api.initial(bundle);r.records.push(r.records[0]);assert.throws(()=>api.validate(bundle,r));});
test('point identity rejected',()=>{const r=api.initial(bundle);r.records[0].points[0].id='invented';assert.throws(()=>api.validate(bundle,r));});
test('bounds rejected',()=>{const r=api.initial(bundle);r.records[0].points[0].output=[200,10];assert.throws(()=>api.validate(bundle,r));});
test('confirmation requires identity and both coordinates',()=>{const r=api.initial(bundle);r.records[0].points[0].status='CONFIRMED';assert.throws(()=>api.validate(bundle,r));r.records[0].reviewer='TEST';Object.assign(r.records[0].points[0],{label:'独特交点',source:[10,20],output:[20,40],uncertainty_px:1});assert.deepEqual(api.validate(bundle,r),r);});
test('unidentifiable preserves notes without coordinates',()=>{const r=api.initial(bundle);Object.assign(r.records[0].points[0],{status:'UNIDENTIFIABLE',notes:'无法确认，生成结构改变'});assert.deepEqual(api.restore(bundle,'\uFEFF'+JSON.stringify(r)),r);});
test('extra fields and trusted admission rejected',()=>{const r=api.initial(bundle);r.records[0].pair_suitability='PASS';assert.throws(()=>api.validate(bundle,r));r.records[0].pair_suitability='PENDING';r.fitted_matrix=[];assert.throws(()=>api.validate(bundle,r));});
test('native coordinates invariant under display zoom',()=>{assert.deepEqual(api.imageCoordinate(60,50,{left:10,top:10,width:100,height:80},[200,160]),[100,80]);assert.deepEqual(api.imageCoordinate(110,90,{left:10,top:10,width:200,height:160},[200,160]),[100,80]);});
test('CSV BOM quoting Unicode preserved',()=>{const r=api.initial(bundle);r.records[0].points[0].notes='含逗号,和"引号"\n换行';const text=api.csv(r,bundle);assert.ok(text.startsWith('\uFEFF'));assert.ok(text.includes('"含逗号,和""引号""\n换行"'));});
console.log(`${count}/${count} contracts passed`);
