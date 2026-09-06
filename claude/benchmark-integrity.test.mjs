import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const script=process.env.BENCH_SCRIPT||fileURLToPath(new URL('./bench-gpu.mjs',import.meta.url));
// Synthetic readbacks exercise the real CLI's acceptance and accounting, not hardware.
for(const fixture of ['matching','wrong-count','missing-variant','invalid-duration','gpu-error','one-iteration'])test(fixture,()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'bench-integrity-'));
 try{
  const input=path.join(root,'input.txt');fs.writeFileSync(input,'=a=b=c=');
  const variants=[['A baseline',9,1,12,0],['B mapped',1,2,4,0],['C vector',2,3,6,0],['D gridstride',3,4,8,0],['E resident',0,2,2.5,10]].map(([name,uploadMs,computeMs,endToEndMs,setupMs])=>({name,note:'Synthetic',groups:1,threads:256,setupMs,runs:Array.from({length:5},()=>({uploadMs,computeMs,endToEndMs,count:4}))}));
  if(fixture==='wrong-count')variants[2].runs[3].count=5;
  if(fixture==='missing-variant')variants.pop();
  if(fixture==='invalid-duration')variants[0].runs[0].endToEndMs=null;
  const result={adapter:{vendor:'synthetic-fixture'},limits:{},bytes:7,paddedBytes:16,workgroupSize:256,variants,errors:fixture==='gpu-error'?['Synthetic validation failure']:[]};
  const mock=path.join(root,'browser.mjs');fs.writeFileSync(mock,`export default {chromium:{launch:async()=>({newPage:async()=>({on(){},route:async()=>{},goto:async()=>{},evaluate:async()=>(${JSON.stringify(result)})}),close:async()=>{}})}};`);
  const run=spawnSync(process.execPath,[script,input,'--iters',fixture==='one-iteration'?'1':'5','--headed=0'],{env:{...process.env,PLAYWRIGHT_PATH:mock},encoding:'utf8',timeout:15000});
  if(process.env.TEST_OUTPUT){fs.mkdirSync(process.env.TEST_OUTPUT,{recursive:true});fs.writeFileSync(path.join(process.env.TEST_OUTPUT,fixture+'.json'),JSON.stringify({script,fixture,status:run.status,stdout:run.stdout,stderr:run.stderr},null,2));}
  assert.equal(run.status,fixture==='matching'?0:fixture==='one-iteration'?2:1,run.stdout+'\n'+run.stderr);
  const line=run.stdout.split(/\r?\n/).find(s=>s.startsWith('JSON '));
  if(fixture!=='matching'){assert.equal(line,undefined,'Rejected run must not publish a benchmark result');return;}
  const report=JSON.parse(line.slice(5));
  assert.equal(report.bestComputeVariant,'A baseline');assert.equal(report.bestEndToEndVariant,'B mapped');
  assert.equal(report.rows[0].endToEndMs,12);assert.equal(report.rows[4].endToEndMs,4.5);assert.equal(report.rows[4].setupMs,10);
  assert.equal(report.rows[4].totalMs,22.5);assert.equal(report.provenance.bytes,7);assert.match(report.provenance.inputSha256,/^[a-f0-9]{64}$/);
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
