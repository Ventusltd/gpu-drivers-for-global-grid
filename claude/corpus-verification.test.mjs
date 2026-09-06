import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
const analyser=process.env.CORPUS_ANALYSER||fileURLToPath(new URL('./analyse-corpus-gpu.mjs',import.meta.url));
// Explicit synthetic browser-result injection tests the acceptance gate, not GPU execution.
for(const fixture of ['matching','wrong-histogram','wrong-similarity','nonfinite-similarity','truncated-histogram','truncated-similarity','gpu-validation-error','invalid-timing','wrong-pair-count'])test(fixture,()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'corpus-verification-'));
 try{
  const input=path.join(root,'input'),out=path.join(root,'out');fs.mkdirSync(input);
  fs.writeFileSync(path.join(input,'a.js'),'AABB');fs.writeFileSync(path.join(input,'b.css'),'ABAB');
  fs.writeFileSync(path.join(input,'empty.html'),'');
  const hist=Array(3*256).fill(0);hist[65]=hist[66]=hist[256+65]=hist[256+66]=2;
  const sim=[1,1,0,0,1,0,0,0,0];
  if(fixture==='wrong-histogram')hist[65]=3;
  if(fixture==='wrong-similarity')sim[1]=0.1;
  if(fixture==='nonfinite-similarity')sim[1]=null;
  if(fixture==='truncated-histogram')hist.pop();
  if(fixture==='truncated-similarity')sim.pop();
  const result={adapter:{vendor:'synthetic-fixture',architecture:'no-GPU-executed'},uploadMs:1,histMs:1,simMs:1,pairs:3,hist,sim};
  if(fixture==='gpu-validation-error')result.errors=['Synthetic invalid bind group'];
  if(fixture==='invalid-timing')result.simMs=null;
  if(fixture==='wrong-pair-count')result.pairs=0;
  const mock=path.join(root,'synthetic-browser.mjs');
  fs.writeFileSync(mock,`export default {chromium:{launch:async()=>({newPage:async()=>({on(){},route:async()=>{},goto:async()=>{},evaluate:async()=>(${JSON.stringify(result)})}),close:async()=>{}})}};`);
  const run=spawnSync(process.execPath,[analyser,input,'--out',out],{env:{...process.env,PLAYWRIGHT_PATH:mock},encoding:'utf8',timeout:15000});
  const evidence=process.env.TEST_OUTPUT;
  if(evidence){fs.mkdirSync(evidence,{recursive:true});fs.writeFileSync(path.join(evidence,fixture+'.json'),JSON.stringify({fixture,analyser,status:run.status,stdout:run.stdout,stderr:run.stderr},null,2));}
  assert.equal(run.status,fixture==='matching'?0:1,run.stdout+'\n'+run.stderr);
  const report=JSON.parse(fs.readFileSync(path.join(out,'corpus-gpu-analysis.json'),'utf8'));
  assert.equal(report.status,fixture==='matching'?'verified':'rejected');
  assert.equal(report.provenance.files.length,3);
  assert(report.provenance.files.every(f=>/^[a-f0-9]{64}$/.test(f.sha256)));
  if(fixture==='matching'){assert.equal(report.verification.histogramMatchesCpu,true);assert.equal(report.verification.similarityMatchesCpu,true);assert.equal(report.verification.comparedSimilarityEntries,9);}
  else {assert.equal(report.nearDuplicates,undefined);assert.equal(report.gpuTiming,undefined);}
 }finally{fs.rmSync(root,{recursive:true,force:true});}
});
test('All-empty corpus rejects before creating a browser',()=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'corpus-empty-'));
 try{fs.writeFileSync(path.join(root,'empty.js'),'');const run=spawnSync(process.execPath,[analyser,root],{encoding:'utf8',timeout:15000});assert.equal(run.status,2);assert.match(run.stderr,/only empty files/);}
 finally{fs.rmSync(root,{recursive:true,force:true});}
});
