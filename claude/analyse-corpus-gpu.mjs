/**
 * WHAT IS ACTUALLY IN THE SERVED PAYLOAD — ANALYSED ON THE GPU.
 * ---------------------------------------------------------------------------
 * Not a benchmark. This asks a question whose answer changes what ships:
 *
 *   HOW MUCH OF THE CORPUS THE CLIENT DOWNLOADS IS THE SAME CODE AGAIN?
 *
 * GridAtlas keeps every generation of every cartridge, which is the right rule
 * for provenance -- a pinned generation must stay byte-identical forever. But
 * it means the directory the world can reach holds many near-copies, and on an
 * ARM phone the thing that matters is bytes over the wire. Knowing WHICH files
 * are near-duplicates of each other, and how near, is what lets a release drop
 * payload without breaking the immutability rule.
 *
 * ALL THE ANALYSIS RUNS ON THE GPU. Two compute passes, no CPU fallback in the
 * measurement path:
 *
 *   pass 1  per-file 256-bin byte histogram. One workgroup per file, a
 *           grid-stride loop over that file's byte range, 256 atomic bins in
 *           workgroup memory so the hot atomics stay on-chip, one global write
 *           per bin at the end.
 *   pass 2  pairwise cosine similarity over the N x 256 histogram matrix.
 *           One thread per (i,j) pair. N^2 threads, which is exactly the shape
 *           a GPU is for and exactly the shape that makes a CPU quadratic.
 *
 * WHAT THE CPU STILL DOES, STATED HONESTLY: reads the files off disk, and
 * verifies pass 1 against a CPU histogram so a wrong answer cannot pass as a
 * fast one. Orchestration and I/O are not "analysis" and cannot be moved onto
 * the GPU; every metric reported below is computed by a shader.
 *
 * COSINE SIMILARITY ON BYTE HISTOGRAMS IS A SCREEN, NOT A PROOF. Two files with
 * identical byte distributions are not necessarily identical files. It is a
 * cheap upper bound on similarity: a LOW score proves difference, a HIGH score
 * says "look here". Exact duplicates are confirmed separately by SHA-256 on the
 * host, and the two are reported separately rather than conflated.
 *
 *   node analyse-corpus-gpu.mjs [corpus-root] [--out <dir>] [--top 25]
 *
 * The corpus root is optional: argv[2], else $CORPUS_ROOT, else the directory
 * this script lives in -- which is a small but real corpus, so the harness runs
 * on any machine without the GridAtlas tree present. Numbers from that fallback
 * are a smoke test of the harness, not the measurements in results/.
 */
import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const HERE = path.dirname(fileURLToPath(import.meta.url));

function resolveRoot() {
  const argvPath = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : undefined;
  const candidates = [argvPath, process.env.CORPUS_ROOT, HERE];
  for (const c of candidates) if (c && existsSync(c) && statSync(c).isDirectory()) return c;
  console.error('usage: node analyse-corpus-gpu.mjs [corpus-root] [--out dir] [--top N]');
  console.error('no corpus root found. Pass a directory or set $CORPUS_ROOT.');
  process.exit(2);
}
const root = resolveRoot();
const oi = process.argv.indexOf('--out');
const OUT = oi > 0 ? process.argv[oi + 1] : null;
const ti = process.argv.indexOf('--top');
const TOP = ti > 0 ? Number(process.argv[ti + 1]) : 25;

const EXT = new Set(['.js', '.mjs', '.html', '.css']);
function collect(d, out = []) {
  for (const e of readdirSync(d, { withFileTypes: true })) {
    if (e.name === 'node_modules' || e.name === '.git') continue;
    const p = path.join(d, e.name);
    if (e.isDirectory()) collect(p, out);
    else if (EXT.has(path.extname(e.name))) out.push(p);
  }
  return out;
}

console.log('=== CORPUS ===');
const files = collect(root);
if (files.length === 0) {
  console.error(`no .js/.mjs/.html/.css files under ${root} -- nothing to analyse.`);
  process.exit(2);
}
const bufs = files.map(f => readFileSync(f));
const sizes = bufs.map(b => b.length);
const total = sizes.reduce((a, b) => a + b, 0);
const offsets = [];
let acc = 0;
for (const s of sizes) { offsets.push(acc); acc += s; }
const corpus = Buffer.concat(bufs);
console.log(`root  ${root}`);
console.log(`files ${files.length}`);
console.log(`bytes ${total.toLocaleString()} (${(total / 1048576).toFixed(2)} MB)`);
console.log(`machine ${os.cpus()[0].model.trim()}, ${(os.totalmem() / 1073741824).toFixed(2)} GB RAM`);

/* Exact duplicates are a host-side SHA-256 fact, reported separately from the
   GPU's similarity screen so the two are never conflated. */
const byHash = new Map();
bufs.forEach((b, i) => {
  const h = createHash('sha256').update(b).digest('hex');
  if (!byHash.has(h)) byHash.set(h, []);
  byHash.get(h).push(i);
});
const dupGroups = [...byHash.values()].filter(g => g.length > 1);
const dupBytes = dupGroups.reduce((a, g) => a + sizes[g[0]] * (g.length - 1), 0);

/* CPU ground truth for pass 1, so a wrong GPU histogram cannot pass. */
const cpuHist = new Uint32Array(files.length * 256);
for (let f = 0; f < files.length; f++) {
  const b = bufs[f];
  for (let i = 0; i < b.length; i++) cpuHist[f * 256 + b[i]] += 1;
}

/* Playwright may live in another project's node_modules rather than next to
   this file, so an absolute path is tried first and the bare specifier is the
   fallback. Override with PLAYWRIGHT_PATH. */
const PW = process.env.PLAYWRIGHT_PATH
  || 'C:/Users/vikra/OneDrive/Documents/GitHub/gridatlas-main-202609050200/node_modules/playwright/index.js';
/* playwright's entry is CommonJS, so its exports arrive on .default. */
const pwMod = existsSync(PW) ? await import(pathToFileURL(PW).href) : await import('playwright');
const chromium = (pwMod.default || pwMod).chromium;

/* Headless Chromium exposes no WebGPU adapter on this machine (measured), so
   the window is real. */
const browser = await chromium.launch({ headless: false, args: [
  '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--ignore-gpu-blocklist',
  '--enable-gpu', '--disable-gpu-sandbox', '--force_high_performance_gpu',
  '--enable-webgpu-developer-features'] });
const page = await browser.newPage();
page.on('console', m => console.log('  [page] ' + m.text()));
page.on('pageerror', e => console.log('  [pageerror] ' + e.message));
await page.route('https://analyse.local/**', r => r.request().url().endsWith('/corpus.bin')
  ? r.fulfill({ status: 200, contentType: 'application/octet-stream', body: corpus })
  : r.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>corpus analysis</title>' }));
await page.goto('https://analyse.local/index.html');

const gpu = await page.evaluate(async ({ offsets, sizes, n }) => {
  if (!navigator.gpu) return { error: 'WebGPU not exposed' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { error: 'no WebGPU adapter' };
  const info = adapter.info || {};
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize, maxBufferSize: L.maxBufferSize } });

  const raw = new Uint8Array(await (await fetch('/corpus.bin')).arrayBuffer());
  const words = Math.ceil(raw.length / 4);
  const padded = new Uint8Array(words * 4);
  padded.set(raw);

  /* CUDA Best Practices Guide, High Priority: the corpus crosses PCIe once and stays resident for
     both passes. */
  const tUp = performance.now();
  const srcBuf = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(srcBuf, 0, padded);
  const rangeBuf = device.createBuffer({ size: n * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const ranges = new Uint32Array(n * 2);
  for (let i = 0; i < n; i++) { ranges[i * 2] = offsets[i]; ranges[i * 2 + 1] = sizes[i]; }
  device.queue.writeBuffer(rangeBuf, 0, ranges);
  await device.queue.onSubmittedWorkDone();
  const uploadMs = performance.now() - tUp;

  const histBuf = device.createBuffer({ size: n * 256 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const simBuf = device.createBuffer({ size: n * n * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

  /* PASS 1 -- one workgroup per file, 256 on-chip bins, grid-stride over the
     file's byte range. Bytes are pulled out of the u32 array by shifting, so
     the loads stay word-aligned and coalesce across the warp. */
  const m1 = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read> ranges : array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> hist : array<atomic<u32>>;
var<workgroup> bins : array<atomic<u32>, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>) {
  atomicStore(&bins[l.x], 0u);
  workgroupBarrier();
  let r = ranges[wg.x];
  let start = r.x;
  let len = r.y;
  var i = l.x;
  loop {
    if (i >= len) { break; }
    let abs = start + i;
    let w = src[abs >> 2u];
    let b = (w >> ((abs & 3u) * 8u)) & 0xffu;
    atomicAdd(&bins[b], 1u);
    i = i + 256u;
  }
  workgroupBarrier();
  let c = atomicLoad(&bins[l.x]);
  if (c > 0u) { atomicAdd(&hist[wg.x * 256u + l.x], c); }
}` });
  const p1 = device.createComputePipeline({ layout: 'auto', compute: { module: m1, entryPoint: 'main' } });
  const bind1 = device.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: srcBuf } },
    { binding: 1, resource: { buffer: rangeBuf } },
    { binding: 2, resource: { buffer: histBuf } }] });

  device.queue.writeBuffer(histBuf, 0, new Uint32Array(n * 256));
  await device.queue.onSubmittedWorkDone();
  let t = performance.now();
  let enc = device.createCommandEncoder();
  let pass = enc.beginComputePass();
  pass.setPipeline(p1); pass.setBindGroup(0, bind1);
  pass.dispatchWorkgroups(n);
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const histMs = performance.now() - t;

  /* PASS 2 -- N^2 cosine similarities, one thread per pair. Each thread reads
     two 256-entry rows and reduces them; this is the quadratic step that makes
     a CPU crawl and a GPU shrug. */
  const m2 = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read> hist : array<u32>;
@group(0) @binding(1) var<storage, read_write> sim : array<f32>;
override N : u32 = 1u;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g : vec3<u32>) {
  let i = g.x; let j = g.y;
  if (i >= N || j >= N) { return; }
  if (j < i) { sim[i * N + j] = 0.0; return; }
  var dot : f32 = 0.0; var na : f32 = 0.0; var nb : f32 = 0.0;
  for (var k : u32 = 0u; k < 256u; k = k + 1u) {
    let a = f32(hist[i * 256u + k]);
    let b = f32(hist[j * 256u + k]);
    dot = dot + a * b; na = na + a * a; nb = nb + b * b;
  }
  let d = sqrt(na) * sqrt(nb);
  sim[i * N + j] = select(0.0, dot / d, d > 0.0);
}` });
  const p2 = device.createComputePipeline({ layout: 'auto',
    compute: { module: m2, entryPoint: 'main', constants: { N: n } } });
  const bind2 = device.createBindGroup({ layout: p2.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: histBuf } }, { binding: 1, resource: { buffer: simBuf } }] });

  t = performance.now();
  enc = device.createCommandEncoder();
  pass = enc.beginComputePass();
  pass.setPipeline(p2); pass.setBindGroup(0, bind2);
  pass.dispatchWorkgroups(Math.ceil(n / 16), Math.ceil(n / 16));
  pass.end();
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const simMs = performance.now() - t;

  async function read(buf, size) {
    const r = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(buf, 0, r, 0, size);
    device.queue.submit([e.finish()]);
    await r.mapAsync(GPUMapMode.READ);
    const copy = r.getMappedRange().slice(0);
    r.unmap(); r.destroy();
    return copy;
  }
  const histOut = new Uint32Array(await read(histBuf, n * 256 * 4));
  const simOut = new Float32Array(await read(simBuf, n * n * 4));

  return {
    adapter: { vendor: info.vendor, architecture: info.architecture, description: info.description },
    uploadMs, histMs, simMs, pairs: (n * (n - 1)) / 2,
    hist: Array.from(histOut), sim: Array.from(simOut)
  };
}, { offsets, sizes, n: files.length });

await browser.close();
if (gpu.error) { console.log('GPU UNAVAILABLE: ' + gpu.error); process.exit(1); }

/* VERIFY pass 1 against the CPU. A fast wrong histogram is not a result. */
let histAgrees = true;
for (let i = 0; i < cpuHist.length; i++) if (cpuHist[i] !== gpu.hist[i]) { histAgrees = false; break; }

const N = files.length;
const MB = total / 1048576;
console.log('\n=== GPU (MEASURED) ===');
console.log(`adapter ${JSON.stringify(gpu.adapter)}`);
console.log(`upload (once)      ${gpu.uploadMs.toFixed(1)} ms  (${(MB / (gpu.uploadMs / 1000)).toFixed(0)} MB/s)`);
console.log(`pass 1 histograms  ${gpu.histMs.toFixed(2)} ms  (${(MB / (gpu.histMs / 1000)).toFixed(0)} MB/s over ${N} files)`);
console.log(`pass 2 similarity  ${gpu.simMs.toFixed(2)} ms  (${gpu.pairs.toLocaleString()} pairs)`);
console.log(`histogram verification vs CPU: ${histAgrees ? 'MATCHES bin for bin' : '*** DISAGREES ***'}`);

console.log('\n=== EXACT DUPLICATES (host SHA-256, reported separately) ===');
console.log(`duplicate groups   ${dupGroups.length}`);
console.log(`redundant bytes    ${dupBytes.toLocaleString()} (${(dupBytes / 1048576).toFixed(2)} MB, ${(dupBytes / total * 100).toFixed(1)}% of corpus)`);
for (const g of dupGroups.sort((a, b) => sizes[b[0]] - sizes[a[0]]).slice(0, 8)) {
  console.log(`  ${(sizes[g[0]] / 1024).toFixed(0).padStart(7)} KB x${g.length}  ${g.map(i => path.relative(root, files[i])).join('  ==  ')}`);
}

console.log(`\n=== GPU SIMILARITY SCREEN: top ${TOP} near-duplicate pairs (not byte-identical) ===`);
const rel = files.map(f => path.relative(root, f).split(path.sep).join('/'));
const pairs = [];
for (let i = 0; i < N; i++) {
  for (let j = i + 1; j < N; j++) {
    const s = gpu.sim[i * N + j];
    if (s >= 0.99) pairs.push({ i, j, s });
  }
}
const exactSet = new Set(dupGroups.flatMap(g => g.flatMap(a => g.map(b => `${Math.min(a, b)}:${Math.max(a, b)}`))));
const near = pairs.filter(p => !exactSet.has(`${p.i}:${p.j}`)).sort((a, b) => b.s - a.s);
console.log(`pairs at cosine >= 0.99 : ${pairs.length.toLocaleString()} of ${gpu.pairs.toLocaleString()}`);
console.log(`  of which byte-identical: ${(pairs.length - near.length).toLocaleString()}`);
/* NOT "the reducible payload" -- that would be reading the screen as a proof.
   These are the pairs a real byte-exact diff should be pointed at next. */
console.log(`  near-duplicate, NOT identical: ${near.length.toLocaleString()}  <- candidates to diff, NOT proven redundancy`);
for (const p of near.slice(0, TOP)) {
  console.log(`  ${p.s.toFixed(5)}  ${(sizes[p.i] / 1024).toFixed(0)}KB ${rel[p.i]}`);
  console.log(`           ${(sizes[p.j] / 1024).toFixed(0)}KB ${rel[p.j]}`);
}

/* NOT a payload saving, and must never be read as one. Pairs overlap heavily --
   one file appearing in fifty pairs is counted fifty times -- so this sum can
   and does exceed the size of the whole corpus. It is printed only to show the
   screen's fan-out, and it is deliberately kept out of the JSON report so it
   cannot be picked up as a number. */
const nearBytes = near.reduce((a, p) => a + Math.min(sizes[p.i], sizes[p.j]), 0);
console.log(`\nbytes sitting in near-duplicate pairs: ${(nearBytes / 1048576).toFixed(2)} MB`);
console.log('  ^ NOT a payload saving. Pairs overlap, so files are counted many');
console.log('    times over; this figure can exceed the corpus itself. The only');
console.log('    defensible redundancy number above is the SHA-256 one.');

if (OUT) {
  mkdirSync(OUT, { recursive: true });
  const report = {
    measuredAt: new Date().toISOString(),
    machine: { cpu: os.cpus()[0].model.trim(), logicalCores: os.cpus().length,
      ramGB: +(os.totalmem() / 1073741824).toFixed(2), gpu: gpu.adapter },
    corpus: { root, files: N, bytes: total },
    gpuTiming: { uploadMs: gpu.uploadMs, histMs: gpu.histMs, simMs: gpu.simMs, pairs: gpu.pairs },
    verification: { histogramMatchesCpu: histAgrees },
    exactDuplicates: { groups: dupGroups.length, redundantBytes: dupBytes,
      pctOfCorpus: +(dupBytes / total * 100).toFixed(2),
      examples: dupGroups.slice(0, 20).map(g => ({ bytes: sizes[g[0]], files: g.map(i => rel[i]) })) },
    nearDuplicates: { threshold: 0.99, pairsAtThreshold: pairs.length,
      byteIdentical: pairs.length - near.length, nearNotIdentical: near.length,
      top: near.slice(0, 100).map(p => ({ cosine: +p.s.toFixed(6), a: rel[p.i], b: rel[p.j],
        bytesA: sizes[p.i], bytesB: sizes[p.j] })) }
  };
  writeFileSync(path.join(OUT, 'corpus-gpu-analysis.json'), JSON.stringify(report, null, 2));
  console.log(`\nwrote ${path.join(OUT, 'corpus-gpu-analysis.json')}`);
}
