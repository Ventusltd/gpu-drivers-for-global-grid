/**
 * THE TELEPRINT ON THE RTX 5070, WITH THE CUDA BEST-PRACTICE RULES APPLIED.
 * ---------------------------------------------------------------------------
 * The same ~27.5 MB artefact the CPU bench ate, uploaded to the discrete GPU and
 * reduced by a compute shader. Offline: a Chromium already installed for
 * Playwright, WebGPU on, and NOTHING fetched from the network -- the bytes are
 * served to the page from disk through a route interceptor.
 *
 * WHY FIVE VARIANTS AND NOT ONE. the CUDA C++ Best Practices Guide gives
 * specific, testable rules. Each variant below turns ONE of them on, so the
 * cost of each is a measured delta rather than a belief:
 *
 *   A baseline      writeBuffer upload, one u32 (4 bytes) per thread, wg 256
 *   B mapped        upload via mappedAtCreation -- the WebGPU analogue of
 *                   the guide's "page-locked/pinned memory transfers attain the
 *                   highest bandwidth", because it writes into driver-owned
 *                   memory once instead of staging a pageable copy
 *   C vectorized    one vec4<u32> (16 bytes) per thread. Guide: coalesced,
 *                   128-bit loads maximise global memory throughput
 *   D gridstride    fewer workgroups, each thread looping. Guide: occupancy
 *                   and instruction-level parallelism hide memory latency
 *   E resident      upload ONCE, run the kernel N times. Guide, High Priority:
 *                   "Minimize data transfer between the host and the device"
 *
 * WHICH GPU ANSWERED is printed, never assumed: this laptop has an Intel iGPU
 * and the 5070, and silently getting the iGPU is the classic false result.
 * Chrome also ignores powerPreference on Windows (crbug.com/369219127), so the
 * printed adapter is the only evidence of which device did the work.
 *
 * EVERY VARIANT MUST AGREE WITH THE CPU. The work is counting bytes equal to
 * '=' (0x3D) -- not arbitrary, since the teleprint's section boundaries are
 * runs of '=', so this is the first pass of a real parse expressed as the
 * embarrassingly-parallel reduction a GPU exists for. A fast wrong answer is
 * not a result, so the CPU count is computed first and every run is checked.
 *
 *   node bench-gpu.mjs [artefact.txt] [--iters 5] [--headed=0]
 *
 * The artefact path is optional: argv[2], else $BENCH_ARTEFACT, else
 * ./sample-artefact.txt next to this file (make-sample-artefact.mjs writes one).
 */
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import path from 'node:path';

/* Playwright may live in another project's node_modules rather than next to
   this file, so an absolute path is tried first and the bare specifier is the
   fallback. Override with PLAYWRIGHT_PATH. */
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PW = process.env.PLAYWRIGHT_PATH
  || 'C:/Users/vikra/OneDrive/Documents/GitHub/gridatlas-main-202609050200/node_modules/playwright/index.js';
/* playwright's entry is CommonJS, so its exports arrive on .default. */
const pwMod = existsSync(PW) ? await import(pathToFileURL(PW).href) : await import('playwright');
const chromium = (pwMod.default || pwMod).chromium;

function resolveArtefact() {
  const argvPath = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : undefined;
  const candidates = [argvPath, process.env.BENCH_ARTEFACT, path.join(HERE, 'sample-artefact.txt')];
  for (const c of candidates) if (c && existsSync(c)) return c;
  console.error('no artefact found. Pass a path, set $BENCH_ARTEFACT, or generate one:');
  console.error('  node make-sample-artefact.mjs sample-artefact.txt 26');
  process.exit(2);
}
const file = resolveArtefact();
const ii = process.argv.indexOf('--iters');
const ITERS = ii > 0 ? Number(process.argv[ii + 1]) : 5;
if (!Number.isInteger(ITERS) || ITERS < 2 || ITERS > 100) { console.error('--iters must be an integer from 2 to 100 (one warm-up plus measured iterations).'); process.exit(2); }

const buf = readFileSync(file);
if (!buf.length) { console.error('Choose a non-empty artefact.'); process.exit(2); }
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const provenance = {inputSha256: sha256(buf), bytes: buf.length, harnessSha256: sha256(readFileSync(fileURLToPath(import.meta.url)))};
const MB = buf.length / 1048576;
console.log(`file  ${file}`);
console.log(`bytes ${buf.length} (${MB.toFixed(2)} MB)`);

/* CPU ground truth, twice: scalar, then the same count via indexOf, which is
   the fair comparison because it is what a real parser would use. */
let cpuCount = 0;
let t0 = process.hrtime.bigint();
for (let i = 0; i < buf.length; i++) if (buf[i] === 0x3d) cpuCount += 1;
const cpuScalarMs = Number(process.hrtime.bigint() - t0) / 1e6;
t0 = process.hrtime.bigint();
let n = 0, at = -1;
while ((at = buf.indexOf(0x3d, at + 1)) !== -1) n += 1;
const cpuIndexOfMs = Number(process.hrtime.bigint() - t0) / 1e6;
if (n !== cpuCount) throw Error('CPU reference counts disagree');
console.log(`CPU scalar  count '=' : ${cpuCount} in ${cpuScalarMs.toFixed(1)}ms (${(MB / (cpuScalarMs / 1000)).toFixed(0)} MB/s)`);
console.log(`CPU indexOf count '=' : ${n} in ${cpuIndexOfMs.toFixed(1)}ms (${(MB / (cpuIndexOfMs / 1000)).toFixed(0)} MB/s)`);

/* HEADLESS CHROMIUM HAS NO GPU ADAPTER ON THIS MACHINE. Measured: headless
   requestAdapter() returns null with "No available adapters", so a headless run
   would silently report "no GPU" on a laptop that has two. The window is
   therefore real. --headed=0 forces the headless path back on if you want to
   see that failure for yourself. */
const HEADLESS = process.argv.includes('--headed=0');
const CHANNEL = process.env.BENCH_CHANNEL || undefined; // e.g. 'chrome' for installed Chrome
const browser = await chromium.launch({
  headless: HEADLESS,
  channel: CHANNEL,
  args: ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
    '--ignore-gpu-blocklist', '--enable-gpu', '--disable-gpu-sandbox',
    '--force_high_performance_gpu', '--enable-webgpu-developer-features']
});
const page = await browser.newPage();
page.on('console', m => console.log('  [page] ' + m.text()));
page.on('pageerror', e => console.log('  [pageerror] ' + e.message));

/* The bytes reach the page from DISK, not from the network and not marshalled
   through CDP as a 27-million-element array. */
await page.route('https://bench.local/**', route => {
  if (route.request().url().endsWith('/payload.bin')) {
    return route.fulfill({ status: 200, contentType: 'application/octet-stream', body: buf });
  }
  return route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>bench</title>' });
});
await page.goto('https://bench.local/index.html');

let result;
try { result = await page.evaluate(async (iters) => {
  if (!navigator.gpu) return { error: 'navigator.gpu is undefined: WebGPU not exposed in this Chromium' };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { error: 'requestAdapter returned null: no WebGPU adapter' };
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  const L = adapter.limits;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: L.maxBufferSize } });
  const errors = [];
  device.addEventListener('uncapturederror', e => errors.push(e.error.message));

  const raw = new Uint8Array(await (await fetch('/payload.bin')).arrayBuffer());
  /* Guide: batch many small transfers into ONE larger transfer. The whole
     artefact goes across in a single copy, padded to a 16-byte boundary so the
     vec4 variant's loads stay aligned. Padding is zero, which is not '='. */
  const quads = Math.ceil(raw.length / 16);
  const padded = new Uint8Array(quads * 16);
  padded.set(raw);
  const words = padded.byteLength / 4;

  const WG = 256; // multiple of 32: guide, block sizes should be warp multiples

  const src = `
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> total : atomic<u32>;
var<workgroup> partial : atomic<u32>;
fn hits(w : u32) -> u32 {
  var c : u32 = 0u;
  if ((w & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 8u) & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 16u) & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 24u) & 0xffu) == 0x3du) { c = c + 1u; }
  return c;
}
@compute @workgroup_size(${WG})
fn scalar(@builtin(global_invocation_id) g : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>) {
  if (l.x == 0u) { atomicStore(&partial, 0u); }
  workgroupBarrier();
  var c : u32 = 0u;
  if (g.x < arrayLength(&src)) { c = hits(src[g.x]); }
  atomicAdd(&partial, c);
  workgroupBarrier();
  if (l.x == 0u) { atomicAdd(&total, atomicLoad(&partial)); }
}`;

  const src4 = `
@group(0) @binding(0) var<storage, read> src : array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> total : atomic<u32>;
var<workgroup> partial : atomic<u32>;
fn hits(w : u32) -> u32 {
  var c : u32 = 0u;
  if ((w & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 8u) & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 16u) & 0xffu) == 0x3du) { c = c + 1u; }
  if (((w >> 24u) & 0xffu) == 0x3du) { c = c + 1u; }
  return c;
}
/* Guide: 128-bit (vec4) loads maximise global memory throughput, and
   consecutive threads touch consecutive 16-byte lanes, so the warp coalesces. */
@compute @workgroup_size(${WG})
fn vectorized(@builtin(global_invocation_id) g : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>) {
  if (l.x == 0u) { atomicStore(&partial, 0u); }
  workgroupBarrier();
  var c : u32 = 0u;
  if (g.x < arrayLength(&src)) {
    let v = src[g.x];
    c = hits(v.x) + hits(v.y) + hits(v.z) + hits(v.w);
  }
  atomicAdd(&partial, c);
  workgroupBarrier();
  if (l.x == 0u) { atomicAdd(&total, atomicLoad(&partial)); }
}
/* Guide: fewer, fatter blocks with a grid-stride loop keep the SMs occupied
   and expose instruction-level parallelism to hide memory latency. */
@compute @workgroup_size(${WG})
fn gridstride(@builtin(global_invocation_id) g : vec3<u32>, @builtin(local_invocation_id) l : vec3<u32>,
              @builtin(num_workgroups) nw : vec3<u32>) {
  if (l.x == 0u) { atomicStore(&partial, 0u); }
  workgroupBarrier();
  let n = arrayLength(&src);
  let stride = nw.x * ${WG}u;
  var c : u32 = 0u;
  var i = g.x;
  loop {
    if (i >= n) { break; }
    let v = src[i];
    c = c + hits(v.x) + hits(v.y) + hits(v.z) + hits(v.w);
    i = i + stride;
  }
  atomicAdd(&partial, c);
  workgroupBarrier();
  if (l.x == 0u) { atomicAdd(&total, atomicLoad(&partial)); }
}`;

  const modScalar = device.createShaderModule({ code: src });
  const modVec = device.createShaderModule({ code: src4 });
  const pipe = (m, e) => device.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: e } });
  const pScalar = pipe(modScalar, 'scalar');
  const pVec = pipe(modVec, 'vectorized');
  const pGrid = pipe(modVec, 'gridstride');

  const outBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const newSrcWriteBuffer = () => {
    const b = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(b, 0, padded);
    return b;
  };
  const newSrcMapped = () => {
    const b = device.createBuffer({ size: padded.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Uint8Array(b.getMappedRange()).set(padded);
    b.unmap();
    return b;
  };

  async function readCount() {
    const e = device.createCommandEncoder();
    e.copyBufferToBuffer(outBuf, 0, readBuf, 0, 4);
    device.queue.submit([e.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(readBuf.getMappedRange().slice(0))[0];
    readBuf.unmap();
    return v;
  }

  async function dispatch(pipeline, srcBuf, groups) {
    device.queue.writeBuffer(outBuf, 0, new Uint32Array([0]));
    await device.queue.onSubmittedWorkDone();
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: srcBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
    const t = performance.now();
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t;
    return { ms, count: await readCount() };
  }

  const variants = [];
  async function measure(name, note, make, pipeline, groups) {
    const runs = [];
    for (let i = 0; i < iters; i++) {
      const tu = performance.now();
      const b = make();
      await device.queue.onSubmittedWorkDone();
      const uploadMs = performance.now() - tu;
      const d = await dispatch(pipeline, b, groups);
      b.destroy?.();
      runs.push({ uploadMs, computeMs: d.ms, endToEndMs: performance.now() - tu, count: d.count });
    }
    variants.push({ name, note, groups, threads: groups * WG, setupMs: 0, runs });
  }

  await measure('A baseline  writeBuffer + u32/thread', 'guide rule: none applied beyond one batched transfer',
    newSrcWriteBuffer, pScalar, Math.ceil(words / WG));
  await measure('B mapped    mappedAtCreation + u32/thread', 'pinned-memory analogue: one write into driver memory',
    newSrcMapped, pScalar, Math.ceil(words / WG));
  await measure('C vector    mappedAtCreation + vec4/thread', '128-bit coalesced loads',
    newSrcMapped, pVec, Math.ceil(quads / WG));
  await measure('D gridstride mappedAtCreation + vec4 + loop', 'occupancy/ILP: 1024 fat workgroups',
    newSrcMapped, pGrid, 1024);

  /* E: Guide High Priority -- minimise host<->device transfer. Upload ONCE,
     then run the kernel `iters` times on the resident buffer. */
  const residentStart = performance.now();
  const resident = newSrcMapped();
  await device.queue.onSubmittedWorkDone();
  const residentSetupMs = performance.now() - residentStart;
  const residentRuns = [];
  for (let i = 0; i < iters; i++) {
    const start = performance.now();
    const d = await dispatch(pVec, resident, Math.ceil(quads / WG));
    residentRuns.push({ uploadMs: 0, computeMs: d.ms, endToEndMs: performance.now() - start, count: d.count });
  }
  variants.push({ name: 'E resident  upload once, N kernels', note: 'Guide High Priority: minimise host<->device transfer',
    groups: Math.ceil(quads / WG), threads: Math.ceil(quads / WG) * WG, setupMs: residentSetupMs, runs: residentRuns });

  return {
    adapter: { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description },
    limits: { maxBufferSize: L.maxBufferSize, maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
      maxComputeWorkgroupsPerDimension: L.maxComputeWorkgroupsPerDimension },
    bytes: raw.length, paddedBytes: padded.byteLength, workgroupSize: WG, variants, errors
  };
}, ITERS); } finally { await browser.close(); }

if (result.error) { console.log('\nGPU UNAVAILABLE: ' + result.error); process.exit(1); }
try {
  if (result.errors?.length) throw Error('WebGPU errors: ' + result.errors.join('; '));
  if (result.bytes !== buf.length || result.paddedBytes !== Math.ceil(buf.length / 16) * 16) throw Error('Input length mismatch');
  if (!Array.isArray(result.variants) || result.variants.length !== 5) throw Error('Expected all five variants');
  for (const [index, variant] of result.variants.entries()) {
    if (!variant.name.startsWith('ABCDE'[index]) || !Number.isFinite(variant.setupMs) || variant.setupMs < 0) throw Error('Invalid variant identity or setup timing');
    if (!Array.isArray(variant.runs) || variant.runs.length !== ITERS) throw Error('Incomplete iteration readback');
    for (const [i, run] of variant.runs.entries()) {
      if (run.count !== cpuCount) throw Error(`${variant.name} iteration ${i + 1} count differs from CPU`);
      for (const key of ['uploadMs', 'computeMs', 'endToEndMs']) if (!Number.isFinite(run[key]) || run[key] < 0) throw Error('Invalid timing: ' + key);
      if (run.endToEndMs < Math.max(run.uploadMs, run.computeMs)) throw Error('Measured total cannot be smaller than a contained stage');
    }
  }
} catch (error) { console.error('GPU BENCHMARK REJECTED: ' + error.message); process.exit(1); }

console.log('\n=== THE ADAPTER THAT ACTUALLY ANSWERED (MEASURED) ===');
console.log(JSON.stringify(result.adapter, null, 1));
console.log('limits ' + JSON.stringify(result.limits));
console.log(`payload ${result.bytes} B padded to ${result.paddedBytes} B, workgroup size ${result.workgroupSize}`);

console.log('\n=== VARIANTS (warm means, iteration 1 discarded) ===');
const rows = [];
for (const v of result.variants) {
  const warm = v.runs.slice(1);
  const mean = k => warm.reduce((a, r) => a + r[k], 0) / warm.length;
  const up = mean('uploadMs'), co = mean('computeMs');
  const ok = v.runs.every(r => r.count === cpuCount);
  const totalMs = v.setupMs + v.runs.reduce((sum, run) => sum + run.endToEndMs, 0);
  rows.push({ name: v.name, note: v.note, threads: v.threads, uploadMs: +up.toFixed(2), computeMs: +co.toFixed(3),
    uploadMBps: up ? Math.round(MB / (up / 1000)) : null, computeMBps: Math.round(MB / (co / 1000)),
    timedStagesMs: +(up + co).toFixed(3), endToEndMs: +(totalMs / v.runs.length).toFixed(3), setupMs: v.setupMs,
    totalMs, iterations: v.runs.length, runs: v.runs, correct: ok, count: v.runs[0].count });
  console.log(`${v.name.padEnd(38)} threads ${String(v.threads).padStart(8)}`
    + `  upload ${up.toFixed(2).padStart(7)}ms ${up ? String(Math.round(MB / (up / 1000))).padStart(6) + ' MB/s' : '     -     '}`
    + `  compute ${co.toFixed(3).padStart(7)}ms ${String(Math.round(MB / (co / 1000))).padStart(6)} MB/s`
    + `  ${ok ? 'count MATCHES CPU' : 'COUNT WRONG (' + v.runs[0].count + ' vs ' + cpuCount + ')'}`);
}

console.log('\n=== WHAT EACH RULE BOUGHT (MEASURED DELTA) ===');
const base = rows[0];
for (const r of rows.slice(1)) {
  console.log(`${r.name.padEnd(38)} compute x${(base.computeMs / r.computeMs).toFixed(2)} vs baseline`
    + `  end-to-end x${(base.endToEndMs / r.endToEndMs).toFixed(2)}`);
}

console.log('\n=== GPU vs CPU ON THE SAME ARTEFACT (MEASURED) ===');
const best = rows.reduce((a, b) => (b.computeMs < a.computeMs ? b : a));
const bestEnd = rows.reduce((a, b) => (b.endToEndMs < a.endToEndMs ? b : a));
console.log(`CPU scalar loop           ${cpuScalarMs.toFixed(1)}ms  (${(MB / (cpuScalarMs / 1000)).toFixed(0)} MB/s)`);
console.log(`CPU Buffer.indexOf        ${cpuIndexOfMs.toFixed(1)}ms  (${(MB / (cpuIndexOfMs / 1000)).toFixed(0)} MB/s)`);
console.log(`GPU best compute-only     ${best.computeMs.toFixed(3)}ms  (${best.computeMBps} MB/s)  [${best.name.trim()}]`);
console.log(`GPU best end-to-end       ${bestEnd.endToEndMs.toFixed(3)}ms  [${bestEnd.name.trim()}] (setup + all iteration wall times, divided by iteration count; includes upload, reset, dispatch and readback)`);
console.log(`\ncompute-only  GPU vs CPU scalar   x${(cpuScalarMs / best.computeMs).toFixed(1)}`);
console.log(`compute-only  GPU vs CPU indexOf  x${(cpuIndexOfMs / best.computeMs).toFixed(1)}`);
console.log(`end-to-end    GPU vs CPU scalar   x${(cpuScalarMs / bestEnd.endToEndMs).toFixed(2)}`);
console.log(`end-to-end    GPU vs CPU indexOf  x${(cpuIndexOfMs / bestEnd.endToEndMs).toFixed(2)}`);

console.log('\nJSON ' + JSON.stringify({ provenance, cpuScalarMs, cpuIndexOfMs, cpuCount, adapter: result.adapter, rows,
  bestComputeVariant: best.name, bestEndToEndVariant: bestEnd.name,
  timingScope: 'Compute/upload columns discard iteration 1. endToEndMs includes setup and every measured iteration, including readback, divided by count. File I/O, browser startup, pipeline creation and CPU verification are outside this GPU operation timing.' }));
