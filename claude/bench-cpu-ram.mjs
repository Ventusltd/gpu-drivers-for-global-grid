/**
 * HOW FAST CAN THIS MACHINE ACTUALLY EAT A TELEPRINT?
 * ---------------------------------------------------------------------------
 * Offline. No network, no model. It loads a large text artefact (the original
 * run used the real 27.5 MB GridAtlas teleprint the architect produced on
 * 2026-09-05) and measures several different things, each of which stresses a
 * different part of the box:
 *
 *   read        cold-ish file read           -> storage + page cache
 *   memcpy      buffer copy in RAM           -> DDR5 bandwidth, one core
 *   sha256      hash the whole file          -> CPU, compute bound
 *   parse       find every BEGIN/END section -> CPU, branch + string bound
 *
 * Then it runs the SAME work on 1..N worker threads, each worker holding its
 * OWN copy of the file, because "load it x several threads" means N resident
 * copies, not one shared buffer sliced N ways. That is the honest test of what
 * a DC machine does when N readers each open the artefact.
 *
 * MEASURED, not asserted: every number below is a timing this process took.
 *
 *   node bench-cpu-ram.mjs [artefact.txt] [--threads 1,2,4,8,16,20]
 *
 * The artefact path is optional. Resolution order:
 *   1. argv[2]
 *   2. $BENCH_ARTEFACT
 *   3. ./sample-artefact.txt next to this file (make it with make-sample-artefact.mjs)
 * The 27.5 MB teleprint itself is deliberately NOT committed: it is evidence,
 * not source.
 */
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readFileSync, statSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import os from 'node:os';

const MB = 1024 * 1024;

/* The unit of work. Deliberately three separate phases so a slow one is
   visible rather than averaged into the others. */
function work(buf) {
  const t = {};
  let m0 = process.hrtime.bigint();
  const copy = Buffer.allocUnsafe(buf.length);
  buf.copy(copy);
  t.memcpyMs = Number(process.hrtime.bigint() - m0) / 1e6;

  m0 = process.hrtime.bigint();
  const digest = createHash('sha256').update(buf).digest('hex');
  t.sha256Ms = Number(process.hrtime.bigint() - m0) / 1e6;

  m0 = process.hrtime.bigint();
  const text = buf.toString('utf8');
  t.decodeMs = Number(process.hrtime.bigint() - m0) / 1e6;

  m0 = process.hrtime.bigint();
  let sections = 0;
  const re = /^=+ (BEGIN|END) /gm;
  while (re.exec(text) !== null) sections += 1;
  t.parseMs = Number(process.hrtime.bigint() - m0) / 1e6;

  t.sections = sections;
  t.digest16 = digest.slice(0, 16);
  return t;
}

/* Path with a sensible fallback, so the bench is runnable on a machine that
   does not have the (uncommitted) teleprint. */
function resolveArtefact(argvPath) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const candidates = [argvPath, process.env.BENCH_ARTEFACT, path.join(here, 'sample-artefact.txt')];
  for (const c of candidates) if (c && existsSync(c)) return c;
  console.error('no artefact found. Pass a path, set $BENCH_ARTEFACT, or generate one:');
  console.error('  node make-sample-artefact.mjs sample-artefact.txt 26');
  process.exit(2);
}

if (!isMainThread) {
  /* Each worker reads the file ITSELF. No shared buffer, no transfer: this is
     N independent readers, which is the case being measured. */
  const t0 = process.hrtime.bigint();
  const buf = readFileSync(workerData.file);
  const readMs = Number(process.hrtime.bigint() - t0) / 1e6;
  const r = work(buf);
  r.readMs = readMs;
  r.rssMB = process.memoryUsage().rss / MB;
  parentPort.postMessage(r);
} else {
  const argvPath = process.argv[2] && !process.argv[2].startsWith('--') ? process.argv[2] : undefined;
  const file = resolveArtefact(argvPath);
  const ti = process.argv.indexOf('--threads');
  const LADDER = ti > 0 ? process.argv[ti + 1].split(',').map(Number) : [1, 2, 4, 8, 16, 20];
  const bytes = statSync(file).size;

  const machine = {
    cpu: os.cpus()[0].model.trim(),
    logicalCores: os.cpus().length,
    totalRamGB: +(os.totalmem() / 1024 / 1024 / 1024).toFixed(2),
    freeRamGB: +(os.freemem() / 1024 / 1024 / 1024).toFixed(2),
    node: process.version,
    file,
    fileBytes: bytes,
    fileMB: +(bytes / MB).toFixed(2)
  };
  console.log(JSON.stringify(machine, null, 1));

  /* Single-threaded baseline, run three times: the first is cold, the later
     two are warm. Reporting only the fastest would flatter the page cache. */
  console.log('\n=== SINGLE THREAD, 3 passes (MEASURED) ===');
  const single = [];
  for (let i = 0; i < 3; i++) {
    const t0 = process.hrtime.bigint();
    const buf = readFileSync(file);
    const readMs = Number(process.hrtime.bigint() - t0) / 1e6;
    const r = work(buf);
    r.readMs = readMs;
    single.push(r);
    console.log(`pass ${i + 1}  read ${readMs.toFixed(1)}ms (${(bytes / MB / (readMs / 1000)).toFixed(0)} MB/s)`
      + `  memcpy ${r.memcpyMs.toFixed(1)}ms (${(bytes / MB / (r.memcpyMs / 1000)).toFixed(0)} MB/s)`
      + `  sha256 ${r.sha256Ms.toFixed(1)}ms (${(bytes / MB / (r.sha256Ms / 1000)).toFixed(0)} MB/s)`
      + `  utf8-decode ${r.decodeMs.toFixed(1)}ms  parse ${r.parseMs.toFixed(1)}ms  sections ${r.sections}`);
  }

  console.log('\n=== THREAD LADDER: N independent readers, each with its own copy (MEASURED) ===');
  const ladder = [];
  for (const n of LADDER) {
    const t0 = process.hrtime.bigint();
    const results = await Promise.all(Array.from({ length: n }, () => new Promise((res, rej) => {
      const w = new Worker(new URL(import.meta.url), { workerData: { file } });
      w.on('message', m => { res(m); w.terminate(); });
      w.on('error', rej);
    })));
    const wallMs = Number(process.hrtime.bigint() - t0) / 1e6;
    const totalMB = (bytes * n) / MB;
    const row = {
      threads: n,
      wallMs: +wallMs.toFixed(1),
      totalMBProcessed: +totalMB.toFixed(1),
      aggregateMBps: +(totalMB / (wallMs / 1000)).toFixed(0),
      perThreadMeanSha256Ms: +(results.reduce((a, r) => a + r.sha256Ms, 0) / n).toFixed(1),
      perThreadMeanMemcpyMs: +(results.reduce((a, r) => a + r.memcpyMs, 0) / n).toFixed(1),
      peakWorkerRssMB: +Math.max(...results.map(r => r.rssMB)).toFixed(0),
      sectionsAgree: new Set(results.map(r => r.sections)).size === 1,
      digestsAgree: new Set(results.map(r => r.digest16)).size === 1
    };
    ladder.push(row);
    console.log(`${String(n).padStart(3)} threads  wall ${row.wallMs.toFixed(0).padStart(6)}ms`
      + `  aggregate ${String(row.aggregateMBps).padStart(5)} MB/s`
      + `  mean sha256 ${String(row.perThreadMeanSha256Ms).padStart(7)}ms`
      + `  mean memcpy ${String(row.perThreadMeanMemcpyMs).padStart(6)}ms`
      + `  peak worker RSS ${String(row.peakWorkerRssMB).padStart(4)} MB`
      + `  identical: sections=${row.sectionsAgree} digest=${row.digestsAgree}`);
  }

  const base = ladder.find(r => r.threads === 1) || ladder[0];
  console.log('\n=== SCALING vs 1 thread (MEASURED) ===');
  for (const r of ladder) {
    console.log(`${String(r.threads).padStart(3)} threads  speedup x${(r.aggregateMBps / base.aggregateMBps).toFixed(2)}`
      + `  efficiency ${((r.aggregateMBps / base.aggregateMBps) / r.threads * 100).toFixed(0)}%`);
  }
  console.log('\nJSON ' + JSON.stringify({ machine, single, ladder }));
}
