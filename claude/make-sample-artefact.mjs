/**
 * A STAND-IN FOR THE TELEPRINT.
 * ---------------------------------------------------------------------------
 * The artefact the measured numbers came from is a 27,568,130-byte GridAtlas
 * teleprint source dump. It is evidence, not source, so it is NOT committed
 * here. This writes a same-shaped synthetic artefact so the benches -- and the
 * CI workflow, which has no teleprint -- are runnable anywhere:
 *
 *   - "===== BEGIN <name> =====" / "===== END <name> =====" section markers,
 *     which is what the CPU bench's /^=+ (BEGIN|END) /gm counts
 *   - runs of '=' between sections, which is what the GPU bench reduces over
 *   - deterministic (seeded), so two runs on the same machine are comparable
 *
 * It is a SHAPE match, not the artefact. Numbers produced from it are not the
 * numbers in claude/results/ -- those name the real teleprint and this machine.
 *
 *   node make-sample-artefact.mjs [out.txt] [sizeMB]
 */
import { createWriteStream } from 'node:fs';
import { once } from 'node:events';

const out = process.argv[2] || 'sample-artefact.txt';
const targetBytes = Math.round((Number(process.argv[3]) || 26) * 1024 * 1024);

/* mulberry32: tiny, seeded, so the file is byte-identical run to run. */
let seed = 0x9e3779b9;
const rnd = () => {
  seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const WORDS = ['grid', 'atlas', 'node', 'substation', 'feeder', 'export', 'capacity',
  'circuit', 'transformer', 'headroom', 'REPD', 'geodesy', 'bearing', 'radius',
  'polyline', 'cartridge', 'sentinel', 'teleprint', 'lane', 'manifest'];

const ws = createWriteStream(out);
const write = async s => { if (!ws.write(s)) await once(ws, 'drain'); };

let written = 0;
let section = 0;
while (written < targetBytes) {
  const name = `SECTION-${String(section).padStart(4, '0')}-${WORDS[section % WORDS.length].toUpperCase()}`;
  const begin = `===== BEGIN ${name} =====\n`;
  await write(begin); written += begin.length;

  /* A body of plausible source-dump lines, plus the occasional rule line of
     '=' so the byte histogram is not degenerate. */
  const lines = 400 + Math.floor(rnd() * 800);
  let body = '';
  for (let i = 0; i < lines; i++) {
    if (i % 37 === 0) { body += '='.repeat(60) + '\n'; continue; }
    const n = 4 + Math.floor(rnd() * 10);
    const parts = [];
    for (let j = 0; j < n; j++) parts.push(WORDS[Math.floor(rnd() * WORDS.length)]);
    body += `${String(i).padStart(5, ' ')}  ${parts.join(' ')} = ${(rnd() * 1000).toFixed(3)}\n`;
  }
  await write(body); written += body.length;

  const end = `===== END ${name} =====\n`;
  await write(end); written += end.length;
  section += 1;
}
ws.end();
await once(ws, 'finish');
console.log(`wrote ${out}  ${written} bytes (${(written / 1048576).toFixed(2)} MB)  ${section} sections `
  + `(${section * 2} BEGIN/END markers)`);
