"""Bounded cross-repository GPU similarity screening using the attributed Claude harness."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--reviews', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
repo = Path(__file__).resolve().parents[1]
corpus = a.out / 'source-sample'; corpus.mkdir(exist_ok=True)
reports = [json.loads(x.read_text(encoding='utf8')) for x in sorted(a.reviews.glob('*/cartridge.json'))]
selected = []; total = 0
# Fair round-robin sampling: no one monolith consumes the whole budget.
for index in range(40):
    for r in reports:
        candidates = [f for f in r['files'] if Path(f['path']).suffix in ('.js', '.mjs', '.cjs')]
        if index >= len(candidates): continue
        f = candidates[index]
        if total + f['bytes'] > 64_000_000 or len(selected) >= 1000: continue
        raw = subprocess.check_output(['git', '-C', str(repo.parent / r['repository']), 'cat-file', 'blob', f['blob']], timeout=30)
        assert hashlib.sha256(raw).hexdigest() == f['sha256']
        name = f'{len(selected):04d}-{r["repository"]}.mjs'
        (corpus / name).write_bytes(raw)
        selected.append({'sample': name, 'repository': r['repository'], 'commit': r['commit'], **f})
        total += len(raw)
source = repo / 'claude/analyse-corpus-gpu.mjs'
raw = source.read_bytes(); code = raw.decode('utf8')
old = 'chromium.launch({ headless: false, args: ['
assert code.count(old) == 1
# Launch-only adapter: preserve the attributed shaders and all measured logic.
adapted = code.replace(old, "chromium.launch({ headless: true, channel: 'chrome', args: [")
harness = a.out / 'attributed-corpus-harness.mjs'; harness.write_text(adapted, encoding='utf8')
(a.out / 'provenance.json').write_text(json.dumps({'source': str(source), 'sourceSha256': hashlib.sha256(raw).hexdigest(), 'adapter': 'Headless installed Chrome only; no shader changes', 'files': selected, 'bytes': total, 'scope': 'Bounded committed-source sample; similarity selects candidates, not proven duplication'}, indent=2), encoding='utf8')
if len(selected) < 2: raise ValueError('At least two source files required')
subprocess.run(['node', str(harness), str(corpus), '--out', str(a.out), '--top', '20'], check=True, timeout=120)
r = json.loads((a.out / 'corpus-gpu-analysis.json').read_text(encoding='utf8'))
assert r['verification']['histogramMatchesCpu'] is True
lookup = {f['sample']: f for f in selected}
brief = '# GPU cross-repository review cartridge\n\n' + f'{len(selected)} sampled files, {total} bytes. GPU histograms verified against CPU.\n\n## Exact duplicates\n\n' + str(r['exactDuplicates']['groups']) + ' exact groups in this sample.\n\n## Candidates for review, NOT proven duplicate code\n\n'
for pair in r['nearDuplicates']['top'][:12]:
    left, right = lookup[pair['a']], lookup[pair['b']]
    brief += f'- {left["repository"]}/{left["path"]} ↔ {right["repository"]}/{right["path"]}: histogram cosine {pair["cosine"]}; verify semantics before any extraction.\n'
(a.out / 'REVIEW.md').write_text(brief, encoding='utf8')
print('JSON ' + json.dumps({'gpuCorpusVerified': True, 'files': len(selected), 'bytes': total, 'card': str(a.out / 'REVIEW.md')}))
