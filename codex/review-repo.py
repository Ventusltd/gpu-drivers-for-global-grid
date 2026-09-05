"""Produce a bounded, commit-pinned source review cartridge without executing source."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], timeout=60)


def inspect(root, entry):
    mode, kind, sha, size, name = entry
    raw = git(root, 'cat-file', 'blob', sha)
    text = raw.decode('utf8', errors='replace')
    result = {'path': name, 'blob': sha, 'bytes': len(raw), 'lines': len(text.splitlines()), 'sha256': hashlib.sha256(raw).hexdigest()}
    suffix = Path(name).suffix.lower()
    try:
        if suffix == '.py': ast.parse(text, filename=name); result['parse'] = 'pass'
        elif suffix == '.json': json.loads(text); result['parse'] = 'pass'
        elif suffix in ('.js', '.mjs', '.cjs'):
            # Parse stdin only; never import/run target-owned code.
            command = ['node', '--check']
            if suffix == '.mjs' or re.search(r'(?m)^\s*(?:import |export )', text): command += ['--input-type=module']
            checked = subprocess.run(command, input=raw, capture_output=True, timeout=15)
            result['parse'] = 'pass' if checked.returncode == 0 else 'failed'
            if checked.returncode: result['error'] = checked.stderr.decode('utf8', errors='replace')[:1200]
        else: result['parse'] = 'not-applicable'
    except Exception as error:
        result.update(parse='failed', error=str(error)[:1200])
    result['imports'] = sorted(set(re.findall(r'''(?:from\s*|require\(\s*|import\(\s*)['"]([^'"]+)''', text)))[:40]
    result['symbols'] = re.findall(r'(?m)^\s*(?:export\s+)?(?:async\s+)?(?:function|class|def)\s+(\w+)', text)[:40]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args(); args.root = args.root.resolve(); args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); head = git(args.root, 'rev-parse', 'HEAD').decode().strip()
    entries = []
    for row in git(args.root, 'ls-tree', '-rlz', head).split(b'\0'):
        if not row: continue
        meta, name = row.split(b'\t', 1); fields = meta.decode().split(); name = name.decode('utf8', errors='replace')
        if len(fields) == 4 and fields[1] == 'blob': entries.append((*fields, name))
    eligible = [e for e in entries if Path(e[4]).suffix.lower() in ('.js', '.mjs', '.cjs', '.py', '.json')
                and int(e[3]) <= 1_000_000 and not any(x in e[4].lower().split('/') for x in ('node_modules', 'vendor', 'data', 'results', 'versions', 'receipts'))
                and not re.search(r'(?:^|/)\d{12}(?:/|$)', e[4])]
    # Largest first exposes monolith boundaries; each blob parsed once per repository.
    eligible.sort(key=lambda e: (-int(e[3]), e[4])); unique = {}; duplicate = {}
    for e in eligible:
        unique.setdefault(e[2], e); duplicate.setdefault(e[2], []).append(e[4])
    selected = list(unique.values())[:240]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(12, args.workers))) as pool:
        results = list(pool.map(lambda e: inspect(args.root, e), selected))
    failures = [r for r in results if r['parse'] == 'failed']
    candidates = [dict(path=r['path'], bytes=r['bytes'], lines=r['lines'], symbols=r['symbols'], imports=r['imports'],
                       decision='Review extraction boundaries; size alone does not prove separability') for r in results if r['lines'] > 800][:12]
    report = {'schema': 'ventus.source-review-cartridge.v1', 'repository': args.root.name, 'commit': head,
              'scope': 'Committed HEAD only, not dirty work. Bounded source parse and lexical references; no runtime or application correctness claim.',
              'trackedFiles': len(entries), 'eligibleFiles': len(eligible), 'uniqueEligibleBlobs': len(unique), 'inspected': len(results),
              'truncated': len(selected) < len(unique), 'seconds': round(time.monotonic() - started, 2),
              'parseFailures': failures, 'extractionCandidates': candidates,
              'exactBlobDuplicates': [dict(blob=k, paths=v[:30], copies=len(v)) for k, v in duplicate.items() if len(v) > 1][:100],
              'files': results}
    raw = (json.dumps(report, indent=2) + '\n').encode()
    if len(raw) > 8_000_000: raise ValueError('Review partition over 8 MB')
    (args.out / 'cartridge.json').write_bytes(raw)
    brief = f'# {args.root.name}: review cartridge\n\nCommit `{head}`. Inspected {len(results)}/{len(unique)} eligible unique blobs; {len(failures)} parse failures.\n\n'
    brief += '## Inspect first\n\n' + '\n'.join('- `' + r['path'] + '`: ' + r.get('error', '')[:180].replace('\n', ' ') for r in failures[:8])
    brief += '\n\n## Candidate extraction boundaries\n\n' + '\n'.join(f'- `{r["path"]}`: {r["lines"]} lines; symbols: ' + ', '.join(r['symbols'][:8]) for r in candidates[:6])
    brief += '\n\nThese are review proposals, not generated or accepted application cartridges. Read cartridge.json for hashes, exclusions and coverage.\n'
    (args.out / 'REVIEW.md').write_text(brief, encoding='utf8')
    print(json.dumps({'repository': args.root.name, 'inspected': len(results), 'parseFailures': len(failures)}))


if __name__ == '__main__': main()
