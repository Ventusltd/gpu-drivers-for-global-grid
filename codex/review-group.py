"""Parallel owner reviews, with one shared CPU worker budget."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', action='append', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--workers', type=int, default=4)
p.add_argument('--offset', type=int, default=0)
p.add_argument('--limit', type=int, default=240)
a = p.parse_args()
parallel = min(3, len(a.root), max(1, a.workers))
per_repo = max(1, a.workers // parallel)


def run(root):
    out = a.out / root.name
    subprocess.run([sys.executable, str(Path(__file__).with_name('review-repo.py')), '--root', str(root), '--out', str(out), '--workers', str(per_repo),'--offset',str(a.offset),'--limit',str(a.limit)], check=True)
    report = json.loads((out / 'cartridge.json').read_text(encoding='utf8'))
    assert report['repository'] == root.name and len(report['commit']) == 40
    return {'repository': report['repository'], 'commit': report['commit'], 'parseFailures': len(report['parseFailures']), 'inspected': report['inspected'], 'truncated': report['truncated']}


with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as pool:
    reports = list(pool.map(run, a.root))
print('JSON ' + json.dumps({'reviews': reports}))
