"""Finish a finite public-source learning queue; never execute target-owned code."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).parent
QUEUE = HERE / 'learning-queue-20260906.json'


def jobs():
    rows = json.loads(QUEUE.read_text(encoding='utf8'))['jobs']
    seen = set()
    for row in rows:
        name = row['repository']
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', name) or name in seen:
            raise ValueError('Unsafe or duplicate repository')
        seen.add(name)
        if not re.fullmatch(r'[0-9a-f]{40}', row['commit']) or row['public'] is not True:
            raise ValueError('Public commit identity required')
        if row['mode'] not in ('learn', 'stability') or not isinstance(row['seed'], int):
            raise ValueError('Invalid job')
    return rows


def verify_model(row, base):
    manifest = json.loads((base / 'manifest.json').read_text())
    if manifest['commit'] != row['commit'] or manifest['repository'] != row['repository']:
        raise ValueError('Model source mismatch')
    for expected, actual in [('expectedSelected', 'selected'), ('expectedDimensions', 'dimensions')]:
        if expected in row and row[expected] != manifest[actual]:
            raise ValueError('Recreated model dimensions differ from local baseline')
    if 'expectedFilesSha256' in row:
        if hashlib.sha256((base / 'files.json').read_bytes()).hexdigest() != row['expectedFilesSha256']:
            raise ValueError('Recreated source inventory differs from local baseline')


def verify_receipt(row, receipt):
    if any(receipt.get(k) != row[k] for k in ('repository', 'commit', 'seed', 'mode')):
        raise ValueError('Receipt belongs to another job')
    if receipt.get('status') != 'complete' or not receipt.get('sourceVerified'):
        raise ValueError('Incomplete or unverified receipt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--matrix', action='store_true')
    p.add_argument('--repository')
    p.add_argument('--out', type=Path, default=Path('artifacts'))
    p.add_argument('--collect', type=Path)
    a = p.parse_args()
    rows = jobs()
    if a.matrix:
        print(json.dumps({'include': rows}, separators=(',', ':')))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    if a.collect:
        receipts = [json.loads(p.read_text()) for p in a.collect.rglob('receipt.json')]
        if len(receipts) != len(rows):
            raise ValueError('Missing or duplicate receipts')
        for row in rows:
            matches = [r for r in receipts if r.get('repository') == row['repository']]
            if len(matches) != 1:
                raise ValueError('Missing or duplicate repository')
            verify_receipt(row, matches[0])
        (a.out / 'SUMMARY.json').write_text(json.dumps({'completed': len(rows), 'failed': 0, 'jobs': receipts,
            'scope': 'Finite lexical learning queue. Not application correctness or GPU performance.'}, indent=2) + '\n')
        print(f'Completed {len(rows)}/{len(rows)} pinned jobs')
        return
    row = next(r for r in rows if r['repository'] == a.repository)
    root = Path('owners') / row['repository']
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != row['commit']:
        raise ValueError('Checkout does not match pinned queue')
    base = a.out / 'model'
    command = [sys.executable, str(HERE / 'learn-repo.py'), '--root', str(root), '--commit', row['commit']]
    subprocess.run(command + ['--out', str(base), '--seed', str(row.get('baselineSeed', row['seed']))], check=True, timeout=900)
    verify_model(row, base)
    result_dir = base
    if row['mode'] == 'stability':
        result_dir = a.out / 'stability'
        subprocess.run(command + ['--out', str(result_dir), '--seed', str(row['seed']), '--reuse', str(base)], check=True, timeout=300)
    result = json.loads((result_dir / 'clusters.json').read_text())
    if result['commit'] != row['commit'] or result['seed'] != row['seed']:
        raise ValueError('Result identity mismatch')
    receipt = dict(row, status='complete', sourceVerified=True, resultSha256=hashlib.sha256((result_dir / 'clusters.json').read_bytes()).hexdigest(),
                   completedAt=dt.datetime.now(dt.timezone.utc).isoformat(),
                   scope='Public committed source; local model reconstructed. Cross-platform timings are not laptop benchmarks.')
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
