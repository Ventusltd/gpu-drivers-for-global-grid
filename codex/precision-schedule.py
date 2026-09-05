"""96 precision-review timeboxes. Hourly queue refresh; never substitutes time for evidence."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

GROUPS = [
 ('Release composition', 'gridatlas', 'Does the current release gate reject removal of every required cartridge?'),
 ('Cable geometry', 'globalgrid2050', 'What is the exact loading and behavior impact of the confirmed app.js syntax defect?'),
 ('GRID / SUBS controls', 'gridatlas', 'Do controls and minimised layers behave correctly across desktop and mobile?'),
 ('Project deep links', 'ventus-grid-engine', 'Does Pipeline News select the intended Atlas project and produce a visible outcome?'),
 ('App-only PDF', 'teleprinter', 'Does the real print action preserve visible app content without screen sharing?'),
 ('GIS SLD isolation', 'gis-sld-sandbox', 'Does the original sandbox retain its behavior as an independently failing Atlas layer?'),
 ('Source print integrity', 'testcode', 'Does a source download contain the complete pinned dependency closure in the tested environment?'),
 ('GPU evidence', 'gpu-drivers-for-global-grid', 'Are useful GPU results correct end-to-end and attributable to the actual adapter and input?'),
]
PHASES = [
 ('Pin identities', 'Record local HEAD, remote main, working-tree state and origin; do not silently equate local and shipped code.'),
 ('Resolve entrypoints', 'Identify and read the complete actual entrypoint and manifest at the selected commit.'),
 ('Resolve dependencies', 'Account for the complete relevant import/workflow-script closure; unresolved dependencies block completion.'),
 ('Read contracts', 'Record intended inputs, outputs, invariants and applicable owner instructions.'),
 ('Trace implementation', 'Read complete relevant functions and document the behavior path with exact source locations.'),
 ('Establish baseline', 'Reproduce the behavior on the pinned unmodified implementation and retain actual evidence.'),
 ('Challenge hypothesis', 'Test the strongest alternative explanation, including environment and reviewer errors.'),
 ('Negative fixture', 'Construct a targeted failing input or missing-module case and show the check detects it.'),
 ('Boundary cases', 'Test scope-specific edge cases and independent plugin failure behavior.'),
 ('Cross-check evidence', 'Independently verify source identity, outputs and any measured numerical claims.'),
 ('Propose minimal change', 'Describe the smallest justified owner change, dependency effects and rollback; no untested deployment.'),
 ('Issue decision card', 'Provide a short evidence-backed conclusion with unresolved limits and links to the complete review packet.'),
]


def write(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    if len(raw) > 8_000_000: raise ValueError('Partition budget exceeded')
    tmp = path.with_suffix('.tmp'); tmp.write_bytes(raw); tmp.replace(path)


def create_plan():
    tasks = []
    for group, (title, repo, question) in enumerate(GROUPS):
        for phase, (name, criterion) in enumerate(PHASES):
            number = group * 12 + phase + 1
            tasks.append({'id': f'R{number:02d}', 'group': title, 'owner': repo, 'question': question,
                          'phase': name, 'acceptance': criterion, 'timeboxMinutes': 5,
                          'dependsOn': [f'R{number-1:02d}'] if phase else [], 'priority': group + 1})
    return {'schema': 'ventus.precision-plan.v1', 'tasks': tasks, 'hours': 8,
            'scope': '96 five-minute planning units, not a promise of 96 completed engineering reviews. Carry unresolved work forward.'}


def pin(task, estate, out):
    root = estate / task['owner']
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], timeout=45).decode('utf8', errors='replace').strip()
    proof = {'repository': task['owner'], 'localCommit': git('rev-parse', 'HEAD'), 'remote': git('remote', 'get-url', 'origin'),
             'remoteMain': git('ls-remote', 'origin', 'refs/heads/main'), 'workingTree': git('status', '--short'),
             'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'scope': 'Identity checkpoint only; no application verification.'}
    if len(proof['localCommit']) != 40 or len(proof['remoteMain'].split()[0]) != 40: raise ValueError('Unresolved commit identity')
    proof_path = out / 'evidence' / (task['id'] + '.json'); write(proof_path, proof)
    write(out / 'receipts' / (task['id'] + '.json'), {'task': task['id'], 'status': 'verified', 'reviewer': 'identity runner',
          'acceptance': task['acceptance'], 'evidence': [{'path': str(proof_path), 'sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest()}]})


def refresh(plan, out, hour):
    states = {}; errors = []
    for task in plan['tasks']:
        states[task['id']] = 'pending'
        receipt = out / 'receipts' / (task['id'] + '.json')
        if not receipt.exists(): continue
        try:
            r = json.loads(receipt.read_text(encoding='utf8'))
            assert r['task'] == task['id'] and r['acceptance'] == task['acceptance'] and r['reviewer']
            assert r['status'] == 'verified' and r['evidence']
            for e in r['evidence']:
                assert hashlib.sha256(Path(e['path']).read_bytes()).hexdigest() == e['sha256']
            assert all(states[d] == 'verified' for d in task['dependsOn'])
            states[task['id']] = 'verified'
        except Exception as error: errors.append({'task': task['id'], 'error': type(error).__name__})
    ready = [t for t in plan['tasks'] if states[t['id']] != 'verified' and all(states[d] == 'verified' for d in t['dependsOn'])]
    ready.sort(key=lambda t: (t['priority'], t['id']))
    record = {'at': dt.datetime.now(dt.timezone.utc).isoformat(), 'hour': hour, 'states': states,
              'nextReady': [t['id'] for t in ready[:12]], 'receiptErrors': errors,
              'schedulerScope': 'Local dependency/evidence checks and prioritisation only. It does not perform unimplemented engineering review or invoke an AI.'}
    write(out / 'PROGRESS.json', record); write(out / f'hour-{hour:02d}-queue.json', record)
    lines = ['# Precision review progress', '', 'Five minutes is a timebox, not proof of completion. These groups are review domains, not simultaneous full-load jobs.', '']
    for title, _, question in GROUPS:
        ts = [t for t in plan['tasks'] if t['group'] == title]; done = sum(states[t['id']] == 'verified' for t in ts)
        lines.append(f'- {title}: ' + '█' * done + '░' * (12 - done) + f' {done}/12 — {question}')
    lines += ['', '## Ready next', ''] + [f'- {t["id"]}: {t["group"]} / {t["phase"]}. {t["acceptance"]}' for t in ready[:12]]
    (out / 'PROGRESS.md').write_text('\n'.join(lines) + '\n', encoding='utf8')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True); p.add_argument('--watch', action='store_true'); p.add_argument('--pin', action='store_true')
    args = p.parse_args(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    for name in ('evidence', 'receipts'): (out / name).mkdir(exist_ok=True)
    plan = create_plan(); write(out / 'PLAN.json', plan)
    (out / 'SCOPES.md').write_text('# 96 precision scopes\n\n' + '\n\n'.join(f'## {t["id"]} — {t["group"]}: {t["phase"]}\n\nQuestion: {t["question"]}\n\nCompletion: {t["acceptance"]}\n\nDependency: {", ".join(t["dependsOn"]) or "none"}. Timebox: 5 minutes; carry forward if unresolved.' for t in plan['tasks']), encoding='utf8')
    if args.pin:
        errors = []
        for task in plan['tasks']:
            if task['phase'] == 'Pin identities':
                try: pin(task, Path(__file__).resolve().parents[2], out)
                except Exception as error: errors.append({'task': task['id'], 'error': str(error)})
        write(out / 'PIN-ERRORS.json', errors)
    start = time.monotonic(); last_hour = -1
    while True:
        hour = int((time.monotonic() - start) // 3600)
        if hour != last_hour: refresh(plan, out, hour); last_hour = hour
        if not args.watch or hour >= 8 or (out / 'STOP').exists(): break
        time.sleep(30)


if __name__ == '__main__': main()
