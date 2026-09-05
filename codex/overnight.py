"""Bounded local CPU/WebGPU soak study; measurements are benchmarks, not source compilation."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import psutil

GIB = 1024 ** 3
HERE = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sample():
    battery = psutil.sensors_battery()
    record = {'at': now(), 'availableGiB': psutil.virtual_memory().available / GIB,
              'cpuPercent': psutil.cpu_percent(), 'plugged': battery.power_plugged if battery else None,
              'cpuTemperature': None, 'cpuTemperatureNote': 'No validated CPU temperature sensor configured; CPU affinity and memory controls are enforced.'}
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
        temp, use, used, total = map(float, r.stdout.splitlines()[0].split(','))
        record.update(gpuTemperature=temp, gpuPercent=use, gpuMemoryUsedMiB=used, gpuMemoryTotalMiB=total)
    except Exception as error:
        record['gpuError'] = type(error).__name__
    return record


def stop_child(child):
    try:
        process = psutil.Process(child.pid)
        descendants = process.children(recursive=True)
        for p in descendants:
            try: p.terminate()
            except psutil.Error: pass
        process.terminate()
        _, alive = psutil.wait_procs(descendants + [process], timeout=3)
        for p in alive:
            try: p.kill()
            except psutil.Error: pass
    except psutil.NoSuchProcess:
        pass


def guard(s):
    if s['plugged'] is False: return 'AC power disconnected'
    if s['availableGiB'] < 2: return 'Less than 2 GiB available RAM'
    if s.get('gpuTemperature', 100) >= 78: return 'GPU temperature high or sensor unavailable'
    if s.get('gpuMemoryTotalMiB', 0) - s.get('gpuMemoryUsedMiB', 0) < 1024: return 'Less than 1 GiB GPU memory available'
    return None


def write_json(path, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    if len(raw) > 8_000_000: raise ValueError('Output partition cap reached')
    temp = path.with_suffix('.tmp')
    temp.write_bytes(raw)
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    args = parser.parse_args()
    if not 0 < args.hours <= 8: parser.error('Duration must be 0 < hours <= 8')
    if not 0 < args.artifact.stat().st_size <= 40_000_000: parser.error('Input must be 1..40 MB')
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'RUN.json').exists(): parser.error('Use a new evidence directory per run')
    started = time.monotonic(); deadline = started + args.hours * 3600
    env = dict(os.environ, BENCH_CHANNEL='chrome', NODE_OPTIONS='--max-old-space-size=512')
    affinity = psutil.Process().cpu_affinity()[:max(1, min(16, (psutil.cpu_count() or 2) - 4))]
    record = {'schema': 'ventus.local-soak.v1', 'startedAt': now(), 'pid': os.getpid(), 'hours': args.hours,
              'artifact': str(args.artifact), 'inputBytes': args.artifact.stat().st_size,
              'sha256': hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
              'runnerCommit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
              'scope': 'Repeated CPU hash/decode/marker scans and WebGPU byte-count variants checked against CPU. Not full code compilation or application correctness.',
              'limits': {'cpuAffinity': affinity, 'maxWorkers': len(affinity), 'minFreeRamGiB': 2, 'gpuStopC': 78,
                         'gpuResumeC': 70, 'minFreeVramMiB': 1024, 'maxTaskSeconds': 180, 'maxEvidenceBytes': 500_000_000},
              'stop': 'Create a file named STOP in this evidence directory; supervisor checks every 5 seconds.',
              'completed': 0, 'failed': 0, 'interrupted': 0, 'interruptedJobs': []}
    write_json(out / 'RUN.json', record)
    owners = json.loads((HERE / 'codex/review-owners.json').read_text(encoding='utf8'))
    queue = [HERE.parent / name for name in owners if (HERE.parent / name / '.git').exists()]
    write_json(out / 'QUEUE.json', {'repositories': list(map(str, queue)), 'scope': 'Review committed local heads; remote survey runs separately in Actions.'})
    active = None; log = None; batch = 0; last_hour = -1; hourly = []; evidence_bytes = 0; paused_hot = False; bench_pending = []; reviews = []
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while time.monotonic() < deadline and not (out / 'STOP').exists():
            s = sample(); elapsed = time.monotonic() - started; hour = int(elapsed // 3600)
            if hour != last_hour:
                if hourly: write_json(out / f'hour-{last_hour:02d}.json', {'samples': hourly, 'completed': record['completed'], 'failed': record['failed']})
                hourly = []; last_hour = hour
                bench_pending = ['gpu', 'cpu']
            hourly.append(s)
            if len(hourly) > 900: hourly = hourly[-900:]
            reason = guard(s)
            if s.get('gpuTemperature', 100) >= 78: paused_hot = True
            if paused_hot and s.get('gpuTemperature', 100) <= 70: paused_hot = False
            if paused_hot: reason = 'Cooling to 70 C'
            if shutil.disk_usage(out).free < 5 * GIB: reason = 'Less than 5 GiB disk space'
            if active and (reason or time.monotonic() - task_started > 180):
                stop_child(active); active.wait(timeout=10); log.close()
                record['interruptedJobs'].append({'mode': mode, 'repository': owner.name if mode == 'review' else None, 'reason': reason or '180 second timeout', 'log': str(log_path)})
                record['interrupted'] += 1; active = None
            if active and active.poll() is not None:
                code = active.returncode; log.close()
                evidence_bytes += log_path.stat().st_size
                text = log_path.read_text(encoding='utf8', errors='replace')
                result = None
                try:
                    if mode == 'review':
                        report_path = review_out / 'cartridge.json'
                        result = {'cartridge': str(report_path), 'bytes': report_path.stat().st_size}
                        evidence_bytes += result['bytes']
                        reviewed = json.loads(report_path.read_text(encoding='utf8'))
                        reviews.append({'repository': reviewed['repository'], 'commit': reviewed['commit'], 'parseFailures': len(reviewed['parseFailures']), 'inspected': reviewed['inspected'], 'truncated': reviewed['truncated']})
                        write_json(out / 'REVIEW-INDEX.json', reviews)
                        (out / 'REVIEW-INDEX.md').write_text('# Overnight source review cards\n\n' + '\n'.join(f'- [{r["repository"]}](reviews/{r["repository"]}/REVIEW.md): {r["parseFailures"]} parse failures; {r["inspected"]} unique blobs checked; truncated={r["truncated"]}' for r in reviews) + '\n', encoding='utf8')
                        correct = reviewed['repository'] == owner.name and bool(reviewed['commit']) and isinstance(reviewed['inspected'], int)
                    else:
                        result = json.loads(next(line[5:] for line in reversed(text.splitlines()) if line.startswith('JSON ')))
                        correct = all(r['correct'] for r in result['rows']) if mode == 'gpu' else all(r['sectionsAgree'] and r['digestsAgree'] for r in result['ladder'])
                except Exception: correct = False
                success = code == 0 and correct
                record['completed' if success else 'failed'] += 1
                write_json(out / f'batch-{batch:05d}.json', {'mode': mode, 'exitCode': code, 'receiptValidated' if mode == 'review' else 'correct': correct, 'result': result, 'finishedAt': now()})
                active = None
                if not success:
                    record['stopReason'] = 'Benchmark failure; stopped for review'; break
            evidence_bytes = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            if evidence_bytes > 450_000_000:
                record['stopReason'] = 'Evidence storage budget reached'; break
            if not active and not reason and s['availableGiB'] >= 2.6 and (bench_pending or queue):
                batch += 1; mode = bench_pending.pop(0) if bench_pending else 'review'
                workers = max(1, min(len(affinity), int((s['availableGiB'] - 2.3) / .2)))
                if mode == 'review':
                    owner = queue.pop(0); review_out = out / 'reviews' / owner.name
                    command = [os.sys.executable, str(HERE / 'codex/review-repo.py'), '--root', str(owner), '--out', str(review_out), '--workers', str(workers)]
                else:
                    command = ['node', str(HERE / 'claude' / ('bench-gpu.mjs' if mode == 'gpu' else 'bench-cpu-ram.mjs')), str(args.artifact)]
                    command += ['--iters', '100', '--headed=0'] if mode == 'gpu' else ['--threads', ','.join(map(str, sorted({1, max(1, workers // 2), workers})))]
                log_path = out / f'batch-{batch:05d}-{mode}.log'; log = log_path.open('w', encoding='utf8')
                active = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                try:
                    process = psutil.Process(active.pid); process.cpu_affinity(affinity); process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                except psutil.Error: pass
                task_started = time.monotonic()
            write_json(out / 'STATUS.json', {'at': now(), 'elapsedSeconds': round(elapsed), 'active': mode if active else None,
                                            'pauseReason': reason or (('Queue complete; waiting for hourly measurement' if not queue and not bench_pending else 'Waiting for 2.6 GiB available RAM') if not active else None),
                                            'remainingReviewJobs': len(queue),
                                            'completed': record['completed'], 'failed': record['failed'], 'sample': s})
            # Check output growth during the task, not only after it exits.
            if active and log_path.stat().st_size > 8_000_000:
                record['stopReason'] = 'Task log budget reached'; break
            time.sleep(5)
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        if active:
            stop_child(active)
            record['interruptedJobs'].append({'mode': mode, 'repository': owner.name if mode == 'review' else None, 'reason': 'Controller stopped', 'log': str(log_path)})
        if log and not log.closed: log.close()
        if hourly: write_json(out / f'hour-{last_hour:02d}.json', {'samples': hourly, 'completed': record['completed'], 'failed': record['failed']})
        record['finishedAt'] = now(); record.setdefault('stopReason', 'STOP requested' if (out / 'STOP').exists() else 'Duration complete')
        record['remainingReviews'] = list(map(str, queue))
        record['reviewCoverageComplete'] = not queue and not any(j['mode'] == 'review' for j in record['interruptedJobs']) and record['failed'] == 0
        write_json(out / 'RUN.json', record)


if __name__ == '__main__':
    main()
