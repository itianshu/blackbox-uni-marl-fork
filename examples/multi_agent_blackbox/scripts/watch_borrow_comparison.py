"""Check a running comparison every ten minutes without changing its workload."""
import argparse
import bisect
import datetime
import json
import os
from pathlib import Path
import statistics
import time


def rows(path):
    if not path.exists():
        return []
    result = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            result.append(json.loads(line))
        except ValueError:
            pass
    return result


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, TypeError):
        return False


def inspect(directory, interval):
    manifest = json.loads((directory / 'manifest.json').read_text())
    mode = manifest.get('active_mode', manifest.get('mode_order', ['dynamic'])[-1])
    info = manifest.get('runs', {}).get(mode, {})
    run = directory / mode
    now = time.time()
    result = {'time_unix_s': now, 'time_local': datetime.datetime.now().isoformat(timespec='seconds'),
              'status': manifest['status'], 'mode': mode, 'runs': {}, 'alerts': []}
    for name in manifest.get('runs', {}):
        steps = rows(directory / name / 'steps.jsonl')
        result['runs'][name] = {'completed_steps': max([s['step'] for s in steps] or [0]),
            'recent_step_s': [s['metrics'].get('perf/time_per_step') for s in steps[-3:]]}
    if manifest['status'] == 'running':
        for name, pid in [('launcher', info.get('launcher_pid')),
                          ('step_observer', manifest.get('steps_observer_pid'))]:
            if not alive(pid):
                result['alerts'].append(f'{name} process is not alive: {pid}')
        for pid in info.get('monitor_pids', []):
            if not alive(pid):
                result['alerts'].append(f'monitor process is not alive: {pid}')
    trace = rows(run / 'checkpoints/dynamic_inference.jsonl')
    trace = [r for r in trace if 'active_lends' in r]
    times = [r['time_unix_s'] for r in trace]
    result['recent_scheduling_events'] = [{'kind': r['kind'], 'detail': r.get('detail', {}),
        'time_unix_s': r['time_unix_s']} for r in trace
        if r['time_unix_s'] >= now - interval and r['kind'] != 'sample']
    pids = info.get('monitor_pids', [])
    serving_path = Path(f'/tmp/serving_monitor_{pids[1]}.jsonl') if len(pids) > 1 else run / 'serving.jsonl'
    if not serving_path.exists():
        serving_path = run / 'serving.jsonl'
    samples = rows(serving_path)
    result['serving_sample_age_s'] = now - samples[-1]['time_unix_s'] if samples else None
    window = [r for r in samples if r['time_unix_s'] >= now - interval]
    policies = {f'policy_{i}': {'kv': [], 'waiting': [], 'running': []} for i in range(1, 4)}
    failures = 0
    for sample in window:
        index = bisect.bisect_right(times, sample['time_unix_s']) - 1
        lends = trace[index]['active_lends'] if index >= 0 else []
        sleeping = {a for lend in lends for a in lend['home_servers']}
        guests = {a for lend in lends for a in lend['guest_servers']}
        for server in sample['servers']:
            if server['address'] in sleeping or (server['rank'] >= 10000 and server['address'] not in guests):
                continue
            if 'metrics' not in server:
                failures += 1
                continue
            metric = server['metrics']
            p = policies[server['policy']]
            p['kv'].append(metric.get('gpu_cache_usage_max', 0))
            p['running'].append(metric.get('requests_running', 0))
            p['waiting'].append(metric.get('requests_waiting', 0))
    result['last_interval_policy_samples'] = {p: {
        'valid_replica_samples': len(v['kv']),
        'kv_mean': statistics.mean(v['kv']) if v['kv'] else None,
        'kv_peak': max(v['kv'], default=None),
        'kv_ge_50_sample_fraction': statistics.mean([x >= .5 for x in v['kv']]) if v['kv'] else None,
        'running_per_replica_max': max(v['running'], default=None),
        'waiting_per_replica_max': max(v['waiting'], default=None)} for p, v in policies.items()}
    result['failed_active_endpoint_scrapes'] = failures
    if manifest['status'] == 'running' and samples and now - samples[-1]['time_unix_s'] > 120:
        result['alerts'].append('Serving samples are over 120 seconds old')
    if manifest['status'] == 'failed':
        result['alerts'].append('Experiment failed; inspect train.log before restarting')
    report = directory / 'comparison_report.md'
    result['report_exists'] = report.exists()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--interval', type=float, default=600)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    spool = Path(f'/tmp/borrow_watch_{os.getpid()}.jsonl')
    while True:
        started = time.monotonic()
        try:
            result = inspect(args.directory, args.interval)
            line = json.dumps(result, ensure_ascii=False) + '\n'
            with spool.open('a') as stream:
                stream.write(line)
            temporary = args.directory / 'watch_latest.tmp'
            temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2))
            temporary.replace(args.directory / 'watch_latest.json')
            history = args.directory / 'watch_history.tmp'
            history.write_bytes(spool.read_bytes())
            history.replace(args.directory / 'watch_history.jsonl')
            print(result['time_local'], result['status'], result['mode'], result['runs'], result['alerts'], flush=True)
            if args.once or result['status'] == 'failed' or (result['status'] == 'completed' and result['report_exists']):
                break
        except Exception as exc:
            print(f'Inspection failed: {exc!r}', flush=True)
            if args.once:
                raise
        time.sleep(max(0, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
