"""Timestamp metric lines for aligning step windows with independent monitors."""
import argparse
import json
from pathlib import Path
import re
import time
import requests


def raw_worker_log(dashboard, actor):
    params = {"node_id": actor["node_id"], "glob": f"*{actor['pid']}*"}
    files = requests.get(dashboard + "/api/v0/logs", params=params, timeout=10).json()["data"]["result"]
    filename = files.get("worker_out", [None])[0]
    if not filename:
        return None
    response = requests.get(dashboard + "/api/v0/logs/file", params={
        "node_id": actor["node_id"], "filename": filename, "lines": 100000}, timeout=15)
    response.raise_for_status()
    return response.text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--dashboard', required=True, help='Ray dashboard URL')
    args = parser.parse_args()
    dashboard = args.dashboard.rstrip('/')
    actors = {}
    positions = {}
    seen = {'dynamic': set(), 'static': set()}
    for mode in seen:
        path = args.directory / mode / 'steps.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    seen[mode].add(json.loads(line)['step'])
                except ValueError:
                    pass
    while True:
        # Shared filesystems can briefly hide a just-renamed manifest from a
        # newly spawned process.  Treat that as startup/metadata propagation,
        # not as a terminal observer failure.
        try:
            manifest = json.loads((args.directory / 'manifest.json').read_text())
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f'manifest read retry: {error}', flush=True)
            time.sleep(1)
            continue
        active_mode = manifest.get('active_mode')
        if active_mode:
            try:
                if active_mode not in actors:
                    entries = requests.get(dashboard + '/api/v0/actors', params={
                        'limit': 1000, 'detail': 'true', 'filter_keys': 'state', 'filter_predicates': '=', 'filter_values': 'ALIVE'}, timeout=10).json()['data']['result']['result']
                    roots = [a for a in entries if a['class_name'] == 'MultiAgentsTaskRunner' and a['state'] == 'ALIVE']
                    if len(roots) == 1:
                        actors[active_mode] = roots[0]
                if active_mode in actors:
                    content = raw_worker_log(dashboard, actors[active_mode])
                    if content:
                        raw_path = args.directory / active_mode / 'train_worker.log'
                        temporary = raw_path.with_suffix('.tmp')
                        temporary.write_text(content)
                        temporary.replace(raw_path)
            except Exception as error:
                print(f'raw log fetch: {error}', flush=True)
        for mode in seen:
            source = args.directory / mode / 'train.log'
            raw = source.parent / 'train_worker.log'
            if raw.exists():
                source = raw
            if not source.exists():
                continue
            with source.open(errors='replace') as stream:
                stream.seek(0)  # Re-scan: Ray driver forwarding can omit entire metric lines.
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith('\n'):
                        positions[mode] = position
                        break
                    if 'perf/time_per_step:' not in line:
                        continue
                    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    match = re.search(r'\bstep:(\d+)', line)
                    if not match or int(match[1]) in seen[mode]:
                        continue
                    metrics = {}
                    for part in line.split(' - ')[1:]:
                        key, separator, value = part.partition(':')
                        if separator:
                            value = value.strip()
                            wrapped = re.fullmatch(r'(?:np|numpy)\.(?:float|int)\d+\(([^()]*)\)', value)
                            if wrapped:
                                value = wrapped[1]
                            try:
                                metrics[key] = float(value)
                            except ValueError:
                                metrics[key] = value.strip()
                    destination = source.parent / 'steps.jsonl'
                    rows = [json.loads(line) for line in destination.read_text().splitlines()] if destination.exists() else []
                    observed = time.time()
                    timestamp_source = 'raw_worker_log_poll' if source == raw else 'driver_log_poll'
                    following = sorted((r for r in rows if r['step'] > int(match[1])), key=lambda r: r['step'])
                    if following and following[0]['step'] == int(match[1]) + 1:
                        observed = following[0]['observed_unix_s'] - following[0]['metrics']['perf/time_per_step']
                        timestamp_source = 'reconstructed_from_next_step_duration'
                    rows.append({'observed_unix_s': observed, 'timestamp_source': timestamp_source,
                                 'step': int(match[1]), 'metrics': metrics})
                    rows.sort(key=lambda row: row['step'])
                    temporary = destination.with_suffix('.tmp')
                    temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                    temporary.replace(destination)
                    seen[mode].add(int(match[1]))
        try:
            status = json.loads((args.directory / 'manifest.json').read_text())['status']
        except (OSError, ValueError):
            status = 'running'
        if status != 'running':
            break
        time.sleep(5)


if __name__ == '__main__':
    main()
