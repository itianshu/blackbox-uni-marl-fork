"""Identical passive vLLM observation for dynamic and static comparison runs."""
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ray
import requests

from uni_agent.trainer.dynamic_inference.metrics import VLLMMetricsScraper, summarize_vllm_metrics, _HISTOGRAMS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', required=True)
    parser.add_argument('--dashboard', required=True)
    parser.add_argument('--launcher-pid', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ray.init(address=args.address, logging_level='ERROR')
    scraper = VLLMMetricsScraper(['kv_cache_usage_perc', 'gpu_cache_usage_perc', 'kv_cache_usage_ratio'])
    servers, pending = {}, {}
    last_discovery = 0
    session = requests.Session()
    session.trust_env = False
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=16) as pool:
        while True:
            try:
                os.kill(args.launcher_pid, 0)
            except ProcessLookupError:
                break
            if time.monotonic() - last_discovery > 20:
                try:
                    actors = session.get(args.dashboard + '/api/v0/actors',
                        params={'limit': 1000, 'detail': 'true', 'filter_keys': 'state', 'filter_predicates': '=', 'filter_values': 'ALIVE'}, timeout=10).json()['data']['result']['result']
                    jobs = {a['job_id'] for a in actors if a['state'] == 'ALIVE'
                            and a['class_name'] == 'MultiAgentsTaskRunner'}
                    for actor in actors:
                        match = re.fullmatch(r'(policy_[123])_vllm_server_(\d+)_0', actor['name'])
                        if actor['state'] != 'ALIVE' or actor['job_id'] not in jobs or not match:
                            continue
                        policy, rank = match[1], int(match[2])
                        if rank < {'policy_1': 2, 'policy_2': 4, 'policy_3': 2}[policy]:
                            continue  # hybrid engines do not serve this separate_async workload
                        if actor['actor_id'] not in servers and actor['actor_id'] not in pending:
                            handle = ray.get_actor(actor['name'], namespace=actor['ray_namespace'])
                            pending[actor['actor_id']] = (handle.get_server_address.remote(), {
                                'policy': policy, 'rank': rank, 'actor_id': actor['actor_id'],
                                'node_id': actor['node_id'], 'pid': actor['pid'], 'job_id': actor['job_id']})
                    last_discovery = time.monotonic()
                except Exception as exc:
                    print('discovery:', repr(exc), flush=True)
            for actor_id, (ref, metadata) in list(pending.items()):
                ready, _ = ray.wait([ref], timeout=0)
                if ready:
                    try:
                        address = ray.get(ref)
                        if isinstance(address, (tuple, list)):
                            address = ':'.join(map(str, address))
                        servers[actor_id] = {**metadata, 'address': address}
                    except Exception as exc:
                        print('address:', repr(exc), flush=True)
                    del pending[actor_id]
            entries = list(servers.values())
            snapshots = list(pool.map(lambda x: scraper.scrape(x['address']), entries))
            rows = []
            for metadata, snapshot in zip(entries, snapshots):
                if snapshot is None:
                    rows.append({**metadata, 'scrape_failed': True})
                    continue
                histograms = {}
                for key in ['request_queue_time_seconds', 'time_to_first_token_seconds',
                            'time_per_output_token_seconds', 'e2e_request_latency_seconds']:
                    for name in _HISTOGRAMS[key]:
                        count = snapshot.aggregate(name + '_count')
                        if count is not None:
                            histograms[key] = {'count': count, 'sum': snapshot.aggregate(name + '_sum'),
                                'buckets': {str(bound): value for bound, value in snapshot._histogram_buckets(name)}}
                            break
                rows.append({**metadata, 'histograms': histograms,
                    'metrics': summarize_vllm_metrics(snapshot, lambda *args: None, lambda *args: None)})
            line = json.dumps({'time_unix_s': time.time(), 'servers': rows}) + '\n'
            # Keep a local copy even if the shared filesystem temporarily fails.
            with Path(f'/tmp/serving_monitor_{os.getpid()}.jsonl').open('a') as stream:
                stream.write(line)
            try:
                with output.open('a') as stream:
                    stream.write(line)
            except OSError as exc:
                print('shared output write failed; local copy retained:', repr(exc), flush=True)
            time.sleep(5)
    ray.shutdown()


if __name__ == '__main__':
    main()
