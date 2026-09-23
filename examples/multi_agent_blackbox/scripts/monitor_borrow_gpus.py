"""Sample physical GPU load on every Ray node while the experiment is alive."""
import argparse
import json
import os
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=0)
def sample_gpus():
    import csv
    import subprocess
    import psutil
    fields = ['index', 'uuid', 'utilization.gpu', 'utilization.memory',
              'memory.used', 'memory.total', 'power.draw']
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=' + ','.join(fields), '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
         '--format=csv,noheader,nounits'], text=True, timeout=10)
    memory = psutil.virtual_memory()
    network = psutil.net_io_counters()
    return {'time_unix_s': time.time(), 'cpu_utilization_percent': psutil.cpu_percent(interval=.1),
            'host_memory_used_bytes': memory.used, 'host_memory_total_bytes': memory.total,
            'network_bytes_sent': network.bytes_sent, 'network_bytes_received': network.bytes_recv,
            'gpus': [dict(zip(fields, [v.strip() for v in row])) for row in csv.reader(output.splitlines())],
            'processes': list(csv.reader(processes.splitlines()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--launcher-pid', type=int, required=True)
    args = parser.parse_args()
    ray.init(address=args.address, logging_level='ERROR')
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    spool = Path(f'/tmp/borrow_gpu_monitor_{os.getpid()}.jsonl')
    print(f'Local durable spool: {spool}', flush=True)
    while True:
        try:
            os.kill(args.launcher_pid, 0)
        except ProcessLookupError:
            break
        nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0)]
        refs = [sample_gpus.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            n['NodeID'], soft=False)).remote() for n in nodes]
        for node, ref in zip(nodes, refs):
            try:
                result = ray.get(ref, timeout=20)
            except Exception as exc:
                result = {'time_unix_s': time.time(), 'error': str(exc)}
            line = json.dumps({'node_ip': node['NodeManagerAddress'], **result}) + '\n'
            with spool.open('a') as stream:
                stream.write(line)
            for attempt in range(5):
                try:
                    with path.open('a') as stream:
                        stream.write(line)
                    break
                except OSError as error:
                    print(f'Shared log write failed ({attempt + 1}/5): {error}; retained in {spool}', flush=True)
                    time.sleep(.2)

        time.sleep(15)
    ray.shutdown()


if __name__ == '__main__':
    main()
