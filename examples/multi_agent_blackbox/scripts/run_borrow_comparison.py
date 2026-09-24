"""Run matched dynamic/static experiments sequentially and retain their artifacts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def recover_monitor_spools(run_dir, monitors):
    """Use complete local samples after shared-filesystem append failures."""
    for monitor, name, prefix in zip(monitors, ['gpu.jsonl', 'serving.jsonl'],
                                     ['borrow_gpu_monitor', 'serving_monitor']):
        source = Path(f'/tmp/{prefix}_{monitor.pid}.jsonl')
        if source.exists():
            destination = run_dir / name
            temporary = destination.with_suffix('.recovered.tmp')
            temporary.write_bytes(source.read_bytes())
            temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--profile', choices=['standard', 'high_kv', 'random_routing', 'two_policy_2b'], default='standard')
    parser.add_argument('--modes', nargs='+', choices=['dynamic', 'static'], default=['dynamic', 'static'])
    parser.add_argument('--baseline', type=Path, help='Historical failed run for a dynamic-only regression report')
    parser.add_argument('--ray-address', default=os.environ.get('RAY_ADDRESS'),
                        help='Ray GCS address, or set RAY_ADDRESS')
    parser.add_argument('--dashboard-address', default=os.environ.get('RAY_DASHBOARD_ADDRESS'),
                        help='Ray dashboard URL, or set RAY_DASHBOARD_ADDRESS')
    args = parser.parse_args()
    if args.baseline and args.modes != ['dynamic']:
        parser.error('--baseline requires --modes dynamic')
    if not args.ray_address or not args.dashboard_address:
        parser.error(
            '--ray-address and --dashboard-address are required (or set '
            'RAY_ADDRESS and RAY_DASHBOARD_ADDRESS)')
    root = Path(__file__).resolve().parents[3]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'steps': args.steps, 'root': str(root), 'runs': {}, 'status': 'running', 'profile': args.profile, 'mode_order': args.modes}
    manifest_path = output / 'manifest.json'

    def save():
        temporary = manifest_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)

    save()
    if args.baseline:
        manifest['baseline'] = str(args.baseline.resolve())
        save()

    def regression_report():
        if args.baseline:
            subprocess.run([sys.executable,
                str(root / 'examples/multi_agent_blackbox/scripts/analyze_stress_regression.py'),
                str(output), '--baseline', str(args.baseline.resolve())], cwd=root, check=True)
    with (output / 'steps_observer.log').open('w') as stream:
        observer = subprocess.Popen([sys.executable,
            str(root / 'examples/multi_agent_blackbox/scripts/observe_comparison_steps.py'), str(output),
            '--dashboard', args.dashboard_address],
            cwd=root, stdout=stream, stderr=subprocess.STDOUT)
    manifest['steps_observer_pid'] = observer.pid
    save()
    subprocess.run(['git', 'diff', '--', '.', ':!verl'], cwd=root,
                   stdout=(output / 'source_changes.patch').open('w'), check=True)
    snapshot = output / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for relative in [
        'examples/multi_agent_blackbox/config/multi_agent_blackbox_2policy_2b_comparison.yaml',
        'examples/multi_agent_blackbox/config/mas_config_2agent_variable.yaml',
        'examples/multi_agent_blackbox/scripts/run_e2e_2policy_2b.sh',
        'examples/multi_agent_blackbox/scripts/run_e2e_train.sh',
        'examples/multi_agent_blackbox/scripts/observe_comparison_steps.py',
        'examples/multi_agent_blackbox/scripts/analyze_2policy_comparison.py',
    ]:
        source = root / relative
        if source.exists():
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    for mode in args.modes:
        run_dir = output / mode
        run_dir.mkdir(exist_ok=True)
        env = os.environ.copy()
        env.update(BORROW_LOAD_PROFILE=args.profile, PYTHON=sys.executable, PYTHONPATH=str(root),
                   RAY_ADDRESS=args.ray_address,
                   RAY_DASHBOARD_ADDRESS=args.dashboard_address,
                   TOTAL_TRAINING_STEPS=str(args.steps),
                   DYNAMIC_INFERENCE_SCHEDULING=str(mode == 'dynamic').lower(),
                   LOG_PATH=str(run_dir / 'train.log'), CKPT_DIR=str(run_dir / 'checkpoints'))
        launcher = 'run_e2e_2policy_2b.sh' if args.profile == 'two_policy_2b' else 'run_e2e_borrow_verify.sh'
        if args.profile == 'two_policy_2b':
            # Standalone TP1 rollout actors are per-policy replica ranks;
            # training workers are WorkerDict actors rather than vLLM servers.
            env['MONITOR_POLICY_TRAIN_RANKS'] = json.dumps({'policy_1': 0, 'policy_2': 0})
        with (run_dir / 'launcher.log').open('w') as stream:
            process = subprocess.Popen(['bash', str(root / 'examples/multi_agent_blackbox/scripts' / launcher)],
                env=env, cwd=root, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        info = {'launcher_pid': process.pid, 'started_unix_s': time.time(),
                'log': env['LOG_PATH'], 'trace': str(run_dir / 'checkpoints/dynamic_inference.jsonl'),
                'gpu_log': str(run_dir / 'gpu.jsonl'), 'serving_log': str(run_dir / 'serving.jsonl')}
        manifest['runs'][mode] = info
        manifest['active_mode'] = mode
        save()
        monitors = []
        for script, file, extra in [
            ('monitor_borrow_gpus.py', 'gpu.jsonl', []),
            ('monitor_borrow_serving.py', 'serving.jsonl', ['--dashboard', args.dashboard_address]),
        ]:
            with (run_dir / (script + '.log')).open('w') as stream:
                monitors.append(subprocess.Popen([sys.executable,
                    str(root / 'examples/multi_agent_blackbox/scripts' / script),
                    '--address', args.ray_address, '--launcher-pid', str(process.pid),
                    '--output', str(run_dir / file), *extra], env=env, cwd=root,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        info['monitor_pids'] = [p.pid for p in monitors]
        save()
        info['exit_code'] = process.wait()
        info['finished_unix_s'] = time.time()
        for monitor in monitors:
            monitor.terminate()
        for monitor in monitors:
            monitor.wait(timeout=30)
        recover_monitor_spools(run_dir, monitors)
        save()
        if info['exit_code'] != 0:
            manifest['status'] = 'failed'
            save()
            observer.wait(timeout=30)
            regression_report()
            return
        # Driver-owned actors exit with the training driver. Detached NCCL ID
        # stores hold no GPUs; leave unrelated actors alone and allow cleanup.
        time.sleep(15)
    manifest['status'] = 'completed'
    manifest.pop('active_mode', None)
    save()
    observer.wait(timeout=30)
    if args.baseline:
        regression_report()
        return
    analyzer = 'analyze_2policy_comparison.py' if args.profile == 'two_policy_2b' else 'analyze_borrow_comparison.py'
    subprocess.run([sys.executable,
        str(root / 'examples/multi_agent_blackbox/scripts' / analyzer), str(output)],
        cwd=root, check=True)


if __name__ == '__main__':
    main()
