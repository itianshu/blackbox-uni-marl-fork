"""Compare a live/completed stress rerun with the historical failed dynamic run."""
import argparse
import collections
import datetime
import json
import math
import os
from pathlib import Path
import re
import statistics
import time


def rows(path):
    if path.exists():
        with path.open(errors='replace') as stream:
            for line in stream:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # shared-file appends may expose an incomplete last row


def stats(values):
    values = sorted(v for v in values if isinstance(v, (float, int)) and math.isfinite(v))
    return ({'n': len(values), 'mean': statistics.mean(values),
             'p90': values[math.ceil(.9 * len(values)) - 1], 'max': values[-1]} if values else {})


def summarize(directory):
    run = directory / 'dynamic'
    manifest = json.loads((directory / 'manifest.json').read_text())
    steps = {r['step']: r['metrics'] for r in rows(run / 'steps.jsonl')}
    # Fallback when the raw-worker observer has not flushed a completed step.
    log = (run / 'train.log').read_text(errors='replace').replace('\x00', '') if (run / 'train.log').exists() else ''
    log = re.sub(r'\x1b\[[0-9;]*m', '', log)
    for line in log.splitlines():
        match = re.search(r'training/global_step:(\d+)', line)
        if not match:
            continue
        metrics = {}
        for key, value in re.findall(r'(?:^| - )([\w/.-]+):((?:np\.float64\()?[-+\d.eE]+\)?)', line):
            try:
                metrics[key] = float(value.removeprefix('np.float64(').removesuffix(')'))
            except ValueError:
                pass
        steps.setdefault(int(match[1]), metrics)
    counts = collections.Counter()
    events = []
    samples = []
    for r in rows(run / 'checkpoints/dynamic_inference.jsonl'):
        counts[r['kind']] += 1
        if r['kind'] != 'sample':
            events.append({k: r.get(k) for k in ('time_unix_s', 'kind', 'step', 'detail')})
        else:
            # Store only the comparable decision signal, not the large raw series.
            samples.append((r['time_unix_s'], {
                p: v.get('kv_decision') for p, v in r.get('policies', {}).items()}))
    active_start = min((e['time_unix_s'] for e in events if e['kind'] == 'borrow'), default=None)
    active_kv = {p: stats([kv.get(p) for ts, kv in samples if active_start is not None and ts >= active_start])
                 for p in ('policy_1', 'policy_2', 'policy_3')}
    syncs = []
    for step, targets, elapsed in re.findall(
            r'CHECKPOINT_SYNC_COMPLETE step=(\d+) targets=(\[[^\n]*?\]) elapsed_s=([\d.]+)', log):
        syncs.append({'step': int(step), 'targets': targets, 'elapsed_s': float(elapsed)})
    failures = [e for e in events if e['kind'].endswith('_failed')]
    info = manifest.get('runs', {}).get('dynamic', {})
    return {'status': manifest['status'], 'exit_code': info.get('exit_code'),
            'steps': steps, 'last_completed_step': max(steps, default=0), 'events': dict(counts),
            'failures': failures, 'kv_since_first_borrow': active_kv,
            'borrow_execution_s': stats([e['detail'].get('execution_s') for e in events if e['kind'] == 'borrow_complete']),
            'return_execution_s': stats([e['detail'].get('execution_s') for e in events if e['kind'] == 'return_complete']),
            'held_s': stats([e['detail'].get('held_s') for e in events if e['kind'] in ('return', 'early_return')]),
            'syncs': syncs, 'sync_failed_markers': log.count('CHECKPOINT_SYNC_FAILED'),
            'nccl_watchdog_seen': 'Watchdog caught collective operation timeout' in log,
            'transaction_events': events}


def render(current, baseline):
    old, new = summarize(baseline), summarize(current)
    summary = {'updated_at': datetime.datetime.now().isoformat(), 'baseline': str(baseline),
               'current': str(current), 'old': old, 'new': new}
    def val(v):
        return '—' if v is None else f'{v:.3f}' if isinstance(v, float) else str(v)
    def count(run, *kinds):
        return sum(run['events'].get(k, 0) for k in kinds)
    final = new['status'] != 'running'
    passed = (new['exit_code'] == 0 and new['last_completed_step'] >= 10
              and count(new, 'borrow') > 0 and count(new, 'return', 'early_return') > 0
              and not new['failures'] and not new['sync_failed_markers'] and not new['nccl_watchdog_seen'])
    outcome = ('本轮同负载 10 step 回归通过，未复现原故障。' if passed else
               '实验已结束，未满足回归通过条件；需结合失败事件和日志判断。' if final else
               '实验仍在运行，以下为阶段数据，尚不能判断原故障是否消失。')
    lines = ['# 高负载借还回归对比', '', outcome, '',
             f'更新时间：{summary["updated_at"]}', '',
             '旧组：2026-09-21 kv_threshold_stress_retry；新组：最新事务化借还代码。',
             '两组均启用真实 KV 自动调度，使用完整互借图；没有使用确定性冒烟的强制借还入口。', '',
             '| 项目 | 旧代码 | 新代码 |', '|---|---:|---:|']
    for label, a, b in [
        ('完成 step', old['last_completed_step'], new['last_completed_step']),
        ('运行状态', old['status'], new['status']), ('退出码', old['exit_code'], new['exit_code']),
        ('成功借入', count(old, 'borrow'), count(new, 'borrow')),
        ('成功归还（含保护性归还）', count(old, 'return', 'early_return'), count(new, 'return', 'early_return')),
        ('借还失败事件数', len(old['failures']), len(new['failures'])),
        ('NCCL watchdog 超时', old['nccl_watchdog_seen'], new['nccl_watchdog_seen']),
        ('单次借入耗时均值（秒）', old['borrow_execution_s'].get('mean'), new['borrow_execution_s'].get('mean')),
        ('单次归还耗时均值（秒）', old['return_execution_s'].get('mean'), new['return_execution_s'].get('mean')),
    ]:
        lines.append(f'| {label} | {val(a)} | {val(b)} |')
    lines += ['', '## 相同 step 的计时', '', '| Step | 旧组总 step 秒 | 新组总 step 秒 |', '|---|---:|---:|']
    for step in sorted(set(old['steps']) | set(new['steps'])):
        def duration(run):
            m = run['steps'].get(step, {})
            return m.get('perf/time_per_step', m.get('timing_s/step'))
        lines.append(f'| {step} | {val(duration(old))} | {val(duration(new))} |')
    lines += ['', '## 压力信号', '',
              '从各组首次成功借入起统计 policy KV 决策信号（policy 平均 KV 的时间窗 P90）。',
              '两组运行时长不同，下面用于确认压力覆盖，不能当作相同时间窗的性能收益。', '',
              '| Policy | 旧组均值 / P90 / 最大值 | 新组均值 / P90 / 最大值 |', '|---|---:|---:|']
    for p in ('policy_1', 'policy_2', 'policy_3'):
        def percentages(run):
            x = run['kv_since_first_borrow'][p]
            return ' / '.join(f'{100*x[k]:.1f}%' if k in x else '—' for k in ('mean', 'p90', 'max'))
        lines.append(f'| {p} | {percentages(old)} | {percentages(new)} |')
    lines += ['', '## 新组失败事件', '',
              '```json', json.dumps(new['failures'], ensure_ascii=False, indent=2), '```', '',
              '## 解释边界', '',
              '- 相同负载：batch 256、n=4、随机 7–10 轮、每轮预算 8192–10240 token；训练卡 4/4/4、推理卡 4/12/4、TP2。',
              '- KV 触发阈值仍为 55%，退出 30%，借出后上限 40%；队列触发关闭。',
              '- 借还操作超时从 10 秒增加为 60 秒，整次同步超时仍为 300 秒；因此本对比不能单独分离延长超时与状态机修改的贡献。',
              '- 旧组归还失败后出现 NCCL 超时是已观测的时间顺序，不能仅凭它证明二者存在唯一因果关系。',
              '- 普通同步耗时不包含旧版独立 renew 的全部开销；需要结合总 step 时间看，不能直接据其声称提速。',
              '- 固定随机种子无法保证异步请求顺序完全相同；未运行独立准确率评测，本报告不作精度提升结论。',
              '- 新组完成 10 step 且覆盖真实借还才算本轮通过；这仍不代表任意长时间和全部外部故障均已覆盖。', '',
              f'旧组证据：`{baseline}`', f'新组证据：`{current}`', '']
    for name, content in [('stress_comparison.json', json.dumps(summary, ensure_ascii=False, indent=2)+'\n'),
                          ('stress_comparison.md', '\n'.join(lines))]:
        path = current / name
        temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        temp.write_text(content)
        temp.replace(path)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--interval', type=float, default=600)
    args = parser.parse_args()
    while True:
        done = render(args.directory.resolve(), args.baseline.resolve())
        if done or not args.follow:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
