"""Produce a measured comparison; keep missing observations and proxies explicit."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys


def read_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass  # an actively appended final line can be incomplete
    return rows


def mean(values):
    values = [v for v in values if isinstance(v, (float, int)) and math.isfinite(v)]
    return statistics.mean(values) if values else None


def describe(values):
    values = sorted(v for v in values if isinstance(v, (float, int)) and math.isfinite(v))
    if not values:
        return {}
    return {'n': len(values), 'mean': mean(values), 'median': statistics.median(values),
            'min': values[0], 'max': values[-1], 'p90': values[math.ceil(.9 * len(values)) - 1],
            'std': statistics.stdev(values) if len(values) > 1 else 0}


def histogram_summary(histogram):
    count = histogram['count']
    if not count:
        return {'count': 0}
    result = {'count': count, 'mean': histogram['sum'] / count}
    for quantile in [.5, .9, .99]:
        target = count * quantile
        lower, previous = 0, 0
        for upper, cumulative in sorted((float(k), v) for k, v in histogram['buckets'].items()):
            if cumulative >= target:
                result[f'p{int(quantile * 100)}'] = lower if math.isinf(upper) else (
                    lower + (upper - lower) * (target - previous) / max(cumulative - previous, 1))
                break
            lower, previous = upper, cumulative
    return result


def summarize_run(directory, first_step=3, last_step=20):
    steps = read_jsonl(directory / 'steps.jsonl')
    warmup = [s for s in steps if s['step'] <= 2]
    steady = [s for s in steps if first_step <= s['step'] <= last_step]
    summary = {'completed_steps': len(steps), 'warmup_step_s': describe([
        s['metrics'].get('perf/time_per_step') for s in warmup]),
        'steady_step_s': describe([s['metrics'].get('perf/time_per_step') for s in steady])}
    keys = sorted({key for row in steady for key in row['metrics']})
    summary['steady_metrics'] = {key: describe([row['metrics'].get(key) for row in steady])
                                 for key in keys}
    summary['steady_metrics'] = {key: value for key, value in summary['steady_metrics'].items() if value}
    # Bound monitoring by observed completion times (raw-log polling uncertainty ~5 s).
    begin = next((s['observed_unix_s'] for s in steps if s['step'] == first_step - 1), None)
    end = next((s['observed_unix_s'] for s in steps if s['step'] == last_step), None)
    summary['steady_monitor_window'] = [begin, end]
    summary['timestamp_sources'] = {str(s['step']): s.get('timestamp_source', 'driver_log_poll') for s in steps}
    summary['sensitivity_step_s'] = {f'{first}-{last_step}': describe([
        s['metrics'].get('perf/time_per_step') for s in steps if first <= s['step'] <= last_step])
        for first in [6, 11] if first <= last_step}
    summary['training_step_s_total'] = sum(s['metrics'].get('perf/time_per_step', 0) for s in steps)
    if begin is None or end is None:
        return summary
    gpu = read_jsonl(directory / 'gpu.jsonl')
    selected = [r for r in gpu if begin <= r['time_unix_s'] <= end and 'gpus' in r]
    for key in ['utilization.gpu', 'utilization.memory', 'memory.used', 'power.draw']:
        values = []
        for row in selected:
            for device in row['gpus']:
                try:
                    values.append(float(device[key]))
                except (ValueError, KeyError):
                    pass
        summary['gpu_' + key] = describe(values)
    summary['gpu_node_samples'] = len(selected)
    summary['host_cpu_percent'] = describe([r.get('cpu_utilization_percent') for r in selected])
    summary['host_memory_used_gib'] = describe([r['host_memory_used_bytes'] / 2**30
                                               for r in selected if 'host_memory_used_bytes' in r])
    # Time integration per node. Do not extrapolate across large monitoring gaps.
    energy_j = observed_gpu_seconds = 0
    nodes = {r['node_ip'] for r in gpu}
    for node in nodes:
        rows = sorted((r for r in gpu if r['node_ip'] == node and 'gpus' in r), key=lambda r: r['time_unix_s'])
        for a, b in zip(rows, rows[1:]):
            dt = min(end, b['time_unix_s']) - max(begin, a['time_unix_s'])
            if dt <= 0 or b['time_unix_s'] - a['time_unix_s'] > 45:
                continue
            try:
                watts = sum(float(g['power.draw']) for g in a['gpus'])
                energy_j += watts * dt
                observed_gpu_seconds += len(a['gpus']) * dt
            except (ValueError, KeyError):
                pass
    summary['observed_energy_kwh'] = energy_j / 3_600_000
    summary['energy_monitor_coverage'] = observed_gpu_seconds / (32 * (end - begin))
    summary['allocated_gpu_hours'] = 32 * (end - begin) / 3600
    serving = [r for r in read_jsonl(directory / 'serving.jsonl') if begin <= r['time_unix_s'] <= end]
    summary['serving_monitor_samples'] = len(serving)
    summary['serving_failed_scrapes'] = sum(bool(server.get('scrape_failed')) for row in serving for server in row['servers'])
    summary['serving_failed_sample_times'] = [row['time_unix_s'] for row in serving
        if any(server.get('scrape_failed') for server in row['servers'])]
    trace = [r for r in read_jsonl(directory / 'checkpoints/dynamic_inference.jsonl')
             if 'policies' in r]
    trace_times = [r['time_unix_s'] for r in trace]
    summaries = {}
    for policy in ['policy_1', 'policy_2', 'policy_3']:
        peak_kv, mean_kv, running, waiting = [], [], [], []
        busy_kv = []
        previous = {}
        previous_histograms, histograms = {}, {}
        generated = guest_generated = preemptions = successes = 0
        latest_histograms = {}
        for row in serving:
            servers = [s for s in row['servers'] if s['policy'] == policy and 'metrics' in s]
            if not servers:
                continue
            active = servers
            if trace:
                index = bisect.bisect_right(trace_times, row['time_unix_s']) - 1
                if index >= 0:
                    state = trace[index]
                    if 'active_lends' in state:
                        sleeping_home = {address for lend in state['active_lends'] if lend['lender'] == policy
                                         for address in lend['home_servers']}
                        active_guest = {address for lend in state['active_lends'] if lend['borrower'] == policy
                                        for address in lend['guest_servers']}
                        active = [s for s in servers if (s.get('rank', 0) < 10000 and s['address'] not in sleeping_home)
                                  or s['address'] in active_guest]
                    else:
                        addresses = state['policies'][policy]['replica_inflight']
                        active = [s for s in servers if s['address'] in addresses]
            if active:
                busy_kv.extend(s['metrics'].get('gpu_cache_usage_max', 0) for s in active
                               if s['metrics'].get('requests_running', 0) > 0)
                peak_kv.append(max(s['metrics'].get('gpu_cache_usage_max', 0) for s in active))
                mean_kv.append(mean([s['metrics'].get('gpu_cache_usage_max', 0) for s in active]))
                running.append(sum(s['metrics'].get('requests_running', 0) for s in active))
                waiting.append(sum(s['metrics'].get('requests_waiting', 0) for s in active))
            for server in servers:
                metrics = server['metrics']
                old = previous.get(server['actor_id'])
                if old:
                    for key, destination in [('generation_tokens_total', 'tokens'),
                                             ('preemptions_total', 'preemptions'),
                                             ('requests_success_total', 'success')]:
                        if key in metrics and key in old:
                            delta = metrics[key] - old[key] if metrics[key] >= old[key] else metrics[key]
                            if destination == 'tokens':
                                generated += delta
                                if server.get('rank', 0) >= 10000:
                                    guest_generated += delta
                            elif destination == 'preemptions': preemptions += delta
                            else: successes += delta
                previous[server['actor_id']] = metrics
                old_histograms = previous_histograms.get(server['actor_id'], {})
                for key, current in server.get('histograms', {}).items():
                    old_histogram = old_histograms.get(key)
                    if old_histogram is None or current['sum'] is None:
                        continue
                    aggregate = histograms.setdefault(key, {'count': 0, 'sum': 0, 'buckets': {}})
                    reset = current['count'] < old_histogram['count']
                    aggregate['count'] += current['count'] - (0 if reset else old_histogram['count'])
                    aggregate['sum'] += current['sum'] - (0 if reset else old_histogram['sum'])
                    for bound, value in current['buckets'].items():
                        delta = value - (0 if reset else old_histogram['buckets'].get(bound, 0))
                        aggregate['buckets'][bound] = aggregate['buckets'].get(bound, 0) + delta
                previous_histograms[server['actor_id']] = server.get('histograms', {})
                latest_histograms[server['actor_id']] = {k: v for k, v in metrics.items()
                    if 'seconds_mean' in k or 'seconds_p90' in k}
        summaries[policy] = {'kv_replica_max': describe(peak_kv), 'kv_replica_mean': describe(mean_kv),
            'kv_busy_replica': describe(busy_kv),
            'kv_busy_samples_ge_50_fraction': mean([float(v >= .5) for v in busy_kv]),
            'kv_max_samples_ge_50_fraction': mean([float(v >= .5) for v in peak_kv]),
            'requests_running': describe(running),
            'requests_waiting': describe(waiting), 'observed_generated_tokens': generated,
            'observed_guest_generated_tokens': guest_generated,
            'guest_generated_fraction': guest_generated / generated if generated else None,
            'observed_generation_tokens_per_second': generated / (end - begin),
            'observed_preemptions': preemptions, 'observed_successes': successes,
            'latency_interval': {key: histogram_summary(value) for key, value in histograms.items()},
            'last_cumulative_replica_latency': latest_histograms}
    summary['serving'] = summaries
    return summary


def summarize_events(directory, last_step=20):
    rows = read_jsonl(directory / 'dynamic/checkpoints/dynamic_inference.jsonl')
    kinds = ['borrow', 'return', 'early_return', 'renew', 'borrow_failed', 'return_failed']
    result = {'counts': {k: sum(r['kind'] == k for r in rows) for k in kinds}}
    result['hold_s'] = describe([r['detail'].get('held_s') for r in rows if r['kind'] in ('return', 'early_return')])
    result['borrow_execution_s'] = describe([r['detail'].get('execution_s') for r in rows if r['kind'] == 'borrow_complete'])
    result['return_execution_s'] = describe([r['detail'].get('execution_s') for r in rows if r['kind'] == 'return_complete'])
    result['directions'] = sorted({r['detail']['lender'] + ' -> ' + r['detail']['borrower']
                                   for r in rows if r['kind'] == 'borrow'})
    # Matched events use measured windows, never the stale completion snapshot.
    comparisons = []
    samples = [r for r in rows if r['kind'] == 'sample']
    result['mean_serving_cards'] = {policy: mean([r['policies'][policy]['serving_cards'] for r in samples])
                                    for policy in ['policy_1', 'policy_2', 'policy_3']}
    starts = {}
    for row in rows:
        if row['kind'] == 'borrow_start':
            starts[(row['detail']['lender'], row['detail']['borrower'])] = row
        elif row['kind'] == 'borrow_complete':
            detail = row['detail']
            start = starts.get((detail['lender'], detail['borrower']))
            if not start:
                continue
            begin, end = start['time_unix_s'], row['time_unix_s']
            before = [r for r in samples if begin - 30 <= r['time_unix_s'] <= begin]
            after = [r for r in samples if end + 10 <= r['time_unix_s'] <= end + 40]
            def window(records):
                ps = [r['policies'][detail['borrower']] for r in records]
                return {'samples': len(ps), 'kv_max': mean([max(p['replica_kv'].values(), default=0) for p in ps]),
                    'kv_replica_mean': mean([mean(list(p['replica_kv'].values())) for p in ps]),
                    'inflight': mean([p['inflight'] for p in ps]),
                    'waiting': mean([p['vllm'].get('requests_waiting') for p in ps]),
                    'generation_tokens_s': mean([p['vllm'].get('generation_tokens_per_second') for p in ps])}
            comparisons.append({'lend_id': detail['lend_id'], 'lender': detail['lender'],
                'borrower': detail['borrower'], 'before': window(before), 'after': window(after),
                'note': 'Descriptive windows; demand, training pauses and other transfers can differ.'})
    steps = read_jsonl(directory / 'dynamic/steps.jsonl')
    begin = next((r['observed_unix_s'] for r in steps if r['step'] == 2), None)
    end = next((r['observed_unix_s'] for r in steps if r['step'] == last_step), None)
    if begin is not None and end is not None:
        cards = {'policy_1': 4, 'policy_2': 12, 'policy_3': 4}
        area = {p: 0 for p in cards}
        allocation_seconds = {}
        previous = min(begin, rows[0]['time_unix_s']) if rows else begin
        for event in rows + [{'time_unix_s': end, 'kind': 'end'}]:
            if event['kind'] not in ['borrow', 'return', 'early_return', 'end']:
                continue
            timestamp = min(end, event['time_unix_s'])
            duration = max(0, timestamp - max(previous, begin))
            for policy in cards:
                area[policy] += cards[policy] * duration
            key = '/'.join(str(cards[p]) for p in cards)
            allocation_seconds[key] = allocation_seconds.get(key, 0) + duration
            previous = timestamp
            if event['time_unix_s'] >= end:
                break
            detail = event['detail']
            change = 2 if event['kind'] == 'borrow' else -2
            cards[detail['borrower']] += change
            cards[detail['lender']] -= change
        result['steady_time_weighted_serving_cards'] = {p: value / (end - begin) for p, value in area.items()}
        result['steady_allocation_seconds'] = allocation_seconds
    result['borrow_windows'] = comparisons
    event_rows = []
    for row in rows:
        if row['kind'] not in ['borrow_start', 'borrow_complete', 'return_start', 'return_complete', 'early_return_requested']:
            continue
        entry = {'time_unix_s': row['time_unix_s'], 'step': row.get('step'), 'kind': row['kind'], **row['detail']}
        for policy, state in row.get('policies', {}).items():
            for key in ['serving_cards', 'kv_decision', 'sample_age_s', 'inflight']:
                entry[f'{policy}/{key}'] = state.get(key)
            entry[f'{policy}/kv_raw_max'] = max(state.get('replica_kv', {}).values(), default=0)
        event_rows.append(entry)
    if event_rows:
        with (directory / 'scheduling_events.csv').open('w') as stream:
            keys = sorted({k for row in event_rows for k in row})
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(event_rows)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    last_step = int(manifest.get('steps', 20))
    steady_start = 3 if last_step >= 3 else 1
    matched_start = max(steady_start, last_step - 9)
    summary = {mode: summarize_run(directory / mode, steady_start, last_step)
               for mode in ['dynamic', 'static']}
    for mode in ['dynamic', 'static']:
        info = manifest.get('runs', {}).get(mode, {})
        steps = read_jsonl(directory / mode / 'steps.jsonl')
        if 'started_unix_s' in info and steps:
            summary[mode]['launch_to_first_step_s'] = steps[0]['observed_unix_s'] - info['started_unix_s']
        if 'finished_unix_s' in info:
            summary[mode]['total_wall_s'] = info['finished_unix_s'] - info['started_unix_s']
            summary[mode]['exit_code'] = info['exit_code']
    summary['matched_last_10'] = {mode: summarize_run(
        directory / mode, first_step=matched_start, last_step=last_step)
                                  for mode in ['dynamic', 'static']}
    summary['events'] = summarize_events(directory, last_step)
    (directory / 'comparison_summary.json').write_text(json.dumps(summary, indent=2))
    def fmt(value):
        return '未观测到' if value is None else f'{value:.3f}'
    def extract(mode, path):
        value = summary[mode]
        for key in path.split('/'):
            value = value.get(key, {}) if isinstance(value, dict) else {}
        return value if isinstance(value, (int, float)) else None
    profile = manifest.get('profile')
    high_load = profile in ('high_kv', 'random_routing')
    if profile == 'random_routing':
        workload = ('固定 seed=42、256 prompt × 4 rollout、1024 并发；请求从 agent_1 '
                    '入口进入，随机执行 1–10 轮，每轮随机路由到 agent_1/2/3，'
                    'max_tokens 从 4096–8192 采样。')
    elif profile == 'high_kv':
        workload = '固定 seed=42、256 prompt × 4 rollout、1024 并发，agent_1/2/3 输出上限为 10240/512/4096 token。'
    else:
        workload = '固定 seed=42、64 prompt × 4 rollout、128 并发，agent_1/2/3 输出上限为 4096/512/4096 token。'
    order = ' 后 '.join(manifest.get('mode_order', ['dynamic', 'static']))
    lines = ['# Dynamic 推理调度对照实验', '', *((directory / 'interpretation.md').read_text().splitlines() if (directory / 'interpretation.md').exists() else []), '', '## 实验设置与完整指标', '',
        f'两组从同一初始模型启动，各 {last_step} step；前 2 step 为预热，step {steady_start}–{last_step} 为主要统计窗口（不代表调度已经收敛）。',
        workload,
        '训练卡为 4/4/4，独立推理卡为 4/12/4，TP2、独立 replica 为 2/6/2。',
        *(['模型为 Qwen2.5-0.5B-Instruct；prompt/总上下文上限为 12288/24576 token，gpu_memory_utilization=0.5，max_num_seqs=256/256/160，max_num_batched_tokens=4096。',
           '权重同步分桶为 128 MiB；外部 MAS 任务使用 SPREAD。Dynamic 仅使用 KV 信号，进入/退出/借出后预测上限为 55%/30%/40%，借卡冷却 30 秒。',
           f'运行顺序为 {order}；KV≥50% 占比按有效采样计算，并非精确的墙钟时间占比。'] if high_load else []), '',
        '| 指标 | Dynamic | Static |', '|---|---:|---:|']
    for label, path in [('已完成 step', 'completed_steps'), ('稳定阶段 step 均值（秒）', 'steady_step_s/mean'),
        ('启动到首 step 完成（秒）', 'launch_to_first_step_s'), ('全部运行墙钟时间（秒）', 'total_wall_s'),
        ('稳定阶段 step P90（秒）', 'steady_step_s/p90'), ('稳定阶段 step 标准差（秒）', 'steady_step_s/std'),
        (f'训练 step 耗时总和（秒，{last_step} 步）', 'training_step_s_total')]:
        lines.append(f'| {label} | {fmt(extract("dynamic", path))} | {fmt(extract("static", path))} |')
    dynamic, static = extract('dynamic', 'steady_step_s/mean'), extract('static', 'steady_step_s/mean')
    if dynamic and static:
        lines += ['', f'稳定阶段平均 step 耗时变化：{(dynamic/static-1)*100:+.2f}%；静态/动态速度比：{static/dynamic:.3f}×。']
    lines += ['', f'## 资源对比的匹配窗口：step {matched_start}–{last_step}', '',
        f'以下使用相同的末段窗口，完整 step {steady_start}–{last_step} 资源统计另存 JSON；请结合实际监控覆盖率判断资源和能耗比较是否有效。', '',
        f'| 指标（step {matched_start}–{last_step}） | Dynamic | Static |', '|---|---:|---:|']
    for label, key, sub in [('step 均值（秒）', 'steady_step_s', 'mean'),
        ('GPU 利用率（%）', 'gpu_utilization.gpu', 'mean'),
        ('GPU 显存（MiB）', 'gpu_memory.used', 'mean'),
        ('GPU 功率（W）', 'gpu_power.draw', 'mean'),
        ('观测能耗（kWh）', 'observed_energy_kwh', None),
        ('能耗覆盖率', 'energy_monitor_coverage', None),
        ('分配 GPU 时', 'allocated_gpu_hours', None),
        ('节点 CPU 利用率（%）', 'host_cpu_percent', 'mean'),
        ('节点主机内存（GiB）', 'host_memory_used_gib', 'mean')]:
        values = [summary['matched_last_10'][mode].get(key) for mode in ['dynamic', 'static']]
        if sub:
            values = [v.get(sub) if isinstance(v, dict) else None for v in values]
        lines.append(f'| {label} | {fmt(values[0])} | {fmt(values[1])} |')
    lines += ['', '## 各 policy 实际推理负载', '', '| Policy / 指标 | Dynamic | Static |', '|---|---:|---:|']
    for policy in ['policy_1', 'policy_2', 'policy_3']:
        for label, metric in [('KV 最大 replica 利用率均值（比例）', 'kv_replica_max/mean'),
                              ('KV 活跃 replica 平均利用率（比例）', 'kv_replica_mean/mean'),
                              ('KV 忙碌 replica 均值（比例）', 'kv_busy_replica/mean'),
                              ('KV 忙碌 replica P90（比例）', 'kv_busy_replica/p90'),
                              ('KV 峰值（比例）', 'kv_replica_max/max'),
                              ('忙碌 replica 采样中 KV≥50% 占比', 'kv_busy_samples_ge_50_fraction'),
                              ('运行请求均值', 'requests_running/mean'), ('排队请求均值', 'requests_waiting/mean'),
                              ('观测生成 token/s', 'observed_generation_tokens_per_second'),
                              ('Guest 生成 token 占比', 'guest_generated_fraction'),
                              ('观测抢占次数', 'observed_preemptions')]:
            path = f'serving/{policy}/{metric}'
            lines.append(f'| {policy} / {label} | {fmt(extract("dynamic", path))} | {fmt(extract("static", path))} |')
        for metric in ['request_queue_time_seconds', 'time_to_first_token_seconds', 'e2e_request_latency_seconds']:
            for stat in ['mean', 'p90']:
                path = f'serving/{policy}/latency_interval/{metric}/{stat}'
                lines.append(f'| {policy} / {metric}/{stat} | {fmt(extract("dynamic", path))} | {fmt(extract("static", path))} |')
    lines += ['', '## 任务质量与训练指标', '',
        '当前 reward 是答案子串命中率：长输出中包含正确答案即得 1 分。它不是严格正确率，也不是独立测试集精度。',
        '1024 行训练数据由 8 道 mock 题重复生成，不能把重复样本当作 1024 个独立任务；验证集同源且未运行独立评测。',
        '三个 policy 共享最终任务 reward，表中重复的 score 不是三份独立精度评测。', '',
        '| Policy / 指标（稳定阶段均值） | Dynamic | Static |', '|---|---:|---:|']
    for policy in ['policy_1', 'policy_2', 'policy_3']:
        for metric in ['critic/score/mean', 'response_length/mean', 'response_length/clip_ratio',
                       'actor/entropy', 'actor/ppo_kl', 'actor/grad_norm', 'actor/advantages_std',
                       'actor/pg_clipfrac', 'off_policy/trajectory_staleness/mean', 'off_policy/trajectory_spans/mean']:
            values = [summary[mode].get('steady_metrics', {}).get(f'{policy}/{metric}', {}).get('mean') for mode in ['dynamic', 'static']]
            if any(v is not None for v in values):
                lines.append(f'| {policy}/{metric} | {fmt(values[0])} | {fmt(values[1])} |')
    events = summary['events']
    lines += ['', '## 调度行为', '', f'事件计数：`{json.dumps(events["counts"], ensure_ascii=False)}`。',
        f'实际方向：{", ".join(events["directions"]) or "未观测到"}。',
        f'调度采样时平均可用推理卡：`{json.dumps(events["mean_serving_cards"])}`（采样均值，非墙钟加权）。',
        f'持有时间均值：{fmt(events["hold_s"].get("mean"))} 秒；最短：{fmt(events["hold_s"].get("min"))} 秒。',
        f'借卡执行耗时均值：{fmt(events["borrow_execution_s"].get("mean"))} 秒；还卡执行耗时均值：{fmt(events["return_execution_s"].get("mean"))} 秒。', '',
        '每次借卡前 30 秒与完成后 10–40 秒的实测窗口已写入 comparison_summary.json 的 events.borrow_windows。',
        '该窗口比较不控制请求到达、训练暂停或其他同时发生的借还，因此仅作描述，整体收益以匹配的静态对照为准。', '',
        '## 解释边界', '',
        f'- step {steady_start}–{last_step} 是预先约定的统计窗口，不代表调度已收敛；另存可用末段窗口的耗时敏感性分析。',
        '- step 时间使用训练器计时；监控窗口边界来自原始 worker 日志轮询（约 5 秒误差）。各步时间来源记录在 JSON 的 timestamp_sources 中。',
        '- 活跃推理副本由借还事件中的 active_lends 重建，排除睡眠 home 与未激活 guest 的旧 gauge；累计 token 差分仍包含各副本实际生成量。',
        '- 仅一组顺序运行；固定随机种子不能保证异步调度后的生成结果完全一致。未进行多 seed 置信区间估计。',
        '- GPU 统计覆盖整个 32 卡训练与推理环境；显存占用、GPU 计算利用率、KV cache 比例是不同指标。',
        '- 节点 CPU、主机内存和网络计数包含系统与其他进程，不能全部归因于本实验。',
        '- 能耗由采样功率积分估计，未覆盖超过 45 秒的采样缺口；覆盖率单独报告。',
        '- Serving 计数器按 replica 差分，动态新增 replica 的首次观测之前 token 不计入，可能低估少量吞吐。',
        '- 延迟表使用稳定窗口逐 replica 直方图增量合并，P90 为桶内插值估计；首次采样前的观测不计入。',
        '- 原始 last_cumulative_replica_latency 另存累计摘要，包含预热，不用于稳定阶段延迟表。',
        ('- 随机路由 harness 保留自然 EOS，并以随机 min_tokens 下限维持负载；截断率需结合每轮随机预算解释。'
         if profile == 'random_routing' else
         '- ignore_eos 强制长输出导致截断率接近 1 是预期负载设置，不直接表示生成失败。'),
        '- 初始化与 guest 预创建开销另见 manifest 的启动时间及首 step 时间；稳定阶段表不含初始化。', '',
        '原始文件：每组 train.log、steps.jsonl、gpu.jsonl、serving.jsonl；dynamic/checkpoints/dynamic_inference.jsonl。']
    (directory / 'comparison_report.md').write_text('\n'.join(lines) + '\n')
    for mode in ['dynamic', 'static']:
        rows = read_jsonl(directory / mode / 'steps.jsonl')
        keys = sorted({key for row in rows for key in row['metrics']})
        with (directory / f'{mode}_steps.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['step', 'observed_unix_s', *keys])
            writer.writeheader()
            for row in rows:
                writer.writerow({'step': row['step'], 'observed_unix_s': row['observed_unix_s'], **row['metrics']})
    if all(summary[mode]['completed_steps'] >= last_step for mode in ['dynamic', 'static']):
        try:
            plot_comparison(directory, summary)
        except ModuleNotFoundError as error:
            # Training remains in zzh_env; the system Python has matplotlib.
            report_python = Path('/usr/bin/python')
            if error.name != 'matplotlib' or Path(sys.executable).resolve() == report_python.resolve():
                raise
            subprocess.run([str(report_python), str(Path(__file__).resolve()), str(directory)], check=True)
            return
    print(directory / 'comparison_report.md')


def plot_comparison(directory, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {'dynamic': '#167d9a', 'static': '#d57a32'}
    for mode in colors:
        rows = read_jsonl(directory / mode / 'steps.jsonl')
        axes[0, 0].plot([r['step'] for r in rows], [r['metrics']['perf/time_per_step'] for r in rows],
                        marker='o', markersize=3, label=mode, color=colors[mode])
    axes[0, 0].axvspan(.5, 2.5, color='grey', alpha=.15, label='warmup')
    axes[0, 0].set(xlabel='Step', ylabel='Seconds', title='Step duration')
    axes[0, 0].legend()
    axes[0, 1].bar(list(colors), [summary['matched_last_10'][m].get('gpu_utilization.gpu', {}).get('mean', float('nan'))
                                  for m in colors], color=list(colors.values()))
    axes[0, 1].set(ylabel='GPU utilization (%)', title='All 32 GPUs, matched steps 11–20', ylim=(0, 100))
    for index, mode in enumerate(colors):
        x = [i + (index - .5) * .36 for i in range(3)]
        kv = [summary[mode].get('serving', {}).get(f'policy_{i}', {}).get('kv_replica_mean', {}).get('mean', float('nan')) * 100
              for i in range(1, 4)]
        rewards = [summary[mode]['steady_metrics'].get(f'policy_{i}/critic/score/mean', {}).get('mean', float('nan'))
                   for i in range(1, 4)]
        axes[1, 0].bar(x, kv, .36, label=mode, color=colors[mode])
        axes[1, 1].bar(x, rewards, .36, label=mode, color=colors[mode])
    for axis in axes[1]:
        axis.set_xticks([0, 1, 2], ['policy_1', 'policy_2', 'policy_3'])
        axis.legend()
    axes[1, 0].set(ylabel='KV utilization (%)', title='Mean across active replicas, steps 3–20')
    axes[1, 1].set(ylabel='Substring reward (proxy)', title='Training reward; not held-out accuracy')
    for axis in axes.flat:
        axis.grid(axis='y', alpha=.2)
    fig.savefig(directory / 'comparison.png', dpi=180)
    fig.savefig(directory / 'comparison.svg')
    plt.close(fig)
    trace = [r for r in read_jsonl(directory / 'dynamic/checkpoints/dynamic_inference.jsonl') if r['kind'] == 'sample']
    if not trace:
        return
    first = trace[0]['time_unix_s']
    times = [(r['time_unix_s'] - first) / 60 for r in trace]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for policy, color in zip(['policy_1', 'policy_2', 'policy_3'], ['#167d9a', '#d57a32', '#8950a1']):
        states = [r['policies'][policy] for r in trace]
        cards = {'policy_1': 4, 'policy_2': 12, 'policy_3': 4}
        event_times, allocations = [0], [cards[policy]]
        for event in read_jsonl(directory / 'dynamic/checkpoints/dynamic_inference.jsonl'):
            if event['kind'] not in ['borrow', 'return', 'early_return']:
                continue
            delta = 2 if event['kind'] == 'borrow' else -2
            cards[event['detail']['borrower']] += delta
            cards[event['detail']['lender']] -= delta
            event_times.append((event['time_unix_s'] - first) / 60)
            allocations.append(cards[policy])
        axes[0].step(event_times, allocations, where='post', label=policy, color=color)
        axes[1].plot(times, [s['kv_decision'] * 100 for s in states], label=policy, color=color)
        axes[2].plot(times, [s['inflight'] for s in states], label=policy, color=color)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    threshold = 55 if manifest.get('profile') in ('high_kv', 'random_routing') else 1
    axes[1].axhline(threshold, color='grey', linestyle='--', linewidth=1, label='Enter / lender recall threshold')
    axes[0].set(ylabel='Serving GPUs', title='Dynamic allocation and observed load')
    axes[1].set(ylabel='Smoothed KV signal (%)')
    axes[2].set(ylabel='Inflight requests', xlabel='Minutes from first scheduler sample')
    for axis in axes:
        axis.legend(loc='upper right', ncol=3)
        axis.grid(alpha=.2)
    fig.savefig(directory / 'scheduling_timeline.png', dpi=180)
    fig.savefig(directory / 'scheduling_timeline.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
