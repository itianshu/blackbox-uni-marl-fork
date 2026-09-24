"""Generate a report for the successfully completed prefix of a dynamic run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any

from analyze_2policy_comparison import build_run, finite, flatten, read_jsonl, write_run


POLICIES = ("policy_1", "policy_2")


def values(rows: list[dict[str, Any]], path: str) -> list[float]:
    result = []
    for row in rows:
        value: Any = row
        for part in path.split("/"):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if finite(value):
            result.append(float(value))
    return result


def avg(rows: list[dict[str, Any]], path: str) -> float | None:
    found = values(rows, path)
    return statistics.mean(found) if found else None


def total(rows: list[dict[str, Any]], path: str) -> float:
    return sum(values(rows, path))


def fmt(value: Any, digits: int = 2, percent: bool = False) -> str:
    if not finite(value):
        return "N/A"
    number = float(value) * (100 if percent else 1)
    return f"{number:.{digits}f}"


def seconds(value: Any) -> str:
    if not finite(value):
        return "N/A"
    value = float(value)
    return f"{value:.1f} ({value / 60:.1f} min)"


def instantaneous_generation_rates(
    trace: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> dict[int, dict[str, float | None]]:
    result: dict[int, dict[str, float | None]] = {}
    for row in rows:
        samples = [
            sample for sample in trace
            if sample.get("kind") == "sample"
            and row["window_start_unix_s"] < sample.get("time_unix_s", 0) <= row["window_end_unix_s"]
        ]
        by_policy: dict[str, float | None] = {}
        for policy in POLICIES:
            rates = [
                sample.get("policies", {}).get(policy, {}).get("vllm", {}).get("generation_tokens_per_second")
                for sample in samples
            ]
            rates = [float(rate) for rate in rates if finite(rate) and rate > 0]
            by_policy[policy] = statistics.mean(rates) if rates else None
        policy_rates = [rate for rate in by_policy.values() if finite(rate)]
        by_policy["total"] = sum(policy_rates) if policy_rates else None
        result[int(row["step"])] = by_policy
    return result


def weighted_prefix_rate(rows: list[dict[str, Any]], policy: str) -> float | None:
    queries = total(rows, f"rollout/policies/{policy}/prefix_cache_queries")
    hits = total(rows, f"rollout/policies/{policy}/prefix_cache_hits")
    return hits / queries if queries else None


def weighted_gpu(rows: list[dict[str, Any]], key: str) -> float | None:
    pairs = [
        (row.get("gpu", {}).get(key), row.get("gpu", {}).get("sample_count"))
        for row in rows
    ]
    pairs = [(float(value), float(count)) for value, count in pairs if finite(value) and finite(count) and count > 0]
    denominator = sum(count for _, count in pairs)
    return sum(value * count for value, count in pairs) / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--max-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    directory = args.directory.resolve()
    dynamic = directory / "dynamic"
    manifest = json.loads((directory / "manifest.json").read_text())
    rows, _ = build_run(dynamic, manifest.get("runs", {}).get("dynamic", {}))
    rows = [row for row in rows if int(row["step"]) <= args.max_step]
    if not rows or int(rows[-1]["step"]) != args.max_step:
        raise SystemExit(f"completed step {args.max_step} was not found")

    # Keep machine-readable artifacts next to the source logs.  The nested
    # JSONL retains every value passed to Tracking.log(); CSV is convenient
    # for plotting and spreadsheet inspection.
    artifact_dir = directory / f"partial_steps_1_{args.max_step}"
    artifact_dir.mkdir(exist_ok=True)
    write_run(artifact_dir, rows)
    tracking_summary = json.loads((artifact_dir / "trainer_tracking_summary.json").read_text())
    tracking_lines = [
        "# MultiAgentsPPOTrainer.fit() Tracking 数值项汇总",
        "",
        f"范围：成功完成的 step 1–{args.max_step}；失败的 step {args.max_step + 1} 不纳入统计。",
        "",
        "| Tracking key | N | Mean | Min | P50 | P90 | Max | Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in sorted(tracking_summary.items()):
        tracking_lines.append(
            f"| `{key}` | {item.get('n', 0)} | {fmt(item.get('mean'), 6)} | "
            f"{fmt(item.get('min'), 6)} | {fmt(item.get('p50'), 6)} | "
            f"{fmt(item.get('p90'), 6)} | {fmt(item.get('max'), 6)} | "
            f"{fmt(item.get('std'), 6)} |"
        )
    (artifact_dir / "trainer_tracking_summary.md").write_text("\n".join(tracking_lines) + "\n")

    trace = read_jsonl(dynamic / "checkpoints/dynamic_inference.jsonl")
    instant = instantaneous_generation_rates(trace, rows)
    flat_rows = [flatten(row) for row in rows]
    numeric_keys = sorted({key for row in flat_rows for key, value in row.items() if finite(value)})
    summary = {
        "source_run": str(directory),
        "included_steps": [int(row["step"]) for row in rows],
        "excluded_failure_step": args.max_step + 1,
        "metrics": {
            key: {
                "mean": statistics.mean(found),
                "min": min(found),
                "max": max(found),
            }
            for key in numeric_keys
            if (found := [float(row.get(key)) for row in flat_rows if finite(row.get(key))])
        },
        "instantaneous_generation_tokens_per_s": instant,
    }
    summary_path = artifact_dir / "partial_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    total_wall = total(rows, "wall_clock_s")
    generation_tokens = total(rows, "rollout/generation_tokens")
    prompt_tokens = total(rows, "rollout/prompt_tokens")
    completed_requests = total(rows, "rollout/completed_requests")
    all_instant = [entry["total"] for entry in instant.values() if finite(entry.get("total"))]
    overall_instant = statistics.mean(all_instant) if all_instant else None
    gpu_util = weighted_gpu(rows, "effective_utilization")
    gpu_bubble = weighted_gpu(rows, "bubble_fraction_lt_10pct")
    memory_peak = max(values(rows, "gpu/memory_used_mib_max"), default=math.nan)
    evicted_samples = sum(
        float(row.get("trainer_tracking", {}).get("training/rollout_failure/evicted_samples", 0))
        for row in rows
    )
    waiting_reason_available = any(
        row.get("rollout", {}).get("waiting_by_reason_available") for row in rows
    )

    lines = [
        "# 第一次 retry：OOM 前 7 个成功 step 实验报告",
        "",
        "## 范围与有效性",
        "",
        f"本报告只包含动态调度运行中完整结束并写入 Tracking 的 step 1–{args.max_step}。"
        f"Step {args.max_step + 1} 在 checkpoint-engine 权重同步时 OOM，未纳入任何均值或总计。",
        "这 7 个 step 的 rollout、训练更新和 step 级指标均已完成，因此可用于分析动态调度行为、吞吐、训练效率与训练信号；但没有对应静态组，不能据此计算动态调度相对静态基线的收益。",
        "该运行的 vLLM `gpu_memory_utilization=0.75`，高于当前修复配置的 0.70。显存结果代表旧配置，不应外推为当前运行的安全余量。",
        "",
        "## 配置摘要",
        "",
        "- 两个 Qwen3-1.7B policy；训练 policy1=8 卡 FSDP8/SP2，policy2=4 卡 FSDP4/SP2。",
        "- 推理基线 policy1=12 个、policy2=8 个 TP1 单卡 replica；允许双向最多借 4 个 replica。",
        "- batch size 128，rollout n=4，最大并发 512；prompt/response/max-model-len 分别为 12288/10240/24576。",
        "- 7–10 轮交互，agent1 首尾固定，中间 agent1:agent2=1:1；单次输出预算 8192–10240 token。",
        "- Prefix caching 关闭；动态瓶颈连续确认 10 次后才允许调度。",
        "",
        "## 7-step 总览",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 完成 step | {len(rows)} |",
        f"| 累计 step wall clock | {seconds(total_wall)} |",
        f"| 平均 step wall clock | {seconds(avg(rows, 'wall_clock_s'))} |",
        f"| 平均 rollout 等待时间 | {seconds(avg(rows, 'rollout_wait_s'))} |",
        f"| Rollout 占 step 时间比例 | {fmt(total(rows, 'rollout_wait_s') / total_wall, percent=True)}% |",
        f"| 累计 generation tokens | {generation_tokens:,.0f} |",
        f"| 累计 prompt + generation tokens | {prompt_tokens + generation_tokens:,.0f} |",
        f"| 完成 vLLM 请求 | {completed_requests:,.0f} |",
        f"| 全 step 摊销 generation tokens/s | {generation_tokens / total_wall:,.1f} |",
        f"| 活跃生成期瞬时 generation tokens/s（step 均值） | {fmt(overall_instant, 1)} |",
        f"| Trainer throughput 平均 | {fmt(avg(rows, 'trainer_throughput'), 1)} |",
        f"| GPU 有效利用率（NVML 样本加权） | {fmt(gpu_util)}% |",
        f"| GPU 空泡率，util<10%（NVML 样本加权） | {fmt(gpu_bubble, percent=True)}% |",
        f"| 训练 MFU 平均 | {fmt(avg(rows, 'mfu_mean'), percent=True)}% |",
        f"| 单卡显存观测峰值 | {fmt(memory_peak, 0)} MiB |",
        f"| 平均 trajectory/step | {fmt(avg(rows, 'trajectory_count'), 1)} |",
        f"| 平均 MAS rollout/step | {fmt(avg(rows, 'mas_rollout_count'), 1)} |",
        f"| Rollout failure evicted samples 总计 | {evicted_samples:.0f} |",
        f"| Staleness 平均（含 warmup step1=0） | {fmt(avg(rows, 'staleness_mean'), 3)} |",
        f"| Policy 并行临界路径节省/step | {seconds(avg(rows, 'policy_parallel_overlap_saved_s'))} |",
        f"| Policy 并行效率 | {fmt(avg(rows, 'policy_parallel_efficiency'), percent=True)}% |",
        "",
        "## 每 step：时长、吞吐和负载",
        "",
        "| Step | Wall min | Rollout min | Gen tok/s（wall 摊销） | 活跃期 tok/s | Trainer TP | GPU util % | 空泡 <10% | MFU % | Trajectory | Staleness |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        step = int(row["step"])
        lines.append(
            f"| {step} | {fmt(row.get('wall_clock_s') / 60)} | {fmt(row.get('rollout_wait_s') / 60)} | "
            f"{fmt(row['rollout'].get('generation_tokens_per_s'), 1)} | {fmt(instant[step].get('total'), 1)} | "
            f"{fmt(row.get('trainer_throughput'), 1)} | {fmt(row['gpu'].get('effective_utilization'))} | "
            f"{fmt(row['gpu'].get('bubble_fraction_lt_10pct'), percent=True)}% | {fmt(row.get('mfu_mean'), percent=True)}% | "
            f"{fmt(row.get('trajectory_count'), 0)} | {fmt(row.get('staleness_mean'), 3)} |"
        )

    lines += [
        "",
        "### fit() Tracking 补充指标",
        "",
        "| Step | Training tokens | Evicted samples | Response clip p1/p2 | Advantage mean p1/p2 | Grad norm p1/p2 | Update actor s p1/p2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tracked = row["trainer_tracking"]
        p1, p2 = row["policies"]["policy_1"], row["policies"]["policy_2"]
        lines.append(
            f"| {row['step']} | {fmt(row.get('training_tokens'), 0)} | "
            f"{fmt(tracked.get('training/rollout_failure/evicted_samples'), 0)} | "
            f"{fmt(tracked.get('policy_1/response_length/clip_ratio'), percent=True)}/{fmt(tracked.get('policy_2/response_length/clip_ratio'), percent=True)}% | "
            f"{fmt(tracked.get('policy_1/actor/advantages_mean'), 5)}/{fmt(tracked.get('policy_2/actor/advantages_mean'), 5)} | "
            f"{fmt(p1.get('grad_norm'), 4)}/{fmt(p2.get('grad_norm'), 4)} | "
            f"{fmt(p1.get('update_actor_s'), 1)}/{fmt(p2.get('update_actor_s'), 1)} |"
        )

    lines += [
        "",
        "## 每 step：KV、排队和 prefix cache",
        "",
        "KV mean 是所有活跃 replica/采样点的均值；KV p90/max 更能体现局部饱和和 OOM 风险。",
        "",
        "| Step | KV p1 mean/p90/max | KV p2 mean/p90/max | Waiting p1 mean/max | Waiting p2 mean/max | Prefix hit p1/p2 | Preemptions p1/p2 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p1 = row["rollout"]["policies"]["policy_1"]
        p2 = row["rollout"]["policies"]["policy_2"]
        lines.append(
            f"| {row['step']} | {fmt(p1.get('kv_mean'), percent=True)}/{fmt(p1.get('kv_p90'), percent=True)}/{fmt(p1.get('kv_max'), percent=True)}% | "
            f"{fmt(p2.get('kv_mean'), percent=True)}/{fmt(p2.get('kv_p90'), percent=True)}/{fmt(p2.get('kv_max'), percent=True)}% | "
            f"{fmt(p1.get('requests_waiting_mean'), 1)}/{fmt(p1.get('requests_waiting_max'), 0)} | "
            f"{fmt(p2.get('requests_waiting_mean'), 1)}/{fmt(p2.get('requests_waiting_max'), 0)} | "
            f"{fmt(p1.get('prefix_cache_hit_rate'), percent=True)}/{fmt(p2.get('prefix_cache_hit_rate'), percent=True)}% | "
            f"{fmt(p1.get('preemptions'), 0)}/{fmt(p2.get('preemptions'), 0)} |"
        )

    lines += [
        "",
        "## 每 step：训练效率、训推一致性和训练信号",
        "",
        "| Step | Corr p1/p2 | Prob diff mean p1/p2 | Rollout/train KL | PPL ratio | Reward p1/p2 | Entropy p1/p2 | PG loss p1/p2 | PPO KL p1/p2 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p1, p2 = row["policies"]["policy_1"], row["policies"]["policy_2"]
        lines.append(
            f"| {row['step']} | {fmt(p1.get('rollout_actor_probs_corr'), 3)}/{fmt(p2.get('rollout_actor_probs_corr'), 3)} | "
            f"{fmt(p1.get('rollout_probs_diff_mean'), 5)}/{fmt(p2.get('rollout_probs_diff_mean'), 5)} | "
            f"{fmt(row.get('rollout_training_kl'), 5)} | {fmt(row.get('rollout_training_ppl_ratio'), 5)} | "
            f"{fmt(p1.get('reward_mean'), 5)}/{fmt(p2.get('reward_mean'), 5)} | "
            f"{fmt(p1.get('entropy'), 5)}/{fmt(p2.get('entropy'), 5)} | "
            f"{fmt(p1.get('pg_loss'), 5)}/{fmt(p2.get('pg_loss'), 5)} | "
            f"{fmt(p1.get('ppo_kl'), 5)}/{fmt(p2.get('ppo_kl'), 5)} |"
        )

    lines += [
        "",
        "| Step | MFU p1/p2 | Actor allocated GiB p1/p2 | Actor reserved GiB p1/p2 | Policy overlap saved s | Parallel efficiency |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        p1, p2 = row["policies"]["policy_1"], row["policies"]["policy_2"]
        lines.append(
            f"| {row['step']} | {fmt(p1.get('mfu'), percent=True)}/{fmt(p2.get('mfu'), percent=True)}% | "
            f"{fmt(p1.get('actor_memory_allocated_gib'))}/{fmt(p2.get('actor_memory_allocated_gib'))} | "
            f"{fmt(p1.get('actor_memory_reserved_gib'))}/{fmt(p2.get('actor_memory_reserved_gib'))} | "
            f"{fmt(row.get('policy_parallel_overlap_saved_s'), 1)} | {fmt(row.get('policy_parallel_efficiency'), percent=True)}% |"
        )

    lines += [
        "",
        "## 每 step：动态调度",
        "",
        "| Step | Borrow | Return | Early return | Active lends mean/max | Cards p1/p2 mean | Borrow/return time s | Clone time s | Failures |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        scheduling = row["dynamic_scheduling"]
        cards = scheduling["serving_cards_mean"]
        lines.append(
            f"| {row['step']} | {scheduling['borrow_events']} | {scheduling['return_events']} | "
            f"{scheduling['early_return_requests']} | {fmt(scheduling.get('active_lends_mean'))}/{scheduling['active_lends_max']} | "
            f"{fmt(cards.get('policy_1'))}/{fmt(cards.get('policy_2'))} | "
            f"{fmt(scheduling.get('borrow_execution_s_mean'))}/{fmt(scheduling.get('return_execution_s_mean'))} | "
            f"{fmt(scheduling.get('replica_clone_s_mean'))} | "
            f"{scheduling['borrow_failures']}/{scheduling['return_failures']} |"
        )

    lines += [
        "",
        "## 结论",
        "",
        f"1. **吞吐受 rollout 主导。** 7 个 step 平均 {fmt(avg(rows, 'wall_clock_s') / 60)} 分钟，其中 rollout 平均 "
        f"{fmt(avg(rows, 'rollout_wait_s') / 60)} 分钟，占累计 wall clock 的 {fmt(total(rows, 'rollout_wait_s') / total_wall, percent=True)}%。",
        f"2. **训练 GPU 尚有明显空泡。** 样本加权 GPU 利用率为 {fmt(gpu_util)}%，util<10% 的空泡占 "
        f"{fmt(gpu_bubble, percent=True)}%；训练 MFU 稳定在 {fmt(avg(rows, 'mfu_mean'), percent=True)}% 左右。Step 1 利用率高，后续异步流水线等待更明显。",
        "3. **平均 KV 不高，但尾部已经饱和。** 两侧每个 step 的 KV p90 约 97–99%，max 均达到 100%；"
        "仅看 mean（约 38–57%）会严重低估局部 replica 压力。高 preemption 与 waiting 也说明长序列负载处于饱和区。",
        f"4. **动态借还本身工作正常。** 共记录 {total(rows, 'dynamic_scheduling/borrow_events'):.0f} 次借卡、"
        f"{total(rows, 'dynamic_scheduling/return_events'):.0f} 次还卡，最大同时借 4 卡，borrow/return failure 都为 0。"
        "显式事件偶有 HDFS 追加缺口，报告已用 active_lends 拓扑变化补全并单独保留原始事件计数。",
        "5. **训推一致性随训练推进改善。** rollout/train KL 从 step1 的 "
        f"{fmt(rows[0].get('rollout_training_kl'), 5)} 降到 step7 的 {fmt(rows[-1].get('rollout_training_kl'), 5)}，"
        f"PPL ratio 从 {fmt(rows[0].get('rollout_training_ppl_ratio'), 5)} 收敛到 {fmt(rows[-1].get('rollout_training_ppl_ratio'), 5)}；"
        "policy1/2 概率相关性也整体上升。",
        "6. **旧显存配置没有安全余量。** 观测单卡显存峰值从 step1 的 "
        f"{fmt(rows[0]['gpu'].get('memory_used_mib_max'), 0)} MiB 上升至 step7 的 "
        f"{fmt(rows[-1]['gpu'].get('memory_used_mib_max'), 0)} MiB，最终 step8 权重同步还需约 1.16 GiB 时 OOM。"
        "因此这 7 步可用于性能分析，但 0.75 配置不能继续使用。",
        "7. **Reward 已高度饱和。** 两个 policy 的平均 reward 长期约 0.994–0.998；它能证明流程有训练信号，"
        "但对策略质量差异的区分度有限，最终对比仍应以当前完整动态/静态实验为准。",
        f"8. **成功 step 中仍有轨迹淘汰。** 7 步累计记录 {evicted_samples:.0f} 个 "
        "rollout failure evicted samples；训练 step 能完成，但这说明旧运行在借还窗口内存在请求失败。"
        "因此性能数据和保留下来的训练指标有效，但不能把该运行描述成无轨迹损失。当前实验已经加入请求重试来解决这一点。",
        "",
        "## 指标口径与缺失项",
        "",
        "- Gen tok/s（wall 摊销）= step 内 vLLM generation-token counter 增量 / 完整 step wall time；活跃期 tok/s 是动态 trace 中非零瞬时生成速率的 step 均值。",
        "- GPU 有效利用率与空泡率来自 15 秒 NVML 采样；空泡定义为 GPU utilization <10%。",
        "- Policy overlap 是两个 policy 的 old-log-prob/update-actor/update-weights 串行耗时减去并行临界路径，是 policy 级 overlap 代理，不等价于 CUDA kernel overlap。",
        "- Prefix cache 配置为关闭；日志仍出现约 0.7% 的内部 counter 命中，只如实记录，不解释为 prefix-cache 加速收益。",
        f"- vLLM request waiting by reason：{'已采集' if waiting_reason_available else '该版本没有导出带 reason 标签的 metric，因此为 N/A；总 waiting 已完整采集'}。",
        "- `per_step_metrics.jsonl` 保存每个 step 的嵌套指标以及 MultiAgentsPPOTrainer.fit() 传给 Tracking.log() 的完整字典；CSV 是其平铺版本。",
        "- `trainer_tracking_summary.json` 汇总所有 Tracking 数值项；`partial_summary.json` 汇总报告派生指标。",
        "- `trainer_tracking_summary.md` 将所有 Tracking 数值项按 mean/min/p50/p90/max/std 展开，便于直接审阅。",
        "",
        "## 机器可读附件",
        "",
        f"- `{artifact_dir.name}/per_step_metrics.jsonl`",
        f"- `{artifact_dir.name}/per_step_metrics.csv`",
        f"- `{artifact_dir.name}/trainer_tracking_summary.json`",
        f"- `{artifact_dir.name}/trainer_tracking_summary.md`",
        f"- `{artifact_dir.name}/partial_summary.json`",
    ]

    report_path = args.output.resolve() if args.output else artifact_dir / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
