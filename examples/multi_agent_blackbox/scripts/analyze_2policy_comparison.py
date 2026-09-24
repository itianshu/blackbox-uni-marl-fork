"""Build step-aligned observability and a dynamic/static two-policy report."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


POLICIES = ("policy_1", "policy_2")
COUNTERS = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "prefix_cache_queries_total",
    "prefix_cache_hits_total",
    "requests_success_total",
    "requests_failure_total",
    "preemptions_total",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def mean(values) -> float | None:
    values = [float(value) for value in values if finite(value)]
    return statistics.mean(values) if values else None


def percentile(values, q: float) -> float | None:
    values = sorted(float(value) for value in values if finite(value))
    if not values:
        return None
    return values[max(0, math.ceil(q * len(values)) - 1)]


def describe(values) -> dict[str, float | int]:
    values = [float(value) for value in values if finite(value)]
    if not values:
        return {}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "p50": statistics.median(values),
        "p90": percentile(values, 0.9),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}/{key}" if prefix else str(key)
            result.update(flatten(item, child))
    else:
        result[prefix] = value
    return result


def active_servers(sample, trace, trace_times):
    servers = [server for server in sample.get("servers", []) if "metrics" in server]
    if not trace:
        return [server for server in servers if int(server.get("rank", 0)) < 10000]
    index = bisect.bisect_right(trace_times, sample["time_unix_s"]) - 1
    if index < 0:
        return [server for server in servers if int(server.get("rank", 0)) < 10000]
    state = trace[index]
    lends = state.get("active_lends")
    if lends is not None:
        sleeping = {address for lend in lends for address in lend.get("home_servers", [])}
        guests = {address for lend in lends for address in lend.get("guest_servers", [])}
        return [
            server for server in servers
            if (int(server.get("rank", 0)) < 10000 and server.get("address") not in sleeping)
            or server.get("address") in guests
        ]
    addresses = {
        policy: set(state.get("policies", {}).get(policy, {}).get("replica_inflight", {}))
        for policy in POLICIES
    }
    return [server for server in servers if server.get("address") in addresses.get(server.get("policy"), set())]


def counter_events(serving):
    previous = {}
    events = []
    for sample in sorted(serving, key=lambda row: row["time_unix_s"]):
        for server in sample.get("servers", []):
            metrics = server.get("metrics")
            if not metrics:
                continue
            actor_id = server.get("actor_id", server.get("address"))
            old = previous.get(actor_id)
            previous[actor_id] = {key: metrics.get(key) for key in COUNTERS}
            if old is None:
                continue
            deltas = {}
            for key in COUNTERS:
                current, prior = metrics.get(key), old.get(key)
                if not finite(current) or not finite(prior):
                    continue
                deltas[key] = float(current - prior if current >= prior else current)
            events.append({
                "time_unix_s": sample["time_unix_s"],
                "policy": server.get("policy"),
                "deltas": deltas,
            })
    return events


def direct_step_metrics(metrics):
    result = {
        "wall_clock_s": metrics.get("perf/time_per_step"),
        "trainer_throughput": metrics.get("perf/throughput"),
        "training_tokens": metrics.get("perf/total_num_tokens"),
        "trajectory_count": metrics.get("training/trajectory_count"),
        "mas_rollout_count": metrics.get("training/mas_rollout_count"),
        "rollout_wait_s": metrics.get("timing_s/gen"),
        "advantage_s": metrics.get("timing_s/adv"),
        "staleness_mean": metrics.get("training/off_policy/trajectory_staleness/mean"),
        "staleness_max": metrics.get("training/off_policy/trajectory_staleness/max"),
        "staleness_worst_mean": metrics.get("training/off_policy/trajectory_staleness_worst/mean"),
        "rollout_training_probs_corr": metrics.get("rollout_corr/rollout_training_probs_pearson_corr",
            metrics.get("rollout_corr/pearson_corr")),
        "rollout_training_log_ppl_abs_diff": metrics.get("rollout_corr/log_ppl_abs_diff"),
        "rollout_training_ppl_ratio": metrics.get("rollout_corr/ppl_ratio"),
        "rollout_training_kl": metrics.get("rollout_corr/kl"),
    }
    phase_names = ("old_log_prob", "update_actor", "update_weights")
    serial = critical = 0.0
    for phase in phase_names:
        values = [metrics.get(f"timing_s/{policy}/{phase}") for policy in POLICIES]
        values = [float(value) for value in values if finite(value)]
        if values:
            serial += sum(values)
            critical += max(values)
    result["policy_phase_serial_s"] = serial
    result["policy_phase_critical_path_s"] = critical
    result["policy_parallel_overlap_saved_s"] = serial - critical
    result["policy_parallel_efficiency"] = serial / (len(POLICIES) * critical) if critical else None
    step_s = result["wall_clock_s"]
    gen_s = result["rollout_wait_s"]
    result["phase_accounted_fraction"] = (
        (float(gen_s or 0) + critical) / float(step_s)
        if finite(step_s) and step_s > 0 else None
    )
    result["policies"] = {}
    for policy in POLICIES:
        result["policies"][policy] = {
            "trajectory_count": metrics.get(f"{policy}/trajectory_count"),
            "mfu": metrics.get(f"{policy}/perf/mfu/actor"),
            "actor_memory_allocated_gib": metrics.get(f"{policy}/actor/perf/max_memory_allocated_gb"),
            "actor_memory_reserved_gib": metrics.get(f"{policy}/actor/perf/max_memory_reserved_gb"),
            "staleness_mean": metrics.get(f"{policy}/off_policy/trajectory_staleness/mean"),
            "staleness_max": metrics.get(f"{policy}/off_policy/trajectory_staleness/max"),
            "rollout_actor_probs_corr": metrics.get(f"{policy}/training/rollout_actor_probs_pearson_corr"),
            "rollout_probs_diff_mean": metrics.get(f"{policy}/training/rollout_probs_diff_mean"),
            "rollout_probs_diff_max": metrics.get(f"{policy}/training/rollout_probs_diff_max"),
            "reward_mean": metrics.get(f"{policy}/critic/score/mean"),
            "entropy": metrics.get(f"{policy}/actor/entropy"),
            "pg_loss": metrics.get(f"{policy}/actor/pg_loss"),
            "ppo_kl": metrics.get(f"{policy}/actor/ppo_kl"),
            "grad_norm": metrics.get(f"{policy}/actor/grad_norm"),
            "old_log_prob_s": metrics.get(f"timing_s/{policy}/old_log_prob"),
            "update_actor_s": metrics.get(f"timing_s/{policy}/update_actor"),
            "update_weights_s": metrics.get(f"timing_s/{policy}/update_weights"),
        }
    result["mfu_mean"] = mean(result["policies"][policy]["mfu"] for policy in POLICIES)
    return result


def summarize_step_window(start, end, gpu, serving, events, trace, trace_times):
    duration = max(end - start, 1e-9)
    result = {}
    gpu_rows = [row for row in gpu if start < row.get("time_unix_s", 0) <= end and "gpus" in row]
    utilization, memory_used, memory_total, power = [], [], [], []
    for row in gpu_rows:
        for device in row["gpus"]:
            for destination, key in [
                (utilization, "utilization.gpu"),
                (memory_used, "memory.used"),
                (memory_total, "memory.total"),
                (power, "power.draw"),
            ]:
                try:
                    destination.append(float(device[key]))
                except (KeyError, ValueError, TypeError):
                    pass
    result["gpu"] = {
        "sample_count": len(utilization),
        "effective_utilization": mean(utilization),
        "utilization_p90": percentile(utilization, 0.9),
        "utilization_max": max(utilization) if utilization else None,
        "bubble_fraction_lt_10pct": mean(float(value < 10) for value in utilization),
        "bubble_fraction_lt_30pct": mean(float(value < 30) for value in utilization),
        "memory_used_mib_mean": mean(memory_used),
        "memory_used_mib_p90": percentile(memory_used, 0.9),
        "memory_used_mib_max": max(memory_used) if memory_used else None,
        "memory_utilization_fraction_mean": mean(
            used / total for used, total in zip(memory_used, memory_total) if total > 0
        ),
        "power_w_mean": mean(power),
    }

    window_samples = [row for row in serving if start < row.get("time_unix_s", 0) <= end]
    policy_values = {policy: {"kv": [], "running": [], "waiting": [], "reasons": {}} for policy in POLICIES}
    reason_available = False
    for sample in window_samples:
        grouped = {policy: [] for policy in POLICIES}
        for server in active_servers(sample, trace, trace_times):
            if server.get("policy") in grouped:
                grouped[server["policy"]].append(server)
        for policy, servers in grouped.items():
            if not servers:
                continue
            metrics = [server["metrics"] for server in servers]
            policy_values[policy]["kv"].extend(
                item.get("gpu_cache_usage_max") for item in metrics if finite(item.get("gpu_cache_usage_max"))
            )
            policy_values[policy]["running"].append(sum(item.get("requests_running", 0) for item in metrics))
            policy_values[policy]["waiting"].append(sum(item.get("requests_waiting", 0) for item in metrics))
            for server in servers:
                reason_available |= bool(server.get("waiting_by_reason_available"))
                for reason, value in server.get("waiting_by_reason", {}).items():
                    policy_values[policy]["reasons"].setdefault(reason, []).append(value)

    event_window = [event for event in events if start < event["time_unix_s"] <= end]
    result["rollout"] = {"waiting_by_reason_available": reason_available, "policies": {}}
    total_prompt = total_generation = total_success = 0.0
    for policy in POLICIES:
        deltas = {
            key: sum(event["deltas"].get(key, 0) for event in event_window if event["policy"] == policy)
            for key in COUNTERS
        }
        total_prompt += deltas["prompt_tokens_total"]
        total_generation += deltas["generation_tokens_total"]
        total_success += deltas["requests_success_total"]
        queries, hits = deltas["prefix_cache_queries_total"], deltas["prefix_cache_hits_total"]
        values = policy_values[policy]
        result["rollout"]["policies"][policy] = {
            "kv_mean": mean(values["kv"]),
            "kv_p90": percentile(values["kv"], 0.9),
            "kv_max": max(values["kv"]) if values["kv"] else None,
            "requests_running_mean": mean(values["running"]),
            "requests_waiting_mean": mean(values["waiting"]),
            "requests_waiting_max": max(values["waiting"]) if values["waiting"] else None,
            "waiting_by_reason_mean": {reason: mean(reason_values) for reason, reason_values in values["reasons"].items()},
            "prompt_tokens": deltas["prompt_tokens_total"],
            "generation_tokens": deltas["generation_tokens_total"],
            "generation_tokens_per_s": deltas["generation_tokens_total"] / duration,
            "total_tokens_per_s": (deltas["prompt_tokens_total"] + deltas["generation_tokens_total"]) / duration,
            "prefix_cache_queries": queries,
            "prefix_cache_hits": hits,
            "prefix_cache_hit_rate": hits / queries if queries else None,
            "completed_requests": deltas["requests_success_total"],
            "failed_requests": deltas["requests_failure_total"],
            "preemptions": deltas["preemptions_total"],
        }
    result["rollout"].update({
        "prompt_tokens": total_prompt,
        "generation_tokens": total_generation,
        "generation_tokens_per_s": total_generation / duration,
        "overall_tokens_per_s": (total_prompt + total_generation) / duration,
        "completed_requests": total_success,
    })

    trace_window = [row for row in trace if start < row.get("time_unix_s", 0) <= end]
    # HDFS append contention can drop an individual event while later samples
    # still durably expose the committed topology. Reconcile explicit events
    # with active-lend set transitions so borrow/return totals describe the
    # topology that actually served traffic. Keep raw start/completion counts
    # separately below: mismatches remain visible as observability gaps.
    previous_active = None
    for row in trace:
        if row.get("time_unix_s", 0) > start:
            break
        if "active_lends" in row:
            previous_active = {
                lend.get("lend_id"): lend
                for lend in row["active_lends"]
                if lend.get("lend_id") is not None
            }
    inferred_borrow_ids, inferred_return_ids = set(), set()
    for row in trace_window:
        if "active_lends" not in row:
            continue
        current_active = {
            lend.get("lend_id"): lend
            for lend in row["active_lends"]
            if lend.get("lend_id") is not None
        }
        if previous_active is not None:
            inferred_borrow_ids.update(set(current_active) - set(previous_active))
            inferred_return_ids.update(set(previous_active) - set(current_active))
        previous_active = current_active
    observed_borrow_ids = {
        row.get("detail", {}).get("lend_id")
        for row in trace_window if row.get("kind") == "borrow"
    } - {None}
    observed_return_ids = {
        row.get("detail", {}).get("lend_id")
        for row in trace_window if row.get("kind") in {"return", "early_return"}
    } - {None}
    reconciled_borrow_ids = observed_borrow_ids | inferred_borrow_ids
    reconciled_return_ids = observed_return_ids | inferred_return_ids
    borrow_durations = [
        row.get("detail", {}).get("execution_s")
        for row in trace_window if row.get("kind") == "borrow_complete"
    ]
    return_durations = [
        row.get("detail", {}).get("execution_s")
        for row in trace_window if row.get("kind") == "return_complete"
    ]
    clone_durations = [
        row.get("detail", {}).get("seconds")
        for row in trace_window if row.get("kind") == "replica_clone"
    ]
    result["dynamic_scheduling"] = {
        "borrow_starts": sum(row.get("kind") == "borrow_start" for row in trace_window),
        "borrow_events": len(reconciled_borrow_ids),
        "borrow_events_explicit": len(observed_borrow_ids),
        "borrow_events_inferred_from_topology": len(
            inferred_borrow_ids - observed_borrow_ids
        ),
        "borrow_completions": sum(row.get("kind") == "borrow_complete" for row in trace_window),
        "return_events": len(reconciled_return_ids),
        "return_events_explicit": len(observed_return_ids),
        "return_events_inferred_from_topology": len(
            inferred_return_ids - observed_return_ids
        ),
        "return_starts": sum(row.get("kind") == "return_start" for row in trace_window),
        "return_completions": sum(row.get("kind") == "return_complete" for row in trace_window),
        "early_return_requests": sum(
            row.get("kind") == "early_return_requested" for row in trace_window
        ),
        "borrow_failures": sum(row.get("kind") == "borrow_failed" for row in trace_window),
        "return_failures": sum(row.get("kind") == "return_failed" for row in trace_window),
        "borrow_execution_s_mean": mean(borrow_durations),
        "borrow_execution_s_p90": percentile(borrow_durations, 0.9),
        "return_execution_s_mean": mean(return_durations),
        "return_execution_s_p90": percentile(return_durations, 0.9),
        "replica_clone_events": sum(row.get("kind") == "replica_clone" for row in trace_window),
        "replica_clone_s_mean": mean(clone_durations),
        "replica_clone_s_p90": percentile(clone_durations, 0.9),
        "active_lends_mean": mean(len(row.get("active_lends", [])) for row in trace_window if "active_lends" in row),
        "active_lends_max": max([len(row.get("active_lends", [])) for row in trace_window if "active_lends" in row] or [0]),
        "serving_cards_mean": {
            policy: mean(row.get("policies", {}).get(policy, {}).get("serving_cards") for row in trace_window)
            for policy in POLICIES
        },
    }
    return result


def build_run(directory: Path, manifest_info: dict[str, Any]):
    steps = sorted(read_jsonl(directory / "steps.jsonl"), key=lambda row: row["step"])
    gpu = read_jsonl(directory / "gpu.jsonl")
    serving = read_jsonl(directory / "serving.jsonl")
    trace = sorted(
        [row for row in read_jsonl(directory / "checkpoints/dynamic_inference.jsonl") if "time_unix_s" in row],
        key=lambda row: row["time_unix_s"],
    )
    trace_times = [row["time_unix_s"] for row in trace]
    events = counter_events(serving)
    per_step = []
    previous_end = manifest_info.get("started_unix_s")
    for row in steps:
        end = float(row["observed_unix_s"])
        tracked_metrics = row.get("metrics", {})
        duration = tracked_metrics.get("perf/time_per_step")
        start = end - float(duration) if finite(duration) else previous_end
        if start is None:
            start = end
        # A reconstructed step duration is generally more accurate than the
        # polling gap.  Never overlap adjacent windows when clocks disagree.
        if previous_end is not None:
            start = max(float(start), float(previous_end))
        result = {
            "step": int(row["step"]),
            "window_start_unix_s": start,
            "window_end_unix_s": end,
            # Preserve every value passed by MultiAgentsPPOTrainer.fit() to
            # Tracking.log().  Selected fields below provide stable report
            # aliases, while this namespace keeps new/upstream metrics from
            # silently disappearing from the experiment artifacts.
            "trainer_tracking": tracked_metrics,
            **direct_step_metrics(tracked_metrics),
        }
        result.update(summarize_step_window(start, end, gpu, serving, events, trace, trace_times))
        per_step.append(result)
        previous_end = end

    flat_rows = [flatten(row) for row in per_step]
    numeric_keys = sorted({key for row in flat_rows for key, value in row.items() if finite(value)})
    summary = {key: describe(row.get(key) for row in flat_rows) for key in numeric_keys}
    summary = {key: value for key, value in summary.items() if value}
    summary["completed_steps"] = len(per_step)
    if manifest_info.get("started_unix_s") and manifest_info.get("finished_unix_s"):
        summary["total_wall_s"] = manifest_info["finished_unix_s"] - manifest_info["started_unix_s"]
    return per_step, summary


def write_run(directory: Path, rows):
    (directory / "per_step_metrics.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    flat = [flatten(row) for row in rows]
    keys = sorted({key for row in flat for key in row})
    with (directory / "per_step_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)

    tracked = [row.get("trainer_tracking", {}) for row in rows]
    tracked_keys = sorted({key for metrics in tracked for key, value in metrics.items() if finite(value)})
    tracked_summary = {
        key: describe(metrics.get(key) for metrics in tracked)
        for key in tracked_keys
    }
    (directory / "trainer_tracking_summary.json").write_text(
        json.dumps(tracked_summary, ensure_ascii=False, indent=2)
    )


def fmt(value, percent=False):
    if value is None:
        return "N/A"
    if percent:
        value *= 100
    return f"{value:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    manifest = json.loads((directory / "manifest.json").read_text())
    all_rows, summaries = {}, {}
    for mode in ("dynamic", "static"):
        rows, summary = build_run(directory / mode, manifest.get("runs", {}).get(mode, {}))
        write_run(directory / mode, rows)
        all_rows[mode], summaries[mode] = rows, summary

    output = {"manifest": manifest, "runs": summaries}
    (directory / "comparison_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2))

    def avg(mode, key):
        return summaries[mode].get(key, {}).get("mean")

    metrics = [
        ("Step wall clock (s)", "wall_clock_s", False),
        ("Rollout wait (s)", "rollout_wait_s", False),
        ("Trainer throughput", "trainer_throughput", False),
        ("Rollout generation tokens/s", "rollout/generation_tokens_per_s", False),
        ("Rollout overall tokens/s", "rollout/overall_tokens_per_s", False),
        ("Trajectories/step", "trajectory_count", False),
        ("MAS rollouts/step", "mas_rollout_count", False),
        ("Overall GPU utilization (%)", "gpu/effective_utilization", False),
        ("GPU bubble fraction (<10%)", "gpu/bubble_fraction_lt_10pct", True),
        ("GPU memory mean (MiB)", "gpu/memory_used_mib_mean", False),
        ("GPU memory peak (MiB)", "gpu/memory_used_mib_max", False),
        ("Mean training MFU", "mfu_mean", True),
        ("Actor allocated GiB p1", "policies/policy_1/actor_memory_allocated_gib", False),
        ("Actor allocated GiB p2", "policies/policy_2/actor_memory_allocated_gib", False),
        ("Actor reserved GiB p1", "policies/policy_1/actor_memory_reserved_gib", False),
        ("Actor reserved GiB p2", "policies/policy_2/actor_memory_reserved_gib", False),
        ("Trajectory staleness mean", "staleness_mean", False),
        ("Rollout/train probability corr p1", "policies/policy_1/rollout_actor_probs_corr", False),
        ("Rollout/train probability corr p2", "policies/policy_2/rollout_actor_probs_corr", False),
        ("Rollout/train log-PPL abs diff", "rollout_training_log_ppl_abs_diff", False),
        ("Rollout/train PPL ratio", "rollout_training_ppl_ratio", False),
        ("Rollout/train KL", "rollout_training_kl", False),
        ("Reward mean p1", "policies/policy_1/reward_mean", False),
        ("Reward mean p2", "policies/policy_2/reward_mean", False),
        ("Entropy p1", "policies/policy_1/entropy", False),
        ("Entropy p2", "policies/policy_2/entropy", False),
        ("PPO KL p1", "policies/policy_1/ppo_kl", False),
        ("PPO KL p2", "policies/policy_2/ppo_kl", False),
        ("Policy parallel overlap saved (s)", "policy_parallel_overlap_saved_s", False),
        ("Policy parallel efficiency", "policy_parallel_efficiency", True),
        ("Phase accounted fraction", "phase_accounted_fraction", True),
        ("Prefix cache hit rate p1", "rollout/policies/policy_1/prefix_cache_hit_rate", True),
        ("Prefix cache hit rate p2", "rollout/policies/policy_2/prefix_cache_hit_rate", True),
        ("Request waiting mean p1", "rollout/policies/policy_1/requests_waiting_mean", False),
        ("Request waiting mean p2", "rollout/policies/policy_2/requests_waiting_mean", False),
        ("KV mean p1", "rollout/policies/policy_1/kv_mean", True),
        ("KV mean p2", "rollout/policies/policy_2/kv_mean", True),
        ("Borrow events/step", "dynamic_scheduling/borrow_events", False),
        ("Borrow events inferred from topology/step", "dynamic_scheduling/borrow_events_inferred_from_topology", False),
        ("Borrow transaction time (s)", "dynamic_scheduling/borrow_execution_s_mean", False),
        ("Return events/step", "dynamic_scheduling/return_events", False),
        ("Return events inferred from topology/step", "dynamic_scheduling/return_events_inferred_from_topology", False),
        ("Return transaction time (s)", "dynamic_scheduling/return_execution_s_mean", False),
        ("Replica weight-clone time (s)", "dynamic_scheduling/replica_clone_s_mean", False),
        ("Borrow failures/step", "dynamic_scheduling/borrow_failures", False),
        ("Return failures/step", "dynamic_scheduling/return_failures", False),
        ("Active lends mean", "dynamic_scheduling/active_lends_mean", False),
    ]
    configured_steps = manifest.get("steps", "N/A")
    lines = [
        "# 两 Policy 动态推理调度对比实验",
        "",
        f"动态调度先运行，静态基线后运行；两组均从相同 Qwen3-1.7B 初始模型启动并运行 {configured_steps} step。",
        "训练侧 policy_1 使用 8 卡 FSDP8，policy_2 使用 4 卡 FSDP4，两者均为 Ulysses SP2。",
        "推理侧基线分配为 policy_1 12 个、policy_2 8 个 TP1/DP1 单卡 replica；动态组允许双向最多借 4 个 replica。",
        "Agent 轮数为 7–10，首尾固定 agent1，中间按 agent1/agent2=1:1 路由，两者使用相同 token 预算。",
        "",
        "## 全部 step 平均",
        "",
        "| 指标 | Dynamic | Static | Dynamic 相对变化 |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percent in metrics:
        dynamic, static = avg("dynamic", key), avg("static", key)
        delta = (dynamic / static - 1) * 100 if finite(dynamic) and finite(static) and static != 0 else None
        lines.append(f"| {label} | {fmt(dynamic, percent)} | {fmt(static, percent)} | {fmt(delta)}% |")

    lines += [
        "",
        "## 运行级总计",
        "",
        "| 指标 | Dynamic | Static |",
        "|---|---:|---:|",
    ]
    for label, getter in [
        ("Completed steps", lambda mode: len(all_rows[mode])),
        ("End-to-end wall clock (s)", lambda mode: summaries[mode].get("total_wall_s")),
        ("Borrow transaction starts", lambda mode: sum(row["dynamic_scheduling"]["borrow_starts"] for row in all_rows[mode])),
        ("Borrow events", lambda mode: sum(row["dynamic_scheduling"]["borrow_events"] for row in all_rows[mode])),
        ("Borrow events inferred from topology", lambda mode: sum(row["dynamic_scheduling"]["borrow_events_inferred_from_topology"] for row in all_rows[mode])),
        ("Borrow transaction completions", lambda mode: sum(row["dynamic_scheduling"]["borrow_completions"] for row in all_rows[mode])),
        ("Return transaction starts", lambda mode: sum(row["dynamic_scheduling"]["return_starts"] for row in all_rows[mode])),
        ("Return events", lambda mode: sum(row["dynamic_scheduling"]["return_events"] for row in all_rows[mode])),
        ("Return events inferred from topology", lambda mode: sum(row["dynamic_scheduling"]["return_events_inferred_from_topology"] for row in all_rows[mode])),
        ("Return transaction completions", lambda mode: sum(row["dynamic_scheduling"]["return_completions"] for row in all_rows[mode])),
        ("Early-return requests", lambda mode: sum(row["dynamic_scheduling"]["early_return_requests"] for row in all_rows[mode])),
        ("Borrow failures", lambda mode: sum(row["dynamic_scheduling"]["borrow_failures"] for row in all_rows[mode])),
        ("Return failures", lambda mode: sum(row["dynamic_scheduling"]["return_failures"] for row in all_rows[mode])),
        ("Generation tokens", lambda mode: sum(row["rollout"]["generation_tokens"] for row in all_rows[mode])),
        ("Completed requests", lambda mode: sum(row["rollout"]["completed_requests"] for row in all_rows[mode])),
    ]:
        lines.append(f"| {label} | {fmt(getter('dynamic'))} | {fmt(getter('static'))} |")

    lines += ["", "## 每 step 核心指标", ""]
    for mode in ("dynamic", "static"):
        lines += [f"### {mode.capitalize()}", "", "| Step | Wall s | Gen tok/s | GPU util % | Bubble <10% | MFU % | Traj | Staleness |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in all_rows[mode]:
            lines.append(
                f"| {row['step']} | {fmt(row.get('wall_clock_s'))} | "
                f"{fmt(row['rollout'].get('generation_tokens_per_s'))} | "
                f"{fmt(row['gpu'].get('effective_utilization'))} | "
                f"{fmt(row['gpu'].get('bubble_fraction_lt_10pct'), True)} | "
                f"{fmt(row.get('mfu_mean'), True)} | {fmt(row.get('trajectory_count'))} | "
                f"{fmt(row.get('staleness_mean'))} |"
            )
        lines += [
            "",
            "| Step | KV p1 % | KV p2 % | Waiting p1 | Waiting p2 | Prefix hit p1 % | Prefix hit p2 % | Borrow | Return |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in all_rows[mode]:
            rollout = row["rollout"]["policies"]
            scheduling = row["dynamic_scheduling"]
            lines.append(
                f"| {row['step']} | {fmt(rollout['policy_1'].get('kv_mean'), True)} | "
                f"{fmt(rollout['policy_2'].get('kv_mean'), True)} | "
                f"{fmt(rollout['policy_1'].get('requests_waiting_mean'))} | "
                f"{fmt(rollout['policy_2'].get('requests_waiting_mean'))} | "
                f"{fmt(rollout['policy_1'].get('prefix_cache_hit_rate'), True)} | "
                f"{fmt(rollout['policy_2'].get('prefix_cache_hit_rate'), True)} | "
                f"{fmt(scheduling.get('borrow_events'))} | "
                f"{fmt(scheduling.get('return_events'))} |"
            )
        lines += [
            "",
            "| Step | Train/rollout corr p1 | Train/rollout corr p2 | Prob diff p1 | Prob diff p2 | Parallel overlap s | GPU mem peak MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in all_rows[mode]:
            policies = row["policies"]
            lines.append(
                f"| {row['step']} | {fmt(policies['policy_1'].get('rollout_actor_probs_corr'))} | "
                f"{fmt(policies['policy_2'].get('rollout_actor_probs_corr'))} | "
                f"{fmt(policies['policy_1'].get('rollout_probs_diff_mean'))} | "
                f"{fmt(policies['policy_2'].get('rollout_probs_diff_mean'))} | "
                f"{fmt(row.get('policy_parallel_overlap_saved_s'))} | "
                f"{fmt(row['gpu'].get('memory_used_mib_max'))} |"
            )
        lines.append("")

    reason_available = any(
        row.get("rollout", {}).get("waiting_by_reason_available")
        for rows in all_rows.values() for row in rows
    )
    lines += [
        "## 指标口径与可用性",
        "",
        "- Prefix cache hit rate 使用每个 step 内 hits/queries 的 counter 增量，而不是累计 gauge。",
        "- 本次实验配置 enable_prefix_caching=false；若 vLLM 仍导出少量 prefix-cache hit counter，报告会如实记录，但不将其解释为开启 prefix cache 后的加速收益。",
        "- 借还总数由显式事件与 active_lends 拓扑变化取并集；由拓扑补全的数量单独列出，用于识别共享文件追加期间的事件日志缺口。",
        "- Rollout 吞吐来自所有活跃 vLLM replica 的 Prometheus token counter 增量。",
        "- GPU 空泡定义为 15 秒 NVML 样本中 GPU utilization <10% 的比例；有效利用率为同窗口平均 GPU utilization。",
        "- Policy overlap 是两个 policy 的 old-log-prob/update/update-weights 并行临界路径代理，不冒充 CUDA kernel overlap。",
        "- MFU、staleness、训推概率相关性和显存 allocated/reserved 直接来自训练 step metrics。",
        "- MultiAgentsPPOTrainer.fit() 传给 Tracking.log() 的完整字典保存在每条记录的 trainer_tracking 下；所有数值项的统计汇总见 trainer_tracking_summary.json。",
        f"- vLLM request waiting by reason：{'已按标签采集' if reason_available else '当前 vLLM 未导出带 reason 标签的 metric，报告为 N/A；总 waiting 仍已采集'}。",
        "- 全量逐 step 嵌套数据见 dynamic/static/per_step_metrics.jsonl；平铺数据见对应 CSV。",
    ]
    (directory / "comparison_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
