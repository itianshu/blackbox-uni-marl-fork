#!/usr/bin/env python3
"""Sample vLLM Prometheus metrics discovered in an E2E training log."""

from __future__ import annotations

import argparse
import ast
import csv
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


MANAGER_RE = re.compile(r"LLMServerManager: (\[[^\n]+\])")
STEP_RE = re.compile(r"training/global_step:(\d+)")
METRIC_RE = re.compile(
    r"^vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|kv_cache_usage_ratio|"
    r"num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total)"
    r"(?:\{[^\n]*\})?\s+([0-9.eE+-]+)",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--post-run-seconds", type=float, default=10.0)
    parser.add_argument("--max-seconds", type=float, default=7200.0)
    return parser.parse_args()


def read_state(log_path: Path) -> tuple[list[list[str]], int]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return [], 0
    groups = []
    for raw in MANAGER_RE.findall(text):
        try:
            addresses = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            continue
        if isinstance(addresses, list) and all(isinstance(item, str) for item in addresses):
            groups.append(addresses)
    steps = [int(value) for value in STEP_RE.findall(text)]
    return groups, max(steps, default=0)


def scrape(opener, address: str) -> dict[str, float] | None:
    try:
        with opener.open(f"http://{address}/metrics", timeout=0.5) as response:
            text = response.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    values: dict[str, list[float]] = {}
    for name, raw_value in METRIC_RE.findall(text):
        values.setdefault(name, []).append(float(raw_value))
    kv = []
    for name in ("kv_cache_usage_perc", "gpu_cache_usage_perc", "kv_cache_usage_ratio"):
        kv.extend(values.get(name, []))
    return {
        "kv_util": max(kv, default=0.0),
        "requests_running": sum(values.get("num_requests_running", [])),
        "requests_waiting": sum(values.get("num_requests_waiting", [])),
        "prompt_tokens_total": sum(values.get("prompt_tokens_total", [])),
        "generation_tokens_total": sum(values.get("generation_tokens_total", [])),
    }


def group_metadata(group_index: int) -> tuple[str, str]:
    mapping = {
        0: ("policy_1", "home"),
        1: ("policy_1", "guest"),
        2: ("policy_2", "home"),
        3: ("policy_2", "guest"),
    }
    return mapping.get(group_index, ("unknown", f"group_{group_index}"))


def main() -> int:
    args = parse_args()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    completed_at: float | None = None
    known_addresses: set[str] = set()
    fieldnames = [
        "timestamp", "elapsed_s", "global_step", "group", "policy", "role", "replica",
        "address", "scrape_ok", "kv_util", "requests_running", "requests_waiting",
        "prompt_tokens_total", "generation_tokens_total",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=16) as pool:
            while time.monotonic() - start < args.max_seconds:
                groups, global_step = read_state(args.log)
                endpoints = []
                for group_index, addresses in enumerate(groups):
                    policy, role = group_metadata(group_index)
                    for replica, address in enumerate(addresses):
                        endpoints.append((group_index, policy, role, replica, address))
                        if address not in known_addresses:
                            known_addresses.add(address)
                            print(f"discovered group={group_index} {policy}/{role}/{replica} {address}", flush=True)
                futures = {
                    pool.submit(scrape, opener, address): metadata
                    for metadata in endpoints
                    for address in [metadata[-1]]
                }
                now = time.time()
                elapsed = time.monotonic() - start
                for future, (group_index, policy, role, replica, address) in futures.items():
                    metrics = future.result()
                    row = {
                        "timestamp": f"{now:.6f}",
                        "elapsed_s": f"{elapsed:.3f}",
                        "global_step": global_step,
                        "group": group_index,
                        "policy": policy,
                        "role": role,
                        "replica": replica,
                        "address": address,
                        "scrape_ok": int(metrics is not None),
                    }
                    if metrics is not None:
                        row.update(metrics)
                    writer.writerow(row)
                handle.flush()
                if global_step >= args.steps:
                    completed_at = completed_at or time.monotonic()
                    if time.monotonic() - completed_at >= args.post_run_seconds:
                        return 0
                time.sleep(args.interval)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
