#!/usr/bin/env python3
"""Create dependency-light PNG plots and summaries from KV monitor CSV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "policy_1": "#2563eb",
    "policy_2": "#dc2626",
    "policy_1/home": "#1d4ed8",
    "policy_1/guest": "#60a5fa",
    "policy_2/home": "#b91c1c",
    "policy_2/guest": "#f87171",
}


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_line_chart(data: pd.DataFrame, output: Path, series_col: str, title: str) -> None:
    width, height = 1600, 900
    left, right, top, bottom = 120, 55, 100, 105
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), title, fill="#111827", font=font(32, True))

    active = data[data["kv_util"] > 0]
    if active.empty:
        raise RuntimeError("No non-zero KV samples found")
    x_min = max(float(data["elapsed_s"].min()), float(active["elapsed_s"].min()) - 5)
    x_max = min(float(data["elapsed_s"].max()), float(active["elapsed_s"].max()) + 5)
    view = data[(data["elapsed_s"] >= x_min) & (data["elapsed_s"] <= x_max)].copy()
    view["bucket"] = ((view["elapsed_s"] - x_min) / 0.5).astype(int)
    grouped = view.groupby(["bucket", series_col], as_index=False).agg(
        elapsed_s=("elapsed_s", "mean"), kv_util=("kv_util", "max")
    )
    y_max = max(float(grouped["kv_util"].max()) * 100 * 1.12, 0.01)

    def px(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1e-9) * plot_w

    def py(y_percent: float) -> float:
        return top + plot_h - y_percent / y_max * plot_h

    for i in range(6):
        y = y_max * i / 5
        yy = py(y)
        draw.line((left, yy, left + plot_w, yy), fill="#e5e7eb", width=1)
        draw.text((18, yy - 10), f"{y:.3f}%", fill="#4b5563", font=font(18))
    for i in range(9):
        x = x_min + (x_max - x_min) * i / 8
        xx = px(x)
        draw.line((xx, top, xx, top + plot_h), fill="#f3f4f6", width=1)
        draw.text((xx - 28, top + plot_h + 18), f"{x - x_min:.0f}s", fill="#4b5563", font=font(18))

    # The CSV records the last completed step. A change to N marks step N's end.
    boundaries = (
        view[view["global_step"] > 0]
        .groupby("global_step", as_index=False)["elapsed_s"].min()
    )
    for row in boundaries.itertuples():
        xx = px(float(row.elapsed_s))
        if left <= xx <= left + plot_w:
            draw.line((xx, top, xx, top + plot_h), fill="#9ca3af", width=2)
            draw.text((xx + 4, top + 5), f"S{int(row.global_step)}", fill="#6b7280", font=font(15))

    names = list(dict.fromkeys(grouped[series_col].tolist()))
    for name in names:
        part = grouped[grouped[series_col] == name].sort_values("elapsed_s")
        points = [(px(float(row.elapsed_s)), py(float(row.kv_util) * 100)) for row in part.itertuples()]
        if len(points) >= 2:
            draw.line(points, fill=COLORS.get(str(name), "#111827"), width=4)

    draw.line((left, top, left, top + plot_h), fill="#111827", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=2)
    draw.text((width // 2 - 130, height - 45), "Time since first KV activity", fill="#111827", font=font(20))
    draw.text((15, 65), "KV cache utilization", fill="#111827", font=font(18))
    legend_x = width - 430
    for i, name in enumerate(names):
        yy = 36 + i * 30
        draw.line((legend_x, yy + 10, legend_x + 42, yy + 10), fill=COLORS.get(str(name), "#111827"), width=5)
        draw.text((legend_x + 52, yy), str(name), fill="#111827", font=font(18))
    image.save(output)


def draw_step_bars(summary: pd.DataFrame, output: Path, steps: int) -> None:
    width, height = 1500, 820
    left, right, top, bottom = 110, 50, 100, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 28), "Peak KV cache utilization by training step", fill="#111827", font=font(32, True))
    y_max = max(float(summary["kv_peak"].max()) * 100 * 1.15, 0.01)
    for i in range(6):
        y = y_max * i / 5
        yy = top + plot_h - y / y_max * plot_h
        draw.line((left, yy, left + plot_w, yy), fill="#e5e7eb", width=1)
        draw.text((12, yy - 10), f"{y:.3f}%", fill="#4b5563", font=font(18))
    slot = plot_w / steps
    bar_w = slot * 0.30
    for step in range(1, steps + 1):
        for offset, policy in [(-0.55, "policy_1"), (0.05, "policy_2")]:
            rows = summary[(summary["training_step"] == step) & (summary["policy"] == policy)]
            value = float(rows["kv_peak"].iloc[0]) * 100 if not rows.empty else 0.0
            x0 = left + (step - 1) * slot + slot / 2 + offset * bar_w
            x1 = x0 + bar_w
            y0 = top + plot_h - value / y_max * plot_h
            draw.rectangle((x0, y0, x1, top + plot_h), fill=COLORS[policy])
        draw.text((left + (step - 1) * slot + slot / 2 - 8, top + plot_h + 18), str(step), fill="#111827", font=font(18))
    draw.line((left, top, left, top + plot_h), fill="#111827", width=2)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="#111827", width=2)
    draw.text((width // 2 - 55, height - 48), "Step", fill="#111827", font=font(21))
    draw.line((width - 360, 55, width - 320, 55), fill=COLORS["policy_1"], width=8)
    draw.text((width - 308, 43), "policy_1", fill="#111827", font=font(18))
    draw.line((width - 190, 55, width - 150, 55), fill=COLORS["policy_2"], width=8)
    draw.text((width - 138, 43), "policy_2", fill="#111827", font=font(18))
    image.save(output)


def main() -> int:
    args = args_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.csv)
    numeric = ["elapsed_s", "global_step", "scrape_ok", "kv_util", "requests_running", "requests_waiting"]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    valid = data[(data["scrape_ok"] == 1) & data["policy"].isin(["policy_1", "policy_2"])].copy()
    valid["pool"] = valid["policy"] + "/" + valid["role"]
    valid["training_step"] = (valid["global_step"] + 1).clip(upper=args.steps).astype(int)

    step_summary = valid[valid["global_step"] < args.steps].groupby(
        ["training_step", "policy"], as_index=False
    ).agg(
        kv_peak=("kv_util", "max"),
        kv_mean=("kv_util", "mean"),
        nonzero_samples=("kv_util", lambda values: int((values > 0).sum())),
        request_peak=("requests_running", "max"),
        samples=("kv_util", "size"),
    )
    step_summary.to_csv(args.output_dir / "kv_step_summary.csv", index=False)

    replica_summary = valid.groupby(
        ["policy", "role", "replica", "address"], as_index=False
    ).agg(
        kv_peak=("kv_util", "max"),
        kv_mean=("kv_util", "mean"),
        nonzero_samples=("kv_util", lambda values: int((values > 0).sum())),
        request_peak=("requests_running", "max"),
        samples=("kv_util", "size"),
    )
    replica_summary.to_csv(args.output_dir / "kv_replica_summary.csv", index=False)

    draw_line_chart(valid, args.output_dir / "kv_curve_by_policy.png", "policy", "KV cache utilization during 10-step dynamic borrowing")
    draw_line_chart(valid, args.output_dir / "kv_curve_by_pool.png", "pool", "KV cache utilization by home/guest pool")
    draw_step_bars(step_summary, args.output_dir / "kv_peak_by_step.png", args.steps)

    total = len(data)
    ok = int((data["scrape_ok"] == 1).sum())
    with (args.output_dir / "kv_collection_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"rows={total}\n")
        handle.write(f"successful_scrapes={ok}\n")
        handle.write(f"success_rate={ok / total if total else 0:.6f}\n")
        handle.write(f"duration_s={data['elapsed_s'].max() - data['elapsed_s'].min():.3f}\n")
        handle.write(f"kv_peak={valid['kv_util'].max():.9f}\n")
        handle.write(f"nonzero_samples={(valid['kv_util'] > 0).sum()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
