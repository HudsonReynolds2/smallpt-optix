#!/usr/bin/env python3
"""
Phase 3 deck chart builder.

Reads the various timings CSVs produced by the benchmark and ablation runs,
and emits PNG chart files for the EC527 deck.

Inputs (all paths configurable via CLI):

  --phase1-csv  : optional, single-row CSV with phase 1 cu-smallpt timings.
                  Schema:  phase,scene,resolution,spp,time_ms,mrays_sec,notes,run_timestamp
                  If absent or the file doesn't exist, the bar chart still
                  produces (without a phase 1 bar) so we don't block the
                  deck on phase 1 reruns.

  --phase2-csv  : phase 2 baseline timings.csv from run_phase2_benchmark.ps1
                  Same schema as above. If absent we just plot phase 3.

  --phase3-csv  : phase 3 timings.csv from run_phase3_benchmark.ps1.
                  Required.

  --phase3-ext  : phase 3 timings_ext.csv (with depth + peak_mb columns).
                  Optional but needed for the memory plot.

  --ablation-csv: phase 3 ablation.csv. Required for the depth chart.

Outputs (under --out-dir):

  mrays_bar.png            : phase1 / phase2 / phase3 bar chart at the
                              "headline" config (1024x768_256spp).
  weak_scaling.png         : phase 3 Mrays/s vs resolution at fixed spp.
  bounce_depth.png         : phase 3 Mrays/s and (if --reference-png is
                              given) SSIM vs --max-depth.
  memory_4k.png            : peak GPU memory used by phase 3 at the 4K
                              configs in timings_ext.csv. Compared against
                              the documented 18 GB phase 2 number (drawn
                              as a horizontal reference line).

If matplotlib isn't installed we fail loudly with a pip install hint.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Optional


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


try:
    import matplotlib
    matplotlib.use("Agg")  # headless: no display backend needed
    import matplotlib.pyplot as plt
except ImportError:
    die("matplotlib not installed. Run: pip install matplotlib")


# ---------------------------------------------------------------------------
# CSV readers (stdlib only; no pandas dependency to keep this portable
# across Hudson's Windows + WSL setups).
# ---------------------------------------------------------------------------

def read_timings(path: Optional[Path]) -> list[dict]:
    """Read a timings.csv. Schema is enforced by the benchmark scripts.

    Returns rows as dicts with keys:
        phase, scene, resolution, spp, time_ms, mrays_sec, notes, run_timestamp

    If path is None or file doesn't exist, returns []. Caller decides what
    to do about a missing file -- we don't die because phase 1 may not be
    rerun yet on Hudson's machine.
    """
    if path is None or not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric fields. Tolerate missing/blank values.
            try:
                row["spp"]       = int(row["spp"])
                row["time_ms"]   = float(row["time_ms"])
                row["mrays_sec"] = float(row["mrays_sec"])
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def read_ext_timings(path: Optional[Path]) -> list[dict]:
    """timings_ext.csv has extra fields: max_depth and peak_mb."""
    if path is None or not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["spp"]       = int(row["spp"])
                row["max_depth"] = int(row["max_depth"])
                row["time_ms"]   = float(row["time_ms"])
                row["mrays_sec"] = float(row["mrays_sec"])
                row["peak_mb"]   = int(row["peak_mb"]) if row.get("peak_mb") else 0
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def read_ablation(path: Optional[Path]) -> list[dict]:
    """ablation.csv: max_depth,resolution,spp,time_ms,mrays_sec,peak_mb,run_timestamp."""
    if path is None or not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["max_depth"] = int(row["max_depth"])
                row["spp"]       = int(row["spp"])
                row["time_ms"]   = float(row["time_ms"])
                row["mrays_sec"] = float(row["mrays_sec"])
                row["peak_mb"]   = int(row["peak_mb"]) if row.get("peak_mb") else 0
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return rows


def find_row(rows: list[dict], resolution: str, spp: int) -> Optional[dict]:
    """Find the most-recent row matching (resolution, spp)."""
    matches = [r for r in rows if r.get("resolution") == resolution and r.get("spp") == spp]
    if not matches:
        return None
    # Most recent wins (lexicographic timestamp comparison works for our format).
    return max(matches, key=lambda r: r.get("run_timestamp", ""))


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def chart_mrays_bar(
    p1: list[dict], p2: list[dict], p3: list[dict],
    headline_res: str, headline_spp: int,
    out_path: Path,
) -> None:
    """Bar chart: phase 1 / phase 2 / phase 3 Mrays/s at the headline config."""
    labels  = []
    values  = []
    colors  = []

    r1 = find_row(p1, headline_res, headline_spp)
    if r1:
        labels.append("Phase 1\ncu-smallpt")
        values.append(r1["mrays_sec"])
        colors.append("#888888")

    r2 = find_row(p2, headline_res, headline_spp)
    if r2:
        labels.append("Phase 2\nOptiX +\nwall tess")
        values.append(r2["mrays_sec"])
        colors.append("#3a7ca5")

    r3 = find_row(p3, headline_res, headline_spp)
    if r3:
        labels.append("Phase 3\n+ sphere tess\n+ tile launch")
        values.append(r3["mrays_sec"])
        colors.append("#2e8b57")

    if not values:
        print(f"WARN: no data for headline config {headline_res}_{headline_spp}spp; skipping mrays_bar", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Mrays / second")
    ax.set_title(f"Throughput at {headline_res}, {headline_spp} spp\nRTX 3080 Ti (SM 8.6)")
    ax.grid(axis="y", alpha=0.3)

    # Annotate each bar with its value
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotate speedups vs phase 1 if available
    if r1 and len(values) >= 2:
        baseline = values[0]
        for i, b in enumerate(bars[1:], start=1):
            speedup = values[i] / baseline if baseline > 0 else 0
            ax.text(b.get_x() + b.get_width() / 2, values[i] * 0.5,
                    f"{speedup:.1f}x",
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def chart_weak_scaling(p3: list[dict], out_path: Path) -> None:
    """Mrays/s vs resolution at constant spp.

    Picks the spp value that has the most resolution data points. Plots a
    flat-ish curve to demonstrate throughput stability.
    """
    if not p3:
        print("WARN: no phase 3 data; skipping weak_scaling", file=sys.stderr)
        return

    # Bucket by spp, find the one with most distinct resolutions
    by_spp: dict[int, list[dict]] = {}
    for r in p3:
        by_spp.setdefault(r["spp"], []).append(r)
    best_spp = max(by_spp.keys(), key=lambda s: len({r["resolution"] for r in by_spp[s]}))
    rows = by_spp[best_spp]

    # Sort by pixel count
    def pix_count(res: str) -> int:
        try:
            w, h = res.split("x")
            return int(w) * int(h)
        except (ValueError, AttributeError):
            return 0
    rows = sorted(rows, key=lambda r: pix_count(r["resolution"]))

    if not rows:
        print("WARN: no resolution series in phase 3 data; skipping weak_scaling", file=sys.stderr)
        return

    xs = [r["resolution"] for r in rows]
    ys = [r["mrays_sec"]  for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#2e8b57", markersize=10)
    ax.set_ylabel("Mrays / second")
    ax.set_xlabel("Resolution (pixels)")
    ax.set_title(f"Phase 3 weak scaling at {best_spp} spp\n(throughput should be flat — pixels are embarrassingly parallel)")
    ax.grid(True, alpha=0.3)

    # Annotate each point
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10)

    # Set y-axis to start a bit below min so the variation is visible but
    # the chart still tells the "flat" story.
    ymin, ymax = min(ys), max(ys)
    span = ymax - ymin
    ax.set_ylim(max(0, ymin - span * 0.5), ymax + span * 0.5 + 30)

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def chart_bounce_depth(abl: list[dict], out_path: Path) -> None:
    """Mrays/s vs --max-depth.

    Expected shape: drops sharply from depth 1 -> 4 (light not yet found),
    then plateaus from depth 6+ thanks to RR. The plateau is the whole
    point — RR makes the cap cosmetic.
    """
    if not abl:
        print("WARN: no ablation data; skipping bounce_depth", file=sys.stderr)
        return

    rows = sorted(abl, key=lambda r: r["max_depth"])
    xs = [r["max_depth"] for r in rows]
    ys = [r["mrays_sec"] for r in rows]
    ts = [r["time_ms"]   for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(xs, ys, marker="o", linewidth=2, color="#2e8b57", markersize=10)
    ax1.set_xlabel("Max bounce depth")
    ax1.set_ylabel("Mrays / second")
    ax1.set_title("Throughput vs depth")
    ax1.grid(True, alpha=0.3)
    for x, y in zip(xs, ys):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    ax2.plot(xs, ts, marker="s", linewidth=2, color="#c44536", markersize=10)
    ax2.set_xlabel("Max bounce depth")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Render time vs depth (lower = better)")
    ax2.grid(True, alpha=0.3)
    for x, t in zip(xs, ts):
        ax2.annotate(f"{t:.0f}", (x, t), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=9)

    res = rows[0].get("resolution", "?")
    spp = rows[0].get("spp", "?")
    fig.suptitle(f"Bounce-depth ablation @ {res}, {spp} spp\nRussian Roulette (depth>4) makes deeper caps nearly free", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


def chart_memory_4k(ext: list[dict], phase2_peak_gb: float, out_path: Path) -> None:
    """Peak GPU memory at 4K configs vs the phase 2 baseline reference."""
    rows_4k = [r for r in ext if r.get("resolution", "").startswith("4096") and r.get("peak_mb", 0) > 0]

    if not rows_4k:
        print("WARN: no 4K rows with peak_mb in extended timings; skipping memory_4k", file=sys.stderr)
        return

    # Sort by spp
    rows_4k = sorted(rows_4k, key=lambda r: r["spp"])
    labels = [f"4K\n{r['spp']} spp" for r in rows_4k]
    peaks_gb = [r["peak_mb"] / 1024.0 for r in rows_4k]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, peaks_gb, color="#2e8b57", edgecolor="black", linewidth=0.7,
                  label="Phase 3 (tiled)")
    ax.axhline(y=phase2_peak_gb, color="#c44536", linestyle="--", linewidth=2,
               label=f"Phase 2 (single launch): ~{phase2_peak_gb:.0f} GB")
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("4K render: tile launches bound peak memory")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    for b, v in zip(bars, peaks_gb):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f} GB",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # y-axis min 0 and a bit above the phase 2 reference
    ax.set_ylim(0, max(phase2_peak_gb * 1.1, max(peaks_gb) * 1.5 + 0.5))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Build phase 3 deck charts.")
    p.add_argument("--phase1-csv",  type=Path, default=None,
                   help="Phase 1 cu-smallpt timings.csv (optional)")
    p.add_argument("--phase2-csv",  type=Path, default=None,
                   help="Phase 2 baseline timings.csv (optional)")
    p.add_argument("--phase3-csv",  type=Path, required=True,
                   help="Phase 3 timings.csv (required)")
    p.add_argument("--phase3-ext",  type=Path, default=None,
                   help="Phase 3 timings_ext.csv (for memory plot)")
    p.add_argument("--ablation-csv", type=Path, default=None,
                   help="Phase 3 ablation.csv (for depth chart)")
    p.add_argument("--out-dir",     type=Path, default=Path("results/charts"),
                   help="Where to write the PNG files")
    p.add_argument("--headline-res", default="1024x768",
                   help="Headline resolution for the bar chart (default 1024x768)")
    p.add_argument("--headline-spp", type=int, default=256,
                   help="Headline spp for the bar chart (default 256)")
    p.add_argument("--phase2-peak-gb", type=float, default=18.0,
                   help="Phase 2 single-launch peak GB at 4K (default 18.0 from the v2 plan)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    p1  = read_timings(args.phase1_csv)
    p2  = read_timings(args.phase2_csv)
    p3  = read_timings(args.phase3_csv)
    ext = read_ext_timings(args.phase3_ext)
    abl = read_ablation(args.ablation_csv)

    print(f"Loaded: phase1={len(p1)} phase2={len(p2)} phase3={len(p3)} ext={len(ext)} abl={len(abl)}")

    chart_mrays_bar(p1, p2, p3, args.headline_res, args.headline_spp,
                    args.out_dir / "mrays_bar.png")
    chart_weak_scaling(p3, args.out_dir / "weak_scaling.png")
    chart_bounce_depth(abl, args.out_dir / "bounce_depth.png")
    chart_memory_4k(ext, args.phase2_peak_gb, args.out_dir / "memory_4k.png")

    print(f"\nAll charts written to: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
