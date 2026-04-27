#!/usr/bin/env python3
"""
Tile-size sweep chart for the EC527 deck.

Reads tile_sweep.csv (from run_tile_sweep.ps1) and produces a single PNG
showing the two stories on one figure:

  Left axis:  peak GPU memory (MB) vs tile size  -- the WHY of tiling
  Right axis: throughput (Mrays/s) vs tile size  -- the cost (or lack of)

X-axis: tile size, log scale. The leftmost point is the smallest tile;
the rightmost point is "tile == image" (single full-image launch, the
phase 2 launch model). A vertical dashed line marks the chosen default
(--default-tile, default 512).

Failed runs (peak_mb == -1, set by run_tile_sweep.ps1 on OOM/exit-failure)
are shown as red Xs at the chart top with a "FAILED" annotation -- this
is the slide-relevant case for showing where tiling was *necessary*.

Usage:
    python build_tile_sweep_chart.py --csv path/to/tile_sweep.csv --out chart.png

Dependencies: matplotlib only (stdlib csv).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional


def die(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)


try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    die("matplotlib not installed. Run: pip install matplotlib")


def read_sweep(path: Path) -> list[dict]:
    """tile_sweep.csv schema:
        tile_size,resolution,spp,tile_w,tile_h,n_tiles,time_ms,mrays_sec,peak_mb,run_timestamp
    Returns rows as dicts, numeric fields coerced. Failed rows kept (peak_mb=-1)."""
    if not path.exists():
        die(f"CSV not found: {path}")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["tile_size"] = int(row["tile_size"])
                row["spp"]       = int(row["spp"])
                row["tile_w"]    = int(row["tile_w"])
                row["tile_h"]    = int(row["tile_h"])
                row["n_tiles"]   = int(row["n_tiles"])
                row["time_ms"]   = float(row["time_ms"])
                row["mrays_sec"] = float(row["mrays_sec"])
                row["peak_mb"]   = int(row["peak_mb"])
            except (KeyError, ValueError) as e:
                print(f"WARN: skipping malformed row: {row} ({e})", file=sys.stderr)
                continue
            rows.append(row)
    return rows


def x_label_for(row: dict, image_w: int, image_h: int) -> str:
    """X-axis tick label: tile size string. tile_size=0 -> 'no tile (image)'."""
    ts = row["tile_size"]
    if ts == 0:
        # Single full-image launch.
        return f"image\n({image_w}×{image_h})"
    return f"{ts}×{ts}"


def main() -> int:
    p = argparse.ArgumentParser(description="Tile-size sweep chart.")
    p.add_argument("--csv", type=Path, required=True,
                   help="tile_sweep.csv from run_tile_sweep.ps1")
    p.add_argument("--out", type=Path, required=True, help="output PNG")
    p.add_argument("--default-tile", type=int, default=512,
                   help="Tile size used by phase 3 default (drawn as a vertical guide). 0 to disable.")
    p.add_argument("--title", type=str, default=None,
                   help="Override the auto-generated title.")
    args = p.parse_args()

    rows = read_sweep(args.csv)
    if not rows:
        die("no rows in csv")

    # Resolution label from the first successful row's res string.
    res_str = rows[0]["resolution"]
    spp     = rows[0]["spp"]
    try:
        image_w, image_h = (int(x) for x in res_str.split("x"))
    except Exception:
        image_w, image_h = 0, 0

    # X-axis numeric: substitute tile_size=0 with image_w (so it sits on the
    # right of the log axis as the largest "tile"). This is honest because
    # a "no tile" launch genuinely IS one image-sized tile.
    def x_value(row: dict) -> int:
        return image_w if row["tile_size"] == 0 else row["tile_size"]

    # Sort by effective x value so the lines connect left-to-right cleanly.
    rows = sorted(rows, key=x_value)

    ok_rows   = [r for r in rows if r["peak_mb"] >= 0]
    fail_rows = [r for r in rows if r["peak_mb"] <  0]

    if not ok_rows:
        die("all rows are marked failed; nothing to plot")

    xs   = [x_value(r)         for r in ok_rows]
    mems = [r["peak_mb"]        for r in ok_rows]
    mrs  = [r["mrays_sec"]      for r in ok_rows]
    tcnt = [r["n_tiles"]        for r in ok_rows]

    fig, ax_mem = plt.subplots(figsize=(12, 7))
    ax_mr = ax_mem.twinx()

    color_mem = "#c44536"  # phase-2-ish red, "the problem"
    color_mr  = "#2e8b57"  # phase-3 green, "the win"

    line_mem, = ax_mem.plot(xs, mems, marker="s", linewidth=2.0, markersize=10,
                            color=color_mem, label="Peak GPU memory (MB)", zorder=3)
    line_mr,  = ax_mr.plot(xs, mrs, marker="o", linewidth=2.0, markersize=10,
                           color=color_mr, label="Throughput (Mrays/s)", zorder=3)

    # Annotate every point. Memory in MB, throughput in Mrays/s, tile count parenthetical.
    # For the last point (rightmost), shift label left to avoid clipping at chart edge.
    last_x = xs[-1]
    for i, (x, m, mr, n) in enumerate(zip(xs, mems, mrs, tcnt)):
        is_last = (x == last_x)
        # Memory label: below the marker
        ax_mem.annotate(f"{m:,} MB", (x, m), textcoords="offset points",
                        xytext=(0, -18), ha="center", fontsize=9, color=color_mem)
        # Throughput label: above for most points; nudge left + above for the last
        tile_str = f"tile{'s' if n != 1 else ''}"
        label = f"{mr:.0f}  ({n} {tile_str})"
        if is_last:
            ax_mr.annotate(label, (x, mr), textcoords="offset points",
                           xytext=(-8, 14), ha="right", fontsize=9, color=color_mr)
        else:
            ax_mr.annotate(label, (x, mr), textcoords="offset points",
                           xytext=(0, 12), ha="center", fontsize=9, color=color_mr)

    # Failed rows: mark at top of memory axis.
    if fail_rows:
        # Use the current memory ymax extrapolated up.
        y_top = max(mems) * 1.4 if mems else 1.0
        for r in fail_rows:
            x = x_value(r)
            ax_mem.scatter([x], [y_top], marker="x", s=200,
                           color="#990000", zorder=5)
            ax_mem.annotate("FAILED", (x, y_top), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=10,
                            fontweight="bold", color="#990000")

    # Default-tile guide line.
    if args.default_tile and args.default_tile > 0:
        ax_mem.axvline(x=args.default_tile, linestyle=":", color="#555555",
                       linewidth=1.2, zorder=1)
        ax_mem.text(args.default_tile, ax_mem.get_ylim()[1] * 0.02,
                    f"  default ({args.default_tile})",
                    fontsize=9, color="#555555", rotation=0)

    ax_mem.set_xscale("log", base=2)
    ax_mem.set_xlabel("Tile size (px, log₂ scale; rightmost = single full-image launch = phase 2 model)")
    ax_mem.set_ylabel("Peak GPU memory (MB)", color=color_mem)
    ax_mem.tick_params(axis="y", labelcolor=color_mem)
    ax_mem.grid(True, alpha=0.3)

    ax_mr.set_ylabel("Throughput (Mrays / second)", color=color_mr)
    ax_mr.tick_params(axis="y", labelcolor=color_mr)

    # Custom x ticks at the data points only (log scale otherwise picks
    # awkward defaults like 64, 256, 1024 ignoring our actual tile sizes).
    ax_mem.set_xticks(xs)
    ax_mem.set_xticklabels(
        [x_label_for(r, image_w, image_h) for r in ok_rows],
        fontsize=9
    )
    ax_mem.minorticks_off()

    # Y-limits with breathing room for the annotations.
    if mems:
        mem_min, mem_max = min(mems), max(mems)
        if fail_rows:
            mem_max = max(mem_max, mem_max * 1.4)
        ax_mem.set_ylim(0, mem_max * 1.25)
    if mrs:
        mr_min, mr_max = min(mrs), max(mrs)
        ax_mr.set_ylim(max(0, mr_min - (mr_max - mr_min) * 0.5),
                       mr_max + (mr_max - mr_min) * 0.5 + 50)

    title = args.title or (
        f"Tile-size sweep at {res_str}, {spp} spp\n"
        f"smaller tiles = bounded memory; throughput holds across the sweep"
    )
    ax_mem.set_title(title)

    # Combined legend.
    lines = [line_mem, line_mr]
    labels = [l.get_label() for l in lines]
    ax_mem.legend(lines, labels, loc="upper left")

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
