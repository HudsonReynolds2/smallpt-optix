#!/usr/bin/env python3
"""
Re-parse an Nsight Compute CSV with the correct sum-vs-mean aggregation.

The original run_nsight.ps1 v3 summed every metric across ncu's per-invocation
rows. That's correct for counters (FFMA, FADD, FMUL, dram__bytes.sum) but
WRONG for percentages and ratios (sm__throughput, dram__throughput,
warps_active, thread_inst_executed_per_inst_executed.ratio etc), where the
right aggregation is the mean -- weighted by time when possible.

Usage:
    python fix_nsight_summary.py <metrics.csv> [<metrics.csv> ...]

For each input CSV, writes a corrected <name>_summary_v2.csv next to it and
prints a derived metrics block in the same style run_nsight.ps1 used.

This is a one-shot utility so you don't have to rerun ncu (which takes ~15s
under the deprecated launcher and blows up wall-time anyway). Future ncu
runs will use the corrected logic baked into run_nsight.ps1 v4.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from collections import defaultdict


def parse_metrics_csv(path: Path) -> dict:
    """Return {metric_name: [(time_ns, value), ...]}.

    Each entry collects every per-invocation reading for that metric, paired
    with the kernel time so we can weight the average if needed. We grab
    gpu__time_duration.sum specially since it's the per-invocation timing
    we use as the weight.
    """
    by_invocation: dict[str, dict[str, float]] = defaultdict(dict)
    metric_unit: dict[str, str] = {}

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = None
        # Column indices, looked up once from the header row.
        idx_id = idx_name = idx_unit = idx_value = -1
        for row in reader:
            if header is None:
                # csv.reader strips surrounding double-quotes, so the column
                # names appear without them. Old code looked for '"ID"'
                # which never matched and silently fell back to magic
                # numbers; that worked but was fragile.
                if row and row[0] == "ID":
                    header = row
                    idx_id    = header.index("ID")
                    idx_name  = header.index("Metric Name")
                    idx_unit  = header.index("Metric Unit")
                    idx_value = header.index("Metric Value")
                continue
            if not row or not row[0]:
                continue
            if max(idx_id, idx_name, idx_unit, idx_value) >= len(row):
                continue
            id_col       = row[idx_id]
            metric_name  = row[idx_name]
            metric_unit_v = row[idx_unit]
            metric_value = row[idx_value]

            if not id_col.isdigit():
                continue
            try:
                v = float(metric_value.replace(",", ""))
            except ValueError:
                continue
            by_invocation[id_col][metric_name] = v
            metric_unit[metric_name] = metric_unit_v

    return by_invocation, metric_unit


def aggregate(by_inv: dict, metric_name: str, mode: str = "auto", unit: str = "") -> float:
    """Aggregate a metric across invocations.

    mode='sum'   : add up (correct for counters, *.sum, bytes, instructions)
    mode='mean'  : arithmetic mean (correct for percentages, ratios)
    mode='wmean' : time-weighted mean (more accurate than 'mean' if the
                   per-invocation times differ; falls back to plain mean
                   if no time data is available)
    mode='auto'  : pick based on unit and metric name suffix
    """
    vals = [(inv.get("gpu__time_duration.sum", 0.0), inv.get(metric_name))
            for inv in by_inv.values()]
    vals = [(t, v) for t, v in vals if v is not None]
    if not vals:
        return 0.0

    if mode == "auto":
        # Counters end in .sum and have unit "" or "byte" or "inst"
        # Percentages have unit "%". Ratios have empty unit but end in .ratio
        if metric_name.endswith(".sum"):
            mode = "sum"
        elif unit == "%" or metric_name.endswith(".ratio"):
            mode = "wmean"
        elif unit in ("byte", "inst"):
            mode = "sum"
        else:
            mode = "wmean"

    if mode == "sum":
        return sum(v for _, v in vals)
    if mode == "mean":
        return sum(v for _, v in vals) / len(vals)
    if mode == "wmean":
        total_t = sum(t for t, _ in vals)
        if total_t > 0:
            return sum(t * v for t, v in vals) / total_t
        return sum(v for _, v in vals) / len(vals)
    raise ValueError(f"unknown mode {mode}")


def fix_one(path: Path) -> None:
    print(f"\n=== {path} ===")
    by_inv, units = parse_metrics_csv(path)

    if not by_inv:
        print(f"  no data parsed from {path}")
        return

    print(f"  invocations: {len(by_inv)}")
    print(f"  unique metrics: {len(units)}")

    # Counters (sum is correct)
    ffma  = aggregate(by_inv, "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "sum", "inst")
    fadd  = aggregate(by_inv, "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "sum", "inst")
    fmul  = aggregate(by_inv, "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "sum", "inst")
    bytes_dram = aggregate(by_inv, "dram__bytes.sum", "sum", "byte")
    time_ns    = aggregate(by_inv, "gpu__time_duration.sum", "sum", "ns")
    lts_total  = aggregate(by_inv, "lts__t_sectors.sum", "sum", "")
    lts_hit    = aggregate(by_inv, "lts__t_sectors_lookup_hit.sum", "sum", "")

    # Percentages and ratios (time-weighted mean is correct)
    sm_thr    = aggregate(by_inv, "sm__throughput.avg.pct_of_peak_sustained_elapsed", "wmean", "%")
    dram_thr  = aggregate(by_inv, "dram__throughput.avg.pct_of_peak_sustained_elapsed", "wmean", "%")
    occup     = aggregate(by_inv, "smsp__warps_active.avg.pct_of_peak_sustained_active", "wmean", "%")
    diverg    = aggregate(by_inv, "smsp__thread_inst_executed_per_inst_executed.ratio", "wmean", "")
    l1tex_thr = aggregate(by_inv, "l1tex__throughput.avg.pct_of_peak_sustained_active", "wmean", "%")
    xu_pct    = aggregate(by_inv, "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active", "wmean", "%")

    # Derived
    flops    = 2.0 * ffma + fadd + fmul
    ai       = flops / bytes_dram if bytes_dram > 0 else 0.0
    gflopsps = flops / (time_ns * 1e-9) / 1e9 if time_ns > 0 else 0.0
    gbps     = bytes_dram / (time_ns * 1e-9) / 1e9 if time_ns > 0 else 0.0
    l2_hit   = (lts_hit / lts_total * 100.0) if lts_total > 0 else 0.0

    print()
    print("  --- Per-invocation values (sanity check) ---")
    print("  invocation  time_ns       sm%    dram%   occ%   div     dram_bytes")
    for inv_id, inv in sorted(by_inv.items(), key=lambda x: int(x[0])):
        t   = inv.get("gpu__time_duration.sum", 0)
        sm  = inv.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        dr  = inv.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", 0)
        oc  = inv.get("smsp__warps_active.avg.pct_of_peak_sustained_active", 0)
        dv  = inv.get("smsp__thread_inst_executed_per_inst_executed.ratio", 0)
        b   = inv.get("dram__bytes.sum", 0)
        print(f"  #{inv_id:>2s}        {t:>11.0f}  {sm:>5.2f}  {dr:>5.2f}  {oc:>5.2f}  {dv:>5.2f}  {b:>14,.0f}")

    print()
    print("  --- Aggregated metrics (CORRECTED) ---")
    print(f"    FFMA       : {ffma:>20,.0f}  inst")
    print(f"    FADD       : {fadd:>20,.0f}  inst")
    print(f"    FMUL       : {fmul:>20,.0f}  inst")
    print(f"    FLOPs      : {flops:>20,.0f}  (2*FFMA + FADD + FMUL)")
    print(f"    DRAM bytes : {bytes_dram:>20,.0f}  bytes")
    print(f"    Time       : {time_ns:>20,.0f}  ns ({time_ns*1e-6:.2f} ms)")
    print()
    print(f"    SM%        : {sm_thr:>5.2f}%   (was wrong: showed {sm_thr*len(by_inv):.2f}%)")
    print(f"    DRAM%      : {dram_thr:>5.2f}%   (was wrong: showed {dram_thr*len(by_inv):.2f}%)")
    print(f"    L1TEX%     : {l1tex_thr:>5.2f}%")
    print(f"    XU%        : {xu_pct:>5.2f}%")
    print(f"    Occupancy  : {occup:>5.2f}%")
    print(f"    Divergence : {diverg:>5.2f}/32  ({diverg/32*100:.0f}% threads/warp inst)")
    print(f"    L2 hit     : {l2_hit:>5.2f}%  ({lts_hit:,.0f} of {lts_total:,.0f} sectors)")
    print()
    print(f"    Throughput : {gbps:>7.1f} GB/s")
    if flops > 0:
        print(f"    Arith Int  : {ai:>7.3f} FLOP/byte (SM-side only)")
        print(f"    GFLOP/s    : {gflopsps:>7.1f}")
    else:
        print(f"    Arith Int  : not computable (FFMA/FADD/FMUL = 0).")
        print(f"                 Ampere + OPTIX_FORCE_DEPRECATED_LAUNCHER does not")
        print(f"                 expose SASS-level FLOP counters for the launcher")
        print(f"                 wrapper. Use the back-of-envelope estimate from")
        print(f"                 plan v3 §6 instead, citing DRAM bytes ({bytes_dram/1e9:.1f} GB)")
        print(f"                 and {gbps:.0f} GB/s sustained.")

    # Write corrected summary CSV
    out_path = path.with_suffix("").with_name(path.stem.replace("_metrics", "") + "_summary_v2.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit", "aggregation"])
        w.writerow(["invocations", len(by_inv), "count", "n/a"])
        w.writerow(["ffma_count", f"{ffma:.0f}", "inst", "sum"])
        w.writerow(["fadd_count", f"{fadd:.0f}", "inst", "sum"])
        w.writerow(["fmul_count", f"{fmul:.0f}", "inst", "sum"])
        w.writerow(["flops_total", f"{flops:.0f}", "FLOPs", "derived"])
        w.writerow(["dram_bytes", f"{bytes_dram:.0f}", "bytes", "sum"])
        w.writerow(["time_ns", f"{time_ns:.0f}", "ns", "sum"])
        w.writerow(["throughput_gbps", f"{gbps:.2f}", "GB/s", "derived"])
        if flops > 0:
            w.writerow(["arithmetic_intensity", f"{ai:.4f}", "FLOP/byte", "derived"])
            w.writerow(["throughput_gflopsps", f"{gflopsps:.2f}", "GFLOP/s", "derived"])
        w.writerow(["sm_throughput_pct", f"{sm_thr:.2f}", "%", "weighted_mean"])
        w.writerow(["dram_throughput_pct", f"{dram_thr:.2f}", "%", "weighted_mean"])
        w.writerow(["l1tex_throughput_pct", f"{l1tex_thr:.2f}", "%", "weighted_mean"])
        w.writerow(["xu_pipe_pct", f"{xu_pct:.2f}", "%", "weighted_mean"])
        w.writerow(["warps_active_pct", f"{occup:.2f}", "%", "weighted_mean"])
        w.writerow(["divergence_threads_per_warp", f"{diverg:.3f}", "threads", "weighted_mean"])
        w.writerow(["l2_hit_pct", f"{l2_hit:.2f}", "%", "derived"])
    print(f"  Wrote: {out_path}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    for arg in sys.argv[1:]:
        p = Path(arg)
        if not p.exists():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            continue
        fix_one(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
