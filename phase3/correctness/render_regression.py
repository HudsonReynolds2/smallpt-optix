#!/usr/bin/env python3
"""
Multi-render regression check against the phase 3 reference image.

Usage:
    python render_regression.py \
        --reference phase3/reference/4096x3072_4096spp.png \
        --renders results/.../renders \
        --report-csv regression_report.csv \
        [--threshold 0.96] \
        [--match "*.png"]

Walks --renders, finds all PNGs (or files matching --match), and for each one:
  - Loads the render and the reference.
  - Downscales the reference to the render's resolution (Lanczos).
  - Computes SSIM and PSNR.
  - Marks PASS if SSIM >= --threshold, else FAIL.

Writes a CSV with one row per render. Exit code 0 if all PASS, 1 otherwise.

Why threshold defaults to 0.96 (not 0.98 like the per-image tool):
    SPP differences alone introduce noise variation that drops SSIM below
    0.98 even when the renders are correct. The reference is 4096 spp; phase 3
    benchmark configs are 64-1024 spp. 0.96 is a "noise but no structural
    artifact" floor. For matched-spp comparisons (1024 spp test vs 1024 spp
    ref slice) you'd raise this to 0.98.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from pathlib import Path


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


try:
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
except ImportError as e:
    die(f"Missing Python deps: {e}\nRun: pip install pillow scikit-image numpy")


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def downscale_to(ref: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Lanczos downscale of ref to (height, width). target_hw = (h, w)."""
    h, w = target_hw
    img = Image.fromarray(ref).resize((w, h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def find_renders(root: Path, pattern: str) -> list[Path]:
    """Recursive glob. Sorted for stable output."""
    return sorted([p for p in root.rglob("*") if p.is_file() and fnmatch.fnmatch(p.name, pattern)])


def main() -> int:
    p = argparse.ArgumentParser(description="Regression check against reference.")
    p.add_argument("--reference", type=Path, required=True,
                   help="Ground-truth render (e.g. phase3/reference/4096x3072_4096spp.png)")
    p.add_argument("--renders", type=Path, required=True,
                   help="Directory containing renders to check (recursive)")
    p.add_argument("--report-csv", type=Path, required=True,
                   help="Output CSV with per-render results")
    p.add_argument("--threshold", type=float, default=0.96,
                   help="SSIM pass threshold (default 0.96 for varied-spp comparisons)")
    p.add_argument("--match", default="*.png",
                   help="Filename glob for renders (default *.png)")
    args = p.parse_args()

    if not args.reference.exists():
        die(f"Reference not found: {args.reference}")
    if not args.renders.exists() or not args.renders.is_dir():
        die(f"Renders dir not found or not a directory: {args.renders}")

    ref_full = load_rgb(args.reference)
    print(f"Reference: {args.reference} {ref_full.shape}")

    renders = find_renders(args.renders, args.match)
    if not renders:
        die(f"No renders match {args.match} under {args.renders}", code=1)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    all_pass = True

    print(f"\nChecking {len(renders)} render(s) against reference at SSIM >= {args.threshold:.3f}\n")
    print(f"{'Render':<60s}  {'SSIM':>6s}  {'PSNR':>6s}  Verdict")
    print("-" * 90)

    # Cache resized references by target shape so we don't redo Lanczos
    # for every same-resolution render.
    resize_cache: dict[tuple[int, int], np.ndarray] = {}

    for r in renders:
        try:
            test = load_rgb(r)
        except Exception as e:
            print(f"{str(r.name):<60s}  -- error loading: {e}")
            results.append({
                "render": str(r), "shape": "error", "ssim": "", "psnr": "",
                "verdict": "ERROR", "note": str(e),
            })
            all_pass = False
            continue

        target_hw = (test.shape[0], test.shape[1])
        if target_hw not in resize_cache:
            resize_cache[target_hw] = downscale_to(ref_full, target_hw)
        ref = resize_cache[target_hw]

        try:
            ssim = float(ssim_fn(ref, test, channel_axis=-1, data_range=255))
            psnr = float(psnr_fn(ref, test, data_range=255))
        except Exception as e:
            print(f"{str(r.name):<60s}  -- metric error: {e}")
            results.append({
                "render": str(r), "shape": str(test.shape), "ssim": "",
                "psnr": "", "verdict": "ERROR", "note": str(e),
            })
            all_pass = False
            continue

        verdict = "PASS" if ssim >= args.threshold else "FAIL"
        if verdict == "FAIL":
            all_pass = False

        # Try to format relative path; fall back to absolute if it can't be
        # made relative to the cwd (different drive on Windows, for example).
        try:
            rel = r.relative_to(Path.cwd())
        except ValueError:
            rel = r
        print(f"{str(rel):<60s}  {ssim:.4f}  {psnr:5.2f}  {verdict}")

        results.append({
            "render":  str(r),
            "shape":   f"{test.shape[1]}x{test.shape[0]}",
            "ssim":    f"{ssim:.4f}",
            "psnr":    f"{psnr:.2f}",
            "verdict": verdict,
            "note":    "",
        })

    # Write CSV
    with open(args.report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["render", "shape", "ssim", "psnr", "verdict", "note"])
        w.writeheader()
        for row in results:
            w.writerow(row)

    print()
    print(f"Report: {args.report_csv}")
    fails = sum(1 for r in results if r["verdict"] != "PASS")
    print(f"Summary: {len(results) - fails} pass / {fails} fail / {len(results)} total")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
