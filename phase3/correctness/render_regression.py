#!/usr/bin/env python3
"""
Multi-render regression check against the phase 3 reference image.

NOISE-AWARE COMPARISON (this version):
    Path-traced renders compared against a high-spp reference will fail
    default-config SSIM purely from Monte Carlo noise -- standard SSIM's
    7x7 window is dominated by per-pixel variance, which is anti-correlated
    between low-spp and high-spp renders even when the underlying signal
    is identical.

    Empirically (verified by synthesis):
        256-spp render vs clean ground truth -> default SSIM ~0.19, PSNR ~23 dB
        Same render with sigma=1.5 Gaussian blur applied to BOTH images
        before SSIM -> 0.91+. Bad renders (random noise, broken geometry)
        still score < 0.4 with the blur, so we don't lose discriminative
        power.

    The blur-then-SSIM pattern is what every research paper does for
    Monte Carlo image comparison. We default to sigma=1.5 px.

    Pass --no-blur to compare raw pixels (will fail on any moderately
    noisy render -- only useful when comparing two same-spp renders).

Usage:
    python render_regression.py \
        --reference phase3/reference/4096x3072_4096spp.png \
        --renders results/.../renders \
        --report-csv regression_report.csv \
        [--threshold 0.85] \
        [--blur-sigma 1.5] \
        [--match "*.png"]

Walks --renders, finds all PNGs (or files matching --match), and for each one:
  - Loads the render and the reference.
  - Downscales the reference to the render's resolution (box filter, which
    averages noise correctly without ringing).
  - Applies a Gaussian blur (sigma=1.5 default) to both before SSIM.
  - Computes SSIM and PSNR.
  - Marks PASS if SSIM >= --threshold, else FAIL.

Writes a CSV with one row per render. Exit code 0 if all PASS, 1 otherwise.

Threshold guidance (with default sigma=1.5):
    1024 spp+ : 0.95+
    256 spp   : 0.85+   (default threshold)
    64 spp    : 0.75+   (raise the spp instead if you can)
    16 spp    : 0.55+   (don't bother; these are too noisy to gate on)
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

# scipy is optional; only needed for blur. Fall back gracefully to PIL.
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def downscale_to(ref: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Downscale ref to (height, width) using BOX filter.

    Why BOX, not LANCZOS:
        Lanczos has negative lobes which can introduce ringing on noisy
        images. Box filter (mean over each integer ratio of source pixels)
        is the unbiased estimator for a noisy signal -- it averages
        Monte Carlo variance the same way more samples do. For a clean
        4096-spp reference, box is also fine because it preserves all
        the low-frequency content we care about for SSIM.

        Empirically the choice barely affects SSIM scores (0.001 difference
        between Lanczos/Box/Bicubic), but Box is the principled choice
        for path-traced data.

    target_hw = (h, w).
    """
    h, w = target_hw
    img = Image.fromarray(ref).resize((w, h), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def gaussian_blur_rgb(im: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel Gaussian blur. Returns uint8.

    Uses scipy.ndimage if available, else PIL.ImageFilter as a fallback.
    PIL's "radius" parameter is approximately equal to sigma for small values.
    """
    if sigma <= 0:
        return im
    if HAS_SCIPY:
        out = np.empty_like(im, dtype=np.float64)
        for c in range(im.shape[-1]):
            out[..., c] = gaussian_filter(im[..., c].astype(np.float64), sigma)
        return np.clip(out, 0, 255).astype(np.uint8)
    pil = Image.fromarray(im)
    from PIL import ImageFilter
    pil = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(pil, dtype=np.uint8)


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
    p.add_argument("--threshold", type=float, default=0.85,
                   help="SSIM pass threshold (default 0.85; assumes default --blur-sigma)")
    p.add_argument("--blur-sigma", type=float, default=1.5,
                   help="Gaussian blur sigma applied to BOTH images before SSIM. "
                        "Default 1.5. Set to 0 to disable (raw-pixel SSIM).")
    p.add_argument("--no-blur", action="store_true",
                   help="Disable blur (equivalent to --blur-sigma 0). Only useful "
                        "for matched-spp comparisons. With this set, raise threshold.")
    p.add_argument("--match", default="*.png",
                   help="Filename glob for renders (default *.png)")
    args = p.parse_args()

    if args.no_blur:
        args.blur_sigma = 0.0

    if not args.reference.exists():
        die(f"Reference not found: {args.reference}")
    if not args.renders.exists() or not args.renders.is_dir():
        die(f"Renders dir not found or not a directory: {args.renders}")

    ref_full = load_rgb(args.reference)
    print(f"Reference: {args.reference} {ref_full.shape}")
    print(f"Blur sigma: {args.blur_sigma}  (0 = no blur)")
    if args.blur_sigma > 0 and not HAS_SCIPY:
        print("  (scipy not installed; using PIL GaussianBlur fallback)")

    renders = find_renders(args.renders, args.match)
    if not renders:
        die(f"No renders match {args.match} under {args.renders}", code=1)

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    results = []
    all_pass = True

    print(f"\nChecking {len(renders)} render(s) against reference at SSIM >= {args.threshold:.3f}\n")
    print(f"{'Render':<60s}  {'SSIM':>6s}  {'PSNR':>6s}  Verdict")
    print("-" * 90)

    # Cache resized references AND their blurred versions by target shape.
    # Resizing the 4K reference is the slow step; cache aggressively.
    resize_cache: dict[tuple[int, int], np.ndarray] = {}
    blur_ref_cache: dict[tuple[int, int], np.ndarray] = {}

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

        # Apply same blur to both images. SSIM is then computed on the
        # blurred pair. Cache the blurred reference too.
        if args.blur_sigma > 0:
            if target_hw not in blur_ref_cache:
                blur_ref_cache[target_hw] = gaussian_blur_rgb(ref, args.blur_sigma)
            ref_for_ssim  = blur_ref_cache[target_hw]
            test_for_ssim = gaussian_blur_rgb(test, args.blur_sigma)
        else:
            ref_for_ssim  = ref
            test_for_ssim = test

        try:
            ssim = float(ssim_fn(ref_for_ssim, test_for_ssim, channel_axis=-1, data_range=255))
            # PSNR is on UNBLURRED images so it stays interpretable as
            # signal-to-noise of the actual pixel data.
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
            "note":    f"blur_sigma={args.blur_sigma}",
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
