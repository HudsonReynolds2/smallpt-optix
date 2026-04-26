#!/usr/bin/env python3
"""
Single-pair SSIM/PSNR comparison.

Usage:
    python ssim_compare.py <reference.png_or_ppm> <test.png_or_ppm> [--threshold 0.98]

Returns exit code 0 if SSIM >= threshold, else 1. Prints SSIM, PSNR, mean-abs-error
to stdout. Used both as a standalone tool and as a building block by render_regression.py.

If the two images differ in size, the larger one is downscaled to the smaller's
size with a Lanczos filter (matters because the 4K reference render is the
"ground truth" we compare 1024x768 phase 3 outputs against).

Dependencies: pillow + scikit-image + numpy. All present in standard data-science
environments. If any are missing, prints a clear pip install hint.
"""

from __future__ import annotations

import argparse
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


def load_image_rgb(path: Path) -> np.ndarray:
    """Load .png or .ppm into a uint8 HxWx3 numpy array."""
    if not path.exists():
        die(f"File not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def match_size(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """If shapes differ, downscale the larger to the smaller. Lanczos for quality."""
    if ref.shape == test.shape:
        return ref, test

    rh, rw = ref.shape[:2]
    th, tw = test.shape[:2]

    if rh * rw > th * tw:
        # ref is bigger - downscale it
        target = (tw, th)
        new = Image.fromarray(ref).resize(target, Image.LANCZOS)
        return np.asarray(new, dtype=np.uint8), test
    else:
        # test is bigger - downscale it
        target = (rw, rh)
        new = Image.fromarray(test).resize(target, Image.LANCZOS)
        return ref, np.asarray(new, dtype=np.uint8)


def compute_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    """Compute SSIM, PSNR, and mean absolute error.

    SSIM uses channel_axis=-1 for color. PSNR uses default 8-bit data range.
    MAE is on uint8 scale (0-255), unbiased between channels.
    """
    if ref.shape != test.shape:
        die(f"Shape mismatch after resize: ref={ref.shape} test={test.shape}")

    ssim = ssim_fn(ref, test, channel_axis=-1, data_range=255)
    psnr = psnr_fn(ref, test, data_range=255)
    mae  = float(np.mean(np.abs(ref.astype(np.int32) - test.astype(np.int32))))
    return {"ssim": float(ssim), "psnr": float(psnr), "mae": mae,
            "shape": ref.shape}


def main() -> int:
    p = argparse.ArgumentParser(description="SSIM/PSNR comparison.")
    p.add_argument("reference", type=Path)
    p.add_argument("test",      type=Path)
    p.add_argument("--threshold", type=float, default=0.98,
                   help="SSIM pass threshold. Below this we exit 1. Default 0.98.")
    p.add_argument("--quiet", action="store_true",
                   help="Print only PASS/FAIL line.")
    args = p.parse_args()

    ref  = load_image_rgb(args.reference)
    test = load_image_rgb(args.test)
    ref, test = match_size(ref, test)

    m = compute_metrics(ref, test)
    passed = m["ssim"] >= args.threshold

    if args.quiet:
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}  ssim={m['ssim']:.4f}  thresh={args.threshold:.4f}  ref={args.reference.name}  test={args.test.name}")
    else:
        print(f"Reference : {args.reference}")
        print(f"Test      : {args.test}")
        print(f"Shape     : {m['shape']}")
        print(f"SSIM      : {m['ssim']:.4f}   (threshold {args.threshold:.4f})")
        print(f"PSNR      : {m['psnr']:.2f} dB")
        print(f"Mean Abs. : {m['mae']:.3f} (out of 255)")
        print(f"Verdict   : {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
