#!/usr/bin/env python3
"""
Single-pair SSIM/PSNR comparison with noise-aware SSIM.

Usage:
    python ssim_compare.py <reference.png_or_ppm> <test.png_or_ppm> \
        [--threshold 0.85] [--blur-sigma 1.5] [--no-blur]

Returns exit code 0 if SSIM >= threshold, else 1. Prints SSIM, PSNR, mean-abs-error
to stdout.

NOISE-AWARE COMPARISON (default):
    Path-traced renders compared against a high-spp reference fail default
    SSIM purely from Monte Carlo noise. We Gaussian-blur both images at
    sigma=1.5 px before SSIM (PSNR stays computed on unblurred images so
    you can still see the noise level). Pass --no-blur to disable.

If the two images differ in size, the larger one is downscaled to the
smaller's size with a BOX filter (averaging downsample, the unbiased
estimator for noisy data; matches what render_regression.py uses).

Dependencies: pillow + scikit-image + numpy. scipy is optional for blur;
we fall back to PIL.ImageFilter.GaussianBlur if scipy is missing.
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

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_image_rgb(path: Path) -> np.ndarray:
    """Load .png or .ppm into a uint8 HxWx3 numpy array."""
    if not path.exists():
        die(f"File not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def match_size(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """If shapes differ, downscale the larger to the smaller. BOX filter
    (averaging) is the unbiased downsampler for noisy data."""
    if ref.shape == test.shape:
        return ref, test

    rh, rw = ref.shape[:2]
    th, tw = test.shape[:2]

    if rh * rw > th * tw:
        # ref is bigger - downscale it
        target = (tw, th)
        new = Image.fromarray(ref).resize(target, Image.BOX)
        return np.asarray(new, dtype=np.uint8), test
    else:
        # test is bigger - downscale it
        target = (rw, rh)
        new = Image.fromarray(test).resize(target, Image.BOX)
        return ref, np.asarray(new, dtype=np.uint8)


def gaussian_blur_rgb(im: np.ndarray, sigma: float) -> np.ndarray:
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


def compute_metrics(ref: np.ndarray, test: np.ndarray, blur_sigma: float) -> dict:
    """Compute SSIM (on optionally-blurred pair), PSNR, MAE (both unblurred).

    SSIM uses channel_axis=-1 for color. PSNR uses 8-bit data range.
    MAE is on uint8 scale (0-255), unbiased between channels.
    """
    if ref.shape != test.shape:
        die(f"Shape mismatch after resize: ref={ref.shape} test={test.shape}")

    if blur_sigma > 0:
        ref_b  = gaussian_blur_rgb(ref,  blur_sigma)
        test_b = gaussian_blur_rgb(test, blur_sigma)
        ssim = ssim_fn(ref_b, test_b, channel_axis=-1, data_range=255)
    else:
        ssim = ssim_fn(ref, test, channel_axis=-1, data_range=255)

    # PSNR + MAE on the original images so the numbers retain their meaning
    # as signal-to-noise of actual pixels.
    psnr = psnr_fn(ref, test, data_range=255)
    mae  = float(np.mean(np.abs(ref.astype(np.int32) - test.astype(np.int32))))

    return {"ssim": float(ssim), "psnr": float(psnr), "mae": mae,
            "shape": ref.shape, "blur_sigma": blur_sigma}


def main() -> int:
    p = argparse.ArgumentParser(description="SSIM/PSNR comparison with noise-aware blur.")
    p.add_argument("reference", type=Path)
    p.add_argument("test",      type=Path)
    p.add_argument("--threshold", type=float, default=0.85,
                   help="SSIM pass threshold. Default 0.85 (assumes default --blur-sigma).")
    p.add_argument("--blur-sigma", type=float, default=1.5,
                   help="Gaussian blur sigma applied to both images before SSIM. "
                        "Default 1.5. Set to 0 to disable.")
    p.add_argument("--no-blur", action="store_true",
                   help="Disable blur (equivalent to --blur-sigma 0).")
    p.add_argument("--quiet", action="store_true",
                   help="Print only PASS/FAIL line.")
    args = p.parse_args()

    if args.no_blur:
        args.blur_sigma = 0.0

    ref  = load_image_rgb(args.reference)
    test = load_image_rgb(args.test)
    ref, test = match_size(ref, test)

    m = compute_metrics(ref, test, args.blur_sigma)
    passed = m["ssim"] >= args.threshold

    if args.quiet:
        verdict = "PASS" if passed else "FAIL"
        print(f"{verdict}  ssim={m['ssim']:.4f}  thresh={args.threshold:.4f}  ref={args.reference.name}  test={args.test.name}")
    else:
        print(f"Reference : {args.reference}")
        print(f"Test      : {args.test}")
        print(f"Shape     : {m['shape']}")
        print(f"Blur sigma: {m['blur_sigma']}")
        print(f"SSIM      : {m['ssim']:.4f}   (threshold {args.threshold:.4f})")
        print(f"PSNR      : {m['psnr']:.2f} dB")
        print(f"Mean Abs. : {m['mae']:.3f} (out of 255)")
        print(f"Verdict   : {'PASS' if passed else 'FAIL'}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
