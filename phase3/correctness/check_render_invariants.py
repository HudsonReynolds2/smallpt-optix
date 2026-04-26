#!/usr/bin/env python3
"""
Non-reference invariant checks on a Cornell-box render.

These checks don't compare against a known-good reference. Instead they
catch the most common ways a path-traced Cornell box can go wrong without
needing the 4K ground-truth image:

  1. No NaN/Inf pixels.
  2. Mean luminance is in a sane band (not all black, not all white).
  3. No huge fully-black contiguous region (catches broken-tile bugs from
     phase 3's tile-launch code).
  4. No huge fully-saturated-white region (catches energy explosions, RR
     gone wrong, missing tone mapping).
  5. Cornell-box color sanity:
       - The light region near the top-middle of the image is bright.
       - The left wall (left strip of pixels at mid-height) is more red
         than blue.
       - The right wall (right strip) is more blue than red.
       - The back/floor/ceiling area (center horizontal band) is roughly
         neutral (not strongly red or blue tinted).

The checks are intentionally loose. The point is to catch broken renders,
not to tightly characterize the image. False negatives (broken render that
passes) are acceptable here -- the SSIM regression catches structural
issues. False positives (correct render that fails) are NOT acceptable
since this runs as a gate.

Usage:
    python check_render_invariants.py <image.png_or_ppm> [--quiet] [--json results.json]

Returns exit 0 if all checks pass, else 1. Prints a per-check breakdown.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


try:
    import numpy as np
    from PIL import Image
except ImportError as e:
    die(f"Missing Python deps: {e}\nRun: pip install pillow numpy")


# Thresholds. Calibrated against the known-good 4K reference.
# Loose enough to pass low-spp renders that are noisy but valid.
NAN_INF_TOLERANCE   = 0           # zero tolerance for NaN/Inf
MEAN_LUM_MIN        = 0.05        # full image mean luminance, normalized 0..1
MEAN_LUM_MAX        = 0.70        # too high = blown out
BLACK_PIXEL_THRESH  = 8           # uint8: anything <= this counts as "black"
WHITE_PIXEL_THRESH  = 250         # uint8: anything >= this counts as "saturated"
MAX_BLACK_FRACTION  = 0.15        # fraction of pixels that may be "black"
MAX_WHITE_FRACTION  = 0.05        # saturated whites allowed up to this fraction
LIGHT_REGION_MIN_LUM = 0.45       # top-middle region must be at least this bright
WALL_TINT_RATIO     = 1.10        # left wall: R/B >= this (and right wall: B/R >= this)
CENTER_NEUTRAL_RATIO = 1.50       # |R-B|/min(R,B) must be < this for center band


def load_rgb(path: Path) -> np.ndarray:
    """Load PNG or PPM into HxWx3 uint8."""
    if not path.exists():
        die(f"File not found: {path}")
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def luminance(rgb_u8: np.ndarray) -> np.ndarray:
    """Rec.709-ish luminance, normalized to 0..1. Float64 for stability."""
    a = rgb_u8.astype(np.float64) / 255.0
    return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]


def fraction_black(rgb_u8: np.ndarray) -> float:
    """Fraction of pixels where ALL channels are <= BLACK_PIXEL_THRESH."""
    mask = np.all(rgb_u8 <= BLACK_PIXEL_THRESH, axis=-1)
    return float(np.mean(mask))


def fraction_white(rgb_u8: np.ndarray) -> float:
    """Fraction of pixels where ALL channels are >= WHITE_PIXEL_THRESH."""
    mask = np.all(rgb_u8 >= WHITE_PIXEL_THRESH, axis=-1)
    return float(np.mean(mask))


def check_no_nan_inf(rgb_u8: np.ndarray) -> tuple[bool, str]:
    """Trivial since we loaded as uint8 (Pillow already coerced); included
    so the test exists as a named line item in the report. The check is
    against the float promotion, since uint8 itself can't hold NaN/Inf."""
    f = rgb_u8.astype(np.float32)
    bad_count = int(np.sum(~np.isfinite(f)))
    ok = bad_count <= NAN_INF_TOLERANCE
    return ok, f"{bad_count} non-finite pixels (allowed: {NAN_INF_TOLERANCE})"


def check_mean_luminance(rgb_u8: np.ndarray) -> tuple[bool, str]:
    lum_mean = float(np.mean(luminance(rgb_u8)))
    ok = MEAN_LUM_MIN <= lum_mean <= MEAN_LUM_MAX
    return ok, f"mean luminance {lum_mean:.3f} (band [{MEAN_LUM_MIN}, {MEAN_LUM_MAX}])"


def check_no_black_dominant(rgb_u8: np.ndarray) -> tuple[bool, str]:
    f = fraction_black(rgb_u8)
    ok = f <= MAX_BLACK_FRACTION
    return ok, f"black-pixel fraction {f:.3f} (max {MAX_BLACK_FRACTION})"


def check_no_white_dominant(rgb_u8: np.ndarray) -> tuple[bool, str]:
    f = fraction_white(rgb_u8)
    ok = f <= MAX_WHITE_FRACTION
    return ok, f"saturated-white fraction {f:.3f} (max {MAX_WHITE_FRACTION})"


def check_light_region_bright(rgb_u8: np.ndarray) -> tuple[bool, str]:
    """The light is at the top of the Cornell box, image-bottom because
    we flip Y on output. After flip-on-write the bright light region is
    actually at image *top*. We sample the top-middle 30% wide x 15% tall
    window and require its mean luminance to exceed the threshold."""
    h, w, _ = rgb_u8.shape
    x0 = int(w * 0.35); x1 = int(w * 0.65)
    y0 = int(h * 0.05); y1 = int(h * 0.20)
    region = rgb_u8[y0:y1, x0:x1, :]
    lum = float(np.mean(luminance(region)))
    ok = lum >= LIGHT_REGION_MIN_LUM
    return ok, f"light-region luminance {lum:.3f} (min {LIGHT_REGION_MIN_LUM})"


def check_left_wall_red(rgb_u8: np.ndarray) -> tuple[bool, str]:
    """Left wall is the red wall. Sample a vertical strip on the left edge
    at mid-height. Skip the top (light) and bottom (floor edge effects)."""
    h, w, _ = rgb_u8.shape
    x0 = 0;          x1 = max(int(w * 0.06), 8)
    y0 = int(h*0.30); y1 = int(h*0.70)
    region = rgb_u8[y0:y1, x0:x1, :].astype(np.float64)
    r_mean = float(np.mean(region[..., 0]))
    b_mean = float(np.mean(region[..., 2])) + 1e-6
    ratio = r_mean / b_mean
    ok = ratio >= WALL_TINT_RATIO
    return ok, f"left wall R/B = {ratio:.2f} (min {WALL_TINT_RATIO}); R={r_mean:.1f} B={b_mean:.1f}"


def check_right_wall_blue(rgb_u8: np.ndarray) -> tuple[bool, str]:
    """Right wall is the blue wall. Same sampling pattern, flipped."""
    h, w, _ = rgb_u8.shape
    x0 = w - max(int(w * 0.06), 8); x1 = w
    y0 = int(h*0.30); y1 = int(h*0.70)
    region = rgb_u8[y0:y1, x0:x1, :].astype(np.float64)
    r_mean = float(np.mean(region[..., 0])) + 1e-6
    b_mean = float(np.mean(region[..., 2]))
    ratio = b_mean / r_mean
    ok = ratio >= WALL_TINT_RATIO
    return ok, f"right wall B/R = {ratio:.2f} (min {WALL_TINT_RATIO}); R={r_mean:.1f} B={b_mean:.1f}"


def check_center_neutral(rgb_u8: np.ndarray) -> tuple[bool, str]:
    """The center band (back wall / floor / ceiling) should be neutral. We
    use a wide horizontal band that excludes both red/blue side walls and
    require |R-B| / min(R,B) to be modest. This catches a swapped-channel
    bug that wouldn't be caught by the wall checks alone (because if R and
    B are swapped uniformly the wall checks could still pass against the
    swapped material colors)."""
    h, w, _ = rgb_u8.shape
    x0 = int(w*0.20); x1 = int(w*0.80)
    y0 = int(h*0.30); y1 = int(h*0.70)
    region = rgb_u8[y0:y1, x0:x1, :].astype(np.float64)
    r_mean = float(np.mean(region[..., 0]))
    b_mean = float(np.mean(region[..., 2]))
    diff = abs(r_mean - b_mean)
    base = max(min(r_mean, b_mean), 1.0)
    ratio = diff / base
    ok = ratio < CENTER_NEUTRAL_RATIO
    return ok, f"center R-B asymmetry {ratio:.2f} (max {CENTER_NEUTRAL_RATIO}); R={r_mean:.1f} B={b_mean:.1f}"


CHECKS = [
    ("no_nan_inf",        check_no_nan_inf),
    ("mean_luminance",    check_mean_luminance),
    ("no_black_dominant", check_no_black_dominant),
    ("no_white_dominant", check_no_white_dominant),
    ("light_region",      check_light_region_bright),
    ("left_wall_red",     check_left_wall_red),
    ("right_wall_blue",   check_right_wall_blue),
    ("center_neutral",    check_center_neutral),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Cornell-box render invariants check.")
    p.add_argument("image", type=Path)
    p.add_argument("--quiet", action="store_true",
                   help="Print only the final PASS/FAIL line.")
    p.add_argument("--json", type=Path, default=None,
                   help="Write per-check results as JSON to this path.")
    args = p.parse_args()

    rgb = load_rgb(args.image)

    if not args.quiet:
        print(f"Image: {args.image}  shape={rgb.shape}")
        print()

    results = []
    all_pass = True
    for name, fn in CHECKS:
        try:
            ok, detail = fn(rgb)
        except Exception as e:
            ok, detail = False, f"check raised: {e}"
        results.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            all_pass = False
        if not args.quiet:
            badge = "PASS" if ok else "FAIL"
            print(f"  [{badge}] {name:<22s} {detail}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"image": str(args.image), "all_pass": all_pass,
                       "checks": results}, f, indent=2)

    if args.quiet:
        verdict = "PASS" if all_pass else "FAIL"
        n_fail = sum(1 for r in results if not r["pass"])
        print(f"{verdict}  {len(results) - n_fail}/{len(results)} checks passed  image={args.image.name}")
    else:
        print()
        n_pass = sum(1 for r in results if r["pass"])
        print(f"Summary: {n_pass}/{len(results)} checks passed -> {'PASS' if all_pass else 'FAIL'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
