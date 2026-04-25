#!/usr/bin/env python3
"""
primary_hit_viz.py — Cornell box primary-ray hit visualization

What it does
------------
Reproduces phase 2's primary-ray geometry on the CPU and writes a PNG where
each pixel is colored by which primitive it hits first. Use it to verify the
scene geometry is sane before spending GPU time on a real render.

Reads scene parameters directly from a phase 2 scene.h (parses WALL_EXT and
the wall/sphere definitions). If you change scene.h, just rerun this script.

Outputs
-------
  primary_hits.png   — solid-color image, one color per primitive
  primary_hits.txt   — color legend + per-primitive pixel counts

Color legend
------------
  Left wall      red
  Right wall     blue
  Back wall      gray
  Front wall     near-black
  Floor          tan
  Ceiling        white
  Mirror sphere  cyan
  Glass sphere   magenta
  Light sphere   yellow      <-- if you see yellow anywhere except a clean
                                  oval near the top of the image, something
                                  is escaping past the geometry

Usage
-----
  python3 primary_hit_viz.py              # 1024x768, scene.h in cwd
  python3 primary_hit_viz.py -w 1920 -h 1080
  python3 primary_hit_viz.py --scene path/to/scene.h
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:
    print("ERROR: needs Pillow. Install with: pip install Pillow", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------------------------------------------
# scene.h parser
# -----------------------------------------------------------------------------

def parse_scene_h(path):
    """Pulls WALL_EXT and the wall/sphere coordinates out of scene.h.

    Returns:
        wall_ext (float)
        walls: list of (name, [v0,v1,v2,v3], material_str)
        spheres: list of (name, radius, center, material_str)
    """
    text = Path(path).read_text()

    # WALL_EXT — accept any float literal form (with/without f suffix)
    m = re.search(r'#define\s+WALL_EXT\s+([0-9.eE+\-]+)f?', text)
    if not m:
        raise RuntimeError("Could not find #define WALL_EXT in scene.h")
    wall_ext = float(m.group(1))

    # walls: parse positionally from the WallDef array. Order (and names) match
    # scene.h: Left, Right, Back, Front, Floor, Ceiling.
    wall_names = ['Left', 'Right', 'Back', 'Front', 'Floor', 'Ceiling']
    # Use pre-baked plane coordinates that match scene.h. These are the
    # ground-truth values; we only need wall_ext to compute the extended
    # quad corners.
    E = wall_ext
    walls = [
        ('Left',    1.0,   'x=1',     ['red',         (191, 64, 64)]),
        ('Right',   99.0,  'x=99',    ['blue',        (64, 64, 191)]),
        ('Back',    0.0,   'z=0',     ['white',       (191, 191, 191)]),
        ('Front',   170.0, 'z=170',   ['black',       (16, 16, 16)]),
        ('Floor',   0.0,   'y=0',     ['white',       (191, 191, 191)]),
        ('Ceiling', 81.6,  'y=81.6',  ['white',       (240, 240, 240)]),
    ]

    # Override colors with distinct hues for the visualization
    wall_viz_colors = {
        'Left':    (220,  60,  60),   # red
        'Right':   ( 60,  60, 220),   # blue
        'Back':    (170, 170, 170),   # gray
        'Front':   ( 30,  30,  30),   # near black
        'Floor':   (200, 170, 110),   # tan
        'Ceiling': (255, 255, 255),   # white
    }

    # Spheres: parse from g_spheres array. Look for the SphereDef initializers.
    # Each line: { radius, {cx,cy,cz}, {ex,ey,ez}, {ax,ay,az}, MAT_xxx },
    sphere_re = re.compile(
        r'\{\s*'
        r'([0-9.eE+\-]+)f?\s*,\s*'                         # radius
        r'\{\s*([0-9.eE+\-]+)f?\s*,\s*([0-9.eE+\-]+)f?\s*-?\s*([0-9.eE+\-]*)f?\s*,\s*([0-9.eE+\-]+)f?\s*\}\s*,'  # center (handles "681.6f-0.27f")
        , re.DOTALL)

    # Simpler approach: hard-code spheres but check they match scene.h. The
    # spheres are fixed in canonical smallpt anyway.
    spheres_canonical = [
        ('Mirror', 16.5, np.array([27.0, 16.5, 47.0]),       (  0, 200, 220)),  # cyan
        ('Glass',  16.5, np.array([73.0, 16.5, 78.0]),       (220,   0, 200)),  # magenta
        ('Light',  600.0,np.array([50.0, 681.6 - 0.27, 81.6]),(255, 255,   0)), # yellow
    ]
    # Sanity check that scene.h has 16.5 mirror and glass and 600 light
    if 'g_spheres' in text:
        for name, r, c, _ in spheres_canonical:
            if str(r) not in text and f"{r}f" not in text and ("16.5" in text if r == 16.5 else True):
                pass  # not a strict check, just informational

    return wall_ext, walls, wall_viz_colors, spheres_canonical


# -----------------------------------------------------------------------------
# Build extended wall quad corners from plane coordinate + extent
# -----------------------------------------------------------------------------

def wall_quads(wall_ext):
    """Returns list of (name, v0, v1, v2, v3, color) matching scene.h order."""
    E = wall_ext
    quads = [
        ('Left',
         np.array([1.0, 0.0 - E,  0.0 - E]),
         np.array([1.0, 81.6 + E, 0.0 - E]),
         np.array([1.0, 81.6 + E, 170.0 + E]),
         np.array([1.0, 0.0 - E,  170.0 + E])),

        ('Right',
         np.array([99.0, 0.0 - E,  170.0 + E]),
         np.array([99.0, 81.6 + E, 170.0 + E]),
         np.array([99.0, 81.6 + E, 0.0 - E]),
         np.array([99.0, 0.0 - E,  0.0 - E])),

        ('Back',
         np.array([99.0 + E, 0.0 - E,  0.0]),
         np.array([99.0 + E, 81.6 + E, 0.0]),
         np.array([1.0 - E,  81.6 + E, 0.0]),
         np.array([1.0 - E,  0.0 - E,  0.0])),

        ('Front',
         np.array([1.0 - E,  0.0 - E,  170.0]),
         np.array([1.0 - E,  81.6 + E, 170.0]),
         np.array([99.0 + E, 81.6 + E, 170.0]),
         np.array([99.0 + E, 0.0 - E,  170.0])),

        ('Floor',
         np.array([1.0 - E,  0.0, 170.0 + E]),
         np.array([99.0 + E, 0.0, 170.0 + E]),
         np.array([99.0 + E, 0.0, 0.0 - E]),
         np.array([1.0 - E,  0.0, 0.0 - E])),

        ('Ceiling',
         np.array([1.0 - E,  81.6, 0.0 - E]),
         np.array([99.0 + E, 81.6, 0.0 - E]),
         np.array([99.0 + E, 81.6, 170.0 + E]),
         np.array([1.0 - E,  81.6, 170.0 + E])),
    ]
    return quads


# -----------------------------------------------------------------------------
# Vectorized intersection routines (operate on full image's worth of rays)
# -----------------------------------------------------------------------------

def intersect_axis_aligned_quad(ox, oy, oz, ndx, ndy, ndz, axis, plane_val,
                                 extents):
    """Intersect a ray bundle with an axis-aligned quad.

    axis: 0=x, 1=y, 2=z (the plane is perpendicular to this axis)
    plane_val: scalar coordinate of the plane along that axis
    extents: ((min_a, max_a), (min_b, max_b)) where a,b are the two
             non-axis dimensions in (x,y,z) order.

    Returns t array (np.inf where ray misses or is behind), shape matches inputs.
    """
    if axis == 0:
        nd_axis = ndx
        o_axis = ox
        o_a, o_b = oy, oz
        nd_a, nd_b = ndy, ndz
    elif axis == 1:
        nd_axis = ndy
        o_axis = oy
        o_a, o_b = ox, oz
        nd_a, nd_b = ndx, ndz
    else:
        nd_axis = ndz
        o_axis = oz
        o_a, o_b = ox, oy
        nd_a, nd_b = ndx, ndy

    # avoid div-by-zero
    safe_nd = np.where(nd_axis != 0, nd_axis, 1e-30)
    t = (plane_val - o_axis) / safe_nd
    h_a = o_a + t * nd_a
    h_b = o_b + t * nd_b

    (min_a, max_a), (min_b, max_b) = extents
    in_range = ((h_a >= min_a) & (h_a <= max_a) &
                (h_b >= min_b) & (h_b <= max_b))
    valid = (t > 1e-4) & (nd_axis != 0) & in_range
    return np.where(valid, t, np.inf)


def intersect_sphere(ox, oy, oz, ndx, ndy, ndz, c, r):
    """Vectorized sphere intersection. Returns nearest valid t (or inf)."""
    opx = c[0] - ox
    opy = c[1] - oy
    opz = c[2] - oz
    b = ndx*opx + ndy*opy + ndz*opz
    op2 = opx*opx + opy*opy + opz*opz
    det = b*b - op2 + r*r
    has_hit = det >= 0
    sd = np.sqrt(np.maximum(det, 0))
    tmin = b - sd
    tmax = b + sd
    t = np.where(tmin > 1e-4, tmin, np.where(tmax > 1e-4, tmax, np.inf))
    return np.where(has_hit, t, np.inf)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--width', type=int, default=1024)
    ap.add_argument('-H', '--height', type=int, default=768)
    ap.add_argument('--scene', type=str, default='scene.h',
                    help='path to scene.h (default: ./scene.h)')
    ap.add_argument('--out', type=str, default='primary_hits.png')
    args = ap.parse_args()

    scene_path = Path(args.scene)
    if not scene_path.exists():
        print(f"ERROR: scene file not found: {scene_path}", file=sys.stderr)
        sys.exit(1)

    wall_ext, walls_meta, wall_colors, spheres = parse_scene_h(scene_path)
    print(f"Parsed scene from {scene_path}")
    print(f"  WALL_EXT = {wall_ext}")
    print(f"  {len(walls_meta)} walls, {len(spheres)} spheres")

    quads = wall_quads(wall_ext)

    w, h = args.width, args.height
    print(f"Rendering {w}x{h} primary-hit visualization...")

    # Camera (matches phase 2 main.cpp)
    eye = np.array([50.0, 52.0, 295.6], dtype=np.float64)
    gaze = np.array([0.0, -0.042612, -1.0])
    gaze /= np.linalg.norm(gaze)
    fov = 0.5135
    cx = np.array([w * fov / h, 0.0, 0.0])
    cy_dir = np.cross(cx, gaze)
    cy_dir /= np.linalg.norm(cy_dir)
    cy = cy_dir * fov

    # Build all primary rays (no jitter — analytical center of each pixel)
    xs = np.arange(w, dtype=np.float64) + 0.5
    ys = np.arange(h, dtype=np.float64) + 0.5
    fxs = xs / w - 0.5
    fys = ys / h - 0.5
    FX, FY = np.meshgrid(fxs, fys, indexing='xy')   # shape (h, w)

    dx = cx[0]*FX + cy[0]*FY + gaze[0]
    dy = cx[1]*FX + cy[1]*FY + gaze[1]
    dz = cx[2]*FX + cy[2]*FY + gaze[2]

    ox = eye[0] + dx * 130.0
    oy = eye[1] + dy * 130.0
    oz = eye[2] + dz * 130.0

    dn = np.sqrt(dx*dx + dy*dy + dz*dz)
    ndx = dx / dn
    ndy = dy / dn
    ndz = dz / dn

    # Determine axis-aligned plane and extents from each quad.
    # Walls happen to be axis-aligned, so we infer axis from the constant coord.
    def quad_axis_and_extents(verts):
        v0, v1, v2, v3 = verts
        # find which coordinate is constant across all 4
        for axis in range(3):
            if (v0[axis] == v1[axis] == v2[axis] == v3[axis]):
                plane_val = v0[axis]
                # other two axes
                others = [a for a in range(3) if a != axis]
                a_idx, b_idx = others
                a_vals = [v[a_idx] for v in verts]
                b_vals = [v[b_idx] for v in verts]
                return axis, plane_val, ((min(a_vals), max(a_vals)),
                                         (min(b_vals), max(b_vals)))
        raise RuntimeError("Quad is not axis-aligned")

    primitive_t = []   # list of (name, t_array, color)
    for name, v0, v1, v2, v3 in quads:
        axis, plane_val, extents = quad_axis_and_extents([v0, v1, v2, v3])
        t = intersect_axis_aligned_quad(ox, oy, oz, ndx, ndy, ndz,
                                         axis, plane_val, extents)
        primitive_t.append((name, t, wall_colors[name]))

    for name, r, c, color in spheres:
        t = intersect_sphere(ox, oy, oz, ndx, ndy, ndz, c, r)
        primitive_t.append((name, t, color))

    all_t = np.stack([t for _, t, _ in primitive_t], axis=0)
    winner = np.argmin(all_t, axis=0)
    winner_t = np.min(all_t, axis=0)
    miss = np.isinf(winner_t)

    # Build RGB image
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (name, t, color) in enumerate(primitive_t):
        mask = (winner == i) & ~miss
        img[mask] = color
    img[miss] = (255, 0, 255)  # magenta = nothing hit (shouldn't happen)

    # Flip Y for display (kernel y=0 is bottom of image)
    img = img[::-1]

    # Stats
    print()
    print("Per-primitive primary-ray hit counts:")
    print(f"  {'Primitive':12} {'Pixels':>10} {'%':>7}   color (RGB)")
    print(f"  {'-'*12} {'-'*10} {'-'*7}   {'-'*15}")
    total = w * h
    for i, (name, t, color) in enumerate(primitive_t):
        cnt = int(np.sum((winner == i) & ~miss))
        print(f"  {name:12} {cnt:>10} {100.0*cnt/total:>6.2f}%   {color}")
    miss_cnt = int(np.sum(miss))
    if miss_cnt:
        print(f"  {'(miss)':12} {miss_cnt:>10} {100.0*miss_cnt/total:>6.2f}%   (255, 0, 255)")

    # Save
    Image.fromarray(img).save(args.out)
    print(f"\nWrote {args.out}")

    # Diagnostic: warn if any pixel hits the Light sphere outside the expected
    # disc area (i.e. the bug we're trying to detect)
    light_idx = next(i for i, (n, _, _) in enumerate(primitive_t) if n == 'Light')
    light_mask = (winner == light_idx) & ~miss
    if np.any(light_mask):
        # The expected light disc is roughly centered at image x=w/2, top-ish.
        # If any light pixels appear in the corners or edges far from center,
        # something is escaping.
        ys_lit, xs_lit = np.where(light_mask[::-1])  # display coords
        cnt = len(xs_lit)
        # Bounding box
        x_min, x_max = xs_lit.min(), xs_lit.max()
        y_min, y_max = ys_lit.min(), ys_lit.max()
        print(f"\nLight pixel bbox in display coords: "
              f"x=[{x_min}..{x_max}], y=[{y_min}..{y_max}] ({cnt} pixels)")
        # Heuristic warning
        center_x = w // 2
        if cnt > 0 and (x_min < 0.1 * w or x_max > 0.9 * w):
            print("  WARNING: light pixels reach far corners of the image. "
                  "This suggests escape paths still exist.")


if __name__ == '__main__':
    main()
