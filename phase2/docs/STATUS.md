# Phase 2 Image Quality Investigation — Status

## What's verified
- Phase 2 build, render, and benchmark script all work. Throughput ~470 Mrays/s at 1080p.
- The PowerShell benchmark script is fixed (Start-Process with distinct temp files; cmd.exe /c for Python checks).
- Pixel-level diffs of Phase 2 vs Phase 1 (1080p/4096spp PNGs) are real and not just MC noise:
  - Glass sphere reflection content differs visibly (sharp specular = geometric, not noise)
  - Faint horizontal "ghost ring" on Phase 2 back wall, absent in Phase 1
  - Phase 2 has wider black side margins (col 0–223) vs Phase 1 (col 0–166)
  - **Phase 1 has ZERO white pixels in top rows**; Phase 2 has saturated-white pixels at the top with d*140

## What I got wrong earlier
- I assumed canonical smallpt `d * 140` was correct. It's not — **cu-smallpt deliberately uses `d * 130`**, which Phase 2 already had. The change to *140 was a regression.
- I burned tokens speculating about what cu-smallpt does instead of reading the source you gave me.

## Reading cu-smallpt vs Phase 2 line-by-line

Source files: `kernel.cu`, `geometry.cuh`, `sphere.hpp`, `vector.hpp`, `sampling.cuh`, `specular.cuh`, `math.hpp`, `imageio.hpp`.

Things that **match** between the two:
- Camera origin push: `eye + d * 130` ✓
- gaze direction `(0, -0.042612, -1)` normalized ✓
- cx = `(w*fov/h, 0, 0)` ✓
- cy = normalize(cx.cross(gaze)) * fov ✓
- fov = 0.5135 ✓
- Tent filter math (`u1<1 ? sqrt(u1)-1 : 1-sqrt(2-u1)`) ✓
- Diffuse hemisphere ONB construction (with the `|w.x| > 0.1` axis selection) ✓
- Specular reflect formula `d - 2*n.dot(d)*n` ✓
- Refractive Schlick + Russian roulette glass weighting ✓
- Russian roulette (after depth > 4, max-channel of albedo as continuation prob) ✓
- Both flip y when writing to output: `pixel = (h-1-y)*w + x` ✓
- Both gamma-correct with 1/2.2 in PPM write ✓
- `tmin = 1e-4` for primary and secondary rays ✓

Things that **differ**:

1. **Subpixel sampling.** cu-smallpt does 4 subpixels per pixel (2×2 grid), with each sample at a quadrant offset:
   ```
   d = cx*((sx+0.5+dx)*0.5/w + x/w - 0.5) + cy*(...) + gaze
   ```
   Phase 2 does single-sample-per-pixel:
   ```
   fx = (x + 0.5 + dx)/w - 0.5
   ```
   In expectation both center on the same pixel. cu-smallpt's spp arg means "samples per subpixel" so total rays per pixel = 4×spp. **Phase 2's spp is total samples per pixel.** This is fine for image equivalence but means the timings between Phase 1 and Phase 2 are at different ray counts.

2. **Precision.** cu-smallpt uses `double` everywhere. Phase 2 uses `float`. With sphere-radius-1e5 walls in cu-smallpt this matters a lot. With Phase 2's triangle walls it should matter less, but specular reflections off the small spheres may show subtle differences.

3. **Max-depth cap.** cu-smallpt has no hard cap, relies on Russian roulette. Phase 2 caps at 20 bounces. With 4096 spp and average path length ~6, this almost never fires; not the cause of the visible difference.

4. **Per-subpixel clamping.** cu-smallpt clamps L per subpixel before averaging (`Ls[i] += 0.25 * Clamp(L)`). Phase 2 clamps the per-pixel L once. For pixels containing the light disc, this differs slightly: per-subpixel clamp limits how bright a single subpixel can contribute, while per-pixel clamp limits the final average. Phase 2's approach lets one bright subpixel sample's high value dominate the average until clamping at the end.

5. **Triangle wall vs sphere wall geometry** (Phase 2's whole point of difference for the project). Phase 2 has documented this.

6. **Front wall.** cu-smallpt's front wall is a radius-1e5 black diffuse sphere centered at z=-1e5+170. Phase 2's front wall is a flat black diffuse triangle at z=170. **In cu-smallpt the front-wall sphere is convex from inside the room, so primary rays pointing forward but slightly off-axis can curve around it; in Phase 2 the front wall is exactly at z=170 and rays pointing in any +z direction hit it.** This is the most likely source of the wider black margins on the sides of Phase 2.

## What's most likely causing each visible artifact

- **Wider black side margins** (col 223 vs 167): primary rays at extreme x exit the room laterally before hitting any side wall, then hit the **front wall triangle** (black). In cu-smallpt the side-wall spheres extend infinitely so rays always hit them. **Likely fix: extend the side wall, floor, ceiling triangles further in z, or remove the front wall triangle entirely (smallpt uses it but it doesn't actually contribute to the visible image — it's a geometric escape valve).**

- **Wrong glass sphere reflection.** Glass reflects/refracts the surrounding scene. If the surrounding scene differs between Phase 1 and Phase 2 (different walls/normals), the reflection differs. The glass sphere is at z=78, looking back toward z=0 at the back wall. Triangle vs sphere back wall barely differs there. But it can also see the front wall (z=170) by reflecting forward — and Phase 2's front wall triangle vs Phase 1's far-away front wall sphere are very different. **Likely cause: front wall geometry difference is showing up in glass reflections.**

- **Ghost ring on back wall.** Less obvious. Could be:
  (a) Multi-bounce paths from the back wall → ceiling → light path, which differ slightly between sphere and triangle ceiling.
  (b) Single-precision artifacts at the back-wall/ceiling triangle seam.
  (c) The visible disc of the light sphere on the ceiling is slightly larger in Phase 2 because the ceiling triangle is exactly at y=81.6 while in cu-smallpt the ceiling sphere curves slightly downward away from the back wall.

## Highest-confidence next experiment

**Hypothesis:** the front wall triangle is the dominant cause of all three artifacts. cu-smallpt's front wall is geometrically present but practically invisible (radius-1e5 sphere far away); Phase 2's flat triangle is right in the camera path.

**Test:** disable the front wall in Phase 2 (remove triangles 6 and 7 from the geometry), re-render at 1080p/256spp, diff against Phase 2 baseline.

If the side margins shrink to col 167 and the glass reflection improves, the front wall is the bug.

## Tools built (in /home/claude)

- **`compare_ppms.py`**: PPM/PNG diff tool. Loads two images, prints mean/max abs diff per channel, identifies anomalous rows/columns, saves a heatmap (abs diff) and signed-diff (red/blue) image.
- **`canonical_smallpt_primary.py`**: Vectorized primary-ray-only renderer for canonical smallpt geometry (sphere walls). Useful for checking what camera math + scene combination produces what hit pattern. Outputs PNG colored by which primitive each pixel hits.
- **`top_row_inspector.py`**: scans top N rows of a PNG and reports white-pixel count and color content per row. Useful for catching the "rays escaping room → hit light directly" signature.
- **Phase 2 benchmark script** (`run_phase2_benchmark.ps1`): runs full grid with logging and CSV output, robust to OptiX info-level stderr.
