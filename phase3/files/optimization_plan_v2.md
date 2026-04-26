# EC527 Phase 2+ Optimization Plan (v2)
## Path to "the optimized version" + 6-minute presentation

**Owner:** Hudson Reynolds
**Course:** EC527 (BU, Spring 2026)
**Plan version:** v2, supersedes v1 (kept as historical record)

### Changes in v2 (vs v1)

1. **Russian Roulette is already in the phase 2 baseline.** v1 listed RR as a future variant to add. It's been in `phase2/src/shaders.cu` (closest-hit, depth>4, max-of-albedo continuation, throughput compensation) the whole time. There is no separate `rr` variant to land. The new variant chain starts directly at `phase3_tess_tile`.
2. **Phase 3 = sphere tessellation + tiling, combined.** Originally split across §3.3 and §3.4. The combined upgrade keeps all phase 2 behaviour, swaps built-in OptiX spheres for tessellated triangle meshes, drops the IAS, and wraps `optixLaunch` in a host-side tile loop.
3. **Compile-time disable for tiling** via `-DTILE_W=0 -DTILE_H=0` so a sphere-tess-only build is one cmake flag away if the A/B isolates a tile-loop bug.
4. **Nsight Compute requires `OPTIX_FORCE_DEPRECATED_LAUNCHER=1`** to see the raygen kernel at all. Without it, ncu sees only the `optixAccelBuild` BVH kernels and reports "no kernels were profiled." `phase3/run_nsight.ps1` sets this env var automatically.

---

## 0. System Context

### Hardware / OS

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3080 Ti (SM 8.6, Ampere) |
| Driver | 596.21 |
| OS (Phase 2/3 builds) | Windows 11 |
| OS (Phase 1 / WSL utility) | Ubuntu 24.04 (WSL2) on same machine "enterprise" |
| CUDA Toolkit | 12.8 (Windows native), 12.0 (WSL apt) |
| OptiX SDK | 8.1.0 (Windows only) |
| Host compiler | MSVC 19.44 (VS 2022 Build Tools) on Windows |
| Arch flag | `-arch=sm_86` |

**Important Ampere fact:** No SER (Ada+ only). `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` disables SER while set, but on Ampere there's nothing to disable, so this has no effect on observed perf.

### Project layout (current)

```
527project/
├── README_phase2_addendum.md
├── phase1/                            # cu-smallpt baseline
├── phase2/                            # OptiX baseline (RR + tessellated walls)
│   ├── CMakeLists.txt
│   ├── src/{main.cpp, shaders.cu, shared.h, scene.h}
│   ├── scenes/scene_default.h
│   ├── scene.h                        # active scene
│   ├── select_scene.ps1
│   ├── run_phase2_benchmark.ps1
│   └── docs/{STATUS.md, plan.md, optimization_plan.md, optimization_plan_v2.md}
├── phase3/                            # sphere tess + tiling
│   ├── CMakeLists.txt                 # copied from phase2/
│   ├── src/{main.cpp, shaders.cu, shared.h, scene.h}     # phase 3 versions
│   ├── scenes/scene_default.h         # walls + sphere tess
│   ├── scene.h
│   ├── select_scene.ps1               # copied from phase2/
│   ├── run_phase3_benchmark.ps1       # copied from phase2/, paths updated
│   ├── run_nsight.ps1                 # NEW
│   ├── reference/4096x3072_4096spp.{png,ppm}
│   └── docs/optimization_plan.md      # v1, this file = v2
```

### Phase 2 baseline (what's actually in the code right now)

- **Pipeline:** raygen iterative bounce loop (max depth 20). Closest-hit branches on `optixGetPrimitiveType()` to handle triangles vs built-in spheres, then on material type. RR after depth 4 with max-of-albedo continuation prob.
- **Geometry:** 6 tessellated walls (16×16 grid + skirts → 3,888 tris). Mirror, glass, light = OptiX built-in spheres. Two GASes (triangles + spheres) under one IAS.
- **SBT:** 6 wall hit-group records (consolidated via `sbtIndexOffsetBuffer`) + 3 sphere records = 9 records. Two CH program groups (triangle CH and sphere CH).
- **Throughput:** ~313 Mrays/s at 1024×768_64spp (just measured), ~272 at 512×384_256spp. Roughly 9× vs phase 1.
- **Memory:** 4096×3072×16384 spp consumed 18 GB (10.7 GB device + 7.1 GB system fallback). Tile launches will fix.
- **Reference image:** `phase3/reference/4096x3072_4096spp.png` is the SSIM ground truth.

---

## 1. The single phase 3 upgrade

Phase 3 lands one variant: `phase3_tess_tile`. It bundles:

- **Sphere tessellation:** mirror, glass, light → tessellated triangle meshes (default 64×64 lat/lon = 8,192 tris each). Single triangle GAS for the entire scene; no IAS, no built-in sphere primitives, no second CH program group. `compute_normal()` in shaders.cu loses its sphere branch.
- **Tile launches:** host-side loop over 512×512 tiles, single accumulation buffer at full image resolution, `tile_origin_{x,y}` added to `Params`. The raygen offsets its launch index by tile origin to reach the right pixel.

### Why bundle them

If they regressed independently we'd want to ship them separately. They don't:

- Sphere tess is a structural change (geometry path + pipeline options).
- Tile loop is a host-side launch reshape (no pipeline changes).

Bundling means one A/B against the phase 2 baseline. If something breaks, isolate by recompiling phase 3 with `-DTILE_W=0 -DTILE_H=0` (single-launch mode, sphere tess only). If that renders correctly, the bug is in the tile loop.

### Alternative path (if sphere tess breaks)

Drop `scenes/scene_default.h` to use only walls + revert sphere geometry to built-ins (i.e. revert most of `main.cpp` and `shaders.cu`). Ship "phase 2 + tile launches" as the optimized variant. The wall-tessellation precision story is already a strong narrative without sphere tess.

---

## 2. Build + Test

### 2.1 Build phase 3

After phase 3 src/scenes are in place:

```powershell
cd C:\Users\hudsonre\527project\phase3
powershell -ExecutionPolicy Bypass -File .\select_scene.ps1 -Name default
```

This copies `scenes/scene_default.h` -> `src/scene.h` and runs cmake + msbuild Release. (Phase 3's `select_scene.ps1` is a copy of phase 2's; it operates in the local phase 3 tree.)

### 2.2 Sanity-render

```powershell
powershell -ExecutionPolicy Bypass -File .\run_phase3_benchmark.ps1 -ConfigFilter "1024x768_256spp"
```

Open the resulting PNG and visually compare to `reference/4096x3072_4096spp.png` (just downscale in your head). Mirror reflection should look smooth at 64×64 tessellation; if you see facets, bump `SPHERE_TESS_LAT` / `SPHERE_TESS_LON` to 128 in `scenes/scene_default.h`.

### 2.3 Nsight A/B

Phase 2 baseline (do this once, never re-run unless phase 2 source changes):

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 `
    -Variant phase2_baseline `
    -ExePath ..\phase2\build\Release\optix_smallpt.exe `
    -PtxPath ..\phase2\build\shaders.ptx `
    -Mode quick

powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 `
    -Variant phase2_baseline `
    -ExePath ..\phase2\build\Release\optix_smallpt.exe `
    -PtxPath ..\phase2\build\shaders.ptx `
    -Mode full
```

Phase 3:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 `
    -Variant phase3_tess_tile `
    -ExePath .\build\Release\optix_smallpt.exe `
    -PtxPath .\build\shaders.ptx `
    -Mode quick

powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 `
    -Variant phase3_tess_tile `
    -ExePath .\build\Release\optix_smallpt.exe `
    -PtxPath .\build\shaders.ptx `
    -Mode full
```

The script automatically sets `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` for ncu, filters to `--kernel-name regex:(?i)raygen`, writes a PPM+PNG of every profiled config to `<run>/renders/`, and fails loudly (and visibly) if no kernel matching `raygen` was actually profiled.

### 2.4 Memory test

To validate the 18 GB → <2 GB claim:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_phase3_benchmark.ps1 -ConfigFilter "4096x3072_1024spp"
```

While it runs, in another terminal: `nvidia-smi --loop-ms=500`. Peak GPU memory should stay well under 2 GB even at 4K. (Compare to phase 2's 10.7 GB device + 7.1 GB system fallback at the same resolution.)

---

## 3. Future work (not for this deck)

- **NEE / MIS** — variance reduction at diffuse hits. 2 hours of math + bias risk. v1 plan included this conditionally; v2 drops it for the same time-budget reason.
- **OptiX denoiser** — 1-spp + denoiser vs 1024-spp ground truth makes a great slide. Out of scope unless time permits after the deck is built.
- **Bounce-depth ablation** — 30 min experiment. Plot time-vs-depth at 1024×768_1024spp for depth in {1, 2, 3, 4, 6, 8, 12, 20}. RR makes the cap mostly cosmetic but the curve is informative for the deck.
- **SER** — Ada+ only. Not on this hardware.
- **Wavefront restructure** — multi-day, future Phase 4.

---

## 4. Slide outline (unchanged from v1)

See v1 §6. The optimization-stack bar chart (slide 5) has fewer bars now: `phase2_baseline → phase3_tess_tile`. If time permits, add a third bar for `phase3_tess_tile_denoised` or `phase3_tess_tile_nee`.

---

## 5. Risks

| Risk | Trigger | Mitigation |
|---|---|---|
| Sphere tessellation looks faceted | Visible polygon edges on glass/mirror | Bump `SPHERE_TESS_LAT/LON` from 64 to 128 (or 96). Higher = slower BVH build, bigger GAS, but smoother edges. |
| Sphere tess SSIM < 0.98 | Geometry changed → small radiance shifts expected | Use SsimThreshold 0.97 for this variant. |
| Tile loop produces banding/seams | Visible tile boundaries in render | Compile with `-DTILE_W=0 -DTILE_H=0` to disable tiling and confirm sphere tess alone is fine; then debug the launch loop. |
| Nsight still says "no kernels profiled" | After OPTIX_FORCE_DEPRECATED_LAUNCHER set | Set `moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE` in main.cpp. Last-resort. |
| 4K still shows >2 GB GPU memory | After tiling | Check accum_buffer is the only full-image buffer. Search for any other allocation scaling with W×H. |

---

## 6. Definition of Done

- [ ] Phase 3 builds and renders 1024×768 cleanly.
- [ ] Visual: mirror + glass spheres look smooth (no facet artifacts) at 64×64 tessellation.
- [ ] Phase 3 ncu quick + full captures both contain a raygen kernel (checked automatically by `run_nsight.ps1`).
- [ ] Phase 3 4K_1024spp peak GPU memory < 2 GB.
- [ ] Phase 2 baseline ncu captures present in `phase3/results/nsight/`.
- [ ] Bar chart (Mrays/s) and image side-by-side (matched-spp render comparison) ready for slides.

---

## 7. Hand-off

If picking this up cold:

- Phase 2 source (the baseline) is what was at HEAD when this v2 was written. Has RR (depth>4), tessellated walls, built-in OptiX spheres, IAS over two GASes.
- Phase 3 source: this directory. Single triangle GAS, no IAS, tile loop on host.
- The reference render is at `phase3/reference/4096x3072_4096spp.png`.
- `run_nsight.ps1` MUST set `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` or ncu won't see the raygen kernel.
