# EC527 Phase 2+ Optimization Plan
## Path to "the optimized version" + 6-minute presentation

**Owner:** Hudson Reynolds
**Course:** EC527 (BU, Spring 2026)
**Time budget:** ~10 working hours today, then writing/recording in remaining days
**Deliverable:** 6-minute presentation + optimized renderer + benchmark/Nsight data
**Plan version:** v1, supersedes the original `phase2/docs/plan.md` (kept as historical record)

---

## 0. System Context (so this doc is self-contained)

If this plan needs to be handed to a fresh conversation, everything below is the relevant state.

### Hardware / OS

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3080 Ti (SM 8.6, Ampere) |
| Driver | 596.21 |
| OS (Phase 2/3 builds) | Windows 11 |
| OS (Phase 1 / WSL utility) | Ubuntu 24.04 (WSL2) on same machine "enterprise" |
| CUDA Toolkit | 12.8 (Windows native), 12.0 (WSL apt) |
| OptiX SDK | 8.1.0 (Windows only — WSL2 driver stack ships a stub `libnvoptix.so.1`) |
| Host compiler | MSVC 19.44 (VS 2022 Build Tools) on Windows |
| Arch flag | `-arch=sm_86` |

**Important Ampere fact:** No Shader Execution Reordering (SER). SER is Ada+ only. Plan assumes no SER. If a SCC RTX 6000 Ada becomes available later, SER becomes the final-stage optimization (Section 9).

### Project layout

```
527project/
├── README_phase2_addendum.md     # wall tessellation postscript
├── phase1/                        # cu-smallpt baseline (separate conversation)
│   ├── CMakeLists.txt
│   └── src/
│       ├── kernel.cu              # main path tracing kernel
│       ├── geometry.cuh, sphere.hpp, vector.hpp, math.hpp, sampling.cuh, specular.cuh
│       └── cuda_tools.hpp, imageio.hpp
├── phase2/                        # OptiX renderer (this plan operates here)
│   ├── CMakeLists.txt
│   ├── docs/
│   │   ├── STATUS.md
│   │   ├── phase2_handoff.md
│   │   ├── plan.md                # original phase 1/2/3 plan, now historical
│   │   └── optimization_plan.md   # THIS FILE
│   ├── primary_hit_viz.py
│   ├── run_phase2_benchmark.ps1   # existing benchmark runner
│   ├── select_scene.ps1           # copies scenes/<name>.h -> scene.h
│   ├── scene.h                    # active scene (overwritten by select_scene)
│   ├── scenes/
│   │   └── scene_default.h        # tessellated Cornell box
│   └── src/
│       ├── main.cpp               # OptiX host: GAS/IAS, pipeline, SBT, launch
│       ├── scene.h                # mirror of root scene.h (build includes)
│       ├── shaders.cu             # raygen + closest-hit + miss
│       └── shared.h               # Params, PRD, HitGroupData
└── phase3/
    └── reference/
        ├── 4096x3072_4096spp.png  # ground truth for SSIM (Phase 2 produced)
        └── 4096x3072_4096spp.ppm
```

### Phase 2 baseline (current state)

- **Pipeline:** raygen with iterative bounce loop (max depth 20, no Russian Roulette), single closest-hit shader branching on `optixGetPrimitiveType()` and material type, miss shader sets `prd.done`.
- **Geometry:** 6 walls tessellated to 16×16 visible grid + skirts → 3,888 triangles total. Mirror, glass, light remain built-in OptiX spheres. Two GASes under one IAS.
- **SBT:** 6 wall hit-group records (consolidated via `sbtIndexOffsetBuffer`) + 3 sphere records = 9 records.
- **Throughput:** ~300 Mrays/s (vs ~35 Mrays/s for Phase 1 cu-smallpt). ~9× speedup.
- **Memory:** 4096×3072×16384 spp run consumed 18 GB GPU memory (10.7 GB device + 7.1 GB system fallback) — likely OptiX per-launch state scaling with launch size, not geometry. Tile launches will fix.
- **Reference image:** `phase3/reference/4096x3072_4096spp.png` is the SSIM ground truth for all regression tests below.

### What "the optimized version" means in this plan

Folding Phase 3 (sphere tessellation) and a stack of optimizations into a single optimized binary, benchmarked against the current Phase 2 baseline:

1. Russian Roulette (RR)
2. Sphere tessellation (= old Phase 3)
3. Tile-based launches (memory + occupancy)
4. Bounce-depth cap reduced from 20, justified by ablation
5. Next Event Estimation (NEE) — optional, time-permitting
6. OptiX Denoiser path — optional, time-permitting
7. Wavefront restructure — out of scope for 1-day budget, listed as future work
8. SER — out of scope unless SCC RTX 6000 Ada arrives

---

## 1. Time Budget

**Today (~10 hours):** Sections 2 → 5. Get scripts + RR + sphere tess + benchmark/Nsight data done. **This is the hard cutoff for engineering work.**

**Tomorrow (presentation day):** Section 6 (slides + recording). No new code. NEE and denoiser slip to "future work" if not done by end of today.

**Stretch (if today goes faster than expected):** NEE first, then denoiser.

If something breaks and steals 3+ hours: drop sphere tessellation, ship "Phase 2 + RR + tile launches" as the optimized version. The wall-tessellation precision story is already a strong narrative without Phase 3.

---

## 2. Test Harness (BUILD THIS FIRST — 2 hours)

**Everything else in this plan calls into the harness built in this section. Do not skip ahead. Do not partially build it. Once Section 2 is done, every optimization in Section 3 reduces to:**

```powershell
# code change committed, then:
.\verify.ps1 -Variant <name>
```

**That single command rebuilds, runs the quick benchmark, computes SSIM/PSNR vs reference, captures Nsight metrics, and either says PASS or tells you what regressed. If PASS, the optimization is done and the data is saved. If FAIL, revert.**

This harness is the most important thing in the whole plan. Time spent here pays back 5× across the optimization stack.

### 2.1 Architecture

Five pieces, each callable independently but designed to compose:

```
verify.ps1                  ← top-level entry point. Calls all four below in order.
├── recompile.ps1           ← clean/incremental build
├── run_benchmark.ps1       ← runs configs, parses timings, writes CSV
├── run_nsight.ps1          ← optional Nsight pass (skip on -SkipNsight)
└── scripts/compare.py      ← SSIM/PSNR vs reference, called per-render
```

All scripts live in `phase2/`. All output lands in `phase2/results/runs/<timestamp>_<variant>/`.

### 2.2 `phase2/recompile.ps1`

A one-command rebuild that:

- Takes optional `-Scene <name>` (default `default`). Calls `select_scene.ps1` to copy `scenes/<name>.h` → `scene.h`.
- Takes optional `-Clean` switch — if set, deletes `build/` first.
- Takes optional `-Config <Release|Debug>` (default Release).
- Runs `cmake -G "Visual Studio 17 2022" -A x64` if `build/` doesn't exist.
- Runs `cmake --build build --config <Config> -- /p:CL_MPCount=8` for parallel build.
- On success, prints the exe path and PTX path so the next script knows where they are.
- Exits non-zero on any failure with the failing log dumped to stderr.

**Acceptance:** `.\recompile.ps1` on a clean machine ends with `optix_smallpt.exe` and `shaders.ptx` ready to run. `.\recompile.ps1 -Scene foo` ends with a build of the foo scene.

### 2.3 `phase2/run_benchmark.ps1`

Adapted from the existing `run_phase2_benchmark.ps1`. Same per-run-folder structure. New behaviors:

- `-Variant` parameter (required): tag like `baseline`, `rr`, `rr_tess`, `rr_tess_tile`. Goes in the run folder name and CSV.
- `-Configs` parameter: `quick` | `full` | `mem` | regex. **`quick` runs 3 configs in <60 s**; `full` runs the existing grid; `mem` runs only the 4K config that exercised 18 GB.
- `-RunDir` parameter (optional): if passed, write into the given directory instead of creating a new timestamped one. **This is the critical knob that lets `verify.ps1` collect benchmark + nsight + sanity outputs in the same folder.**
- After each render: calls `scripts/compare.py` against `phase3/reference/4096x3072_4096spp.png` (downscaled to match render resolution). SSIM and PSNR appended to the CSV row.
- CSV schema: `variant,scene,resolution,spp,time_ms,mrays_sec,ssim,psnr,gpu_mem_peak_mb,run_timestamp,notes`.
- `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` polled every 500 ms during render via background job; peak captured per config.

**`quick` config set** (this is what `verify.ps1` calls):
```
512x384_64spp        # smoke test, ~1 s
1024x768_256spp      # the headline regression target, ~5–10 s
2048x1536_256spp     # catches resolution-dependent bugs, ~30 s
```

**`full` config set:** the existing grid in `run_phase2_benchmark.ps1` (9 configs).

**Acceptance:** `.\run_benchmark.ps1 -Variant baseline -Configs quick` produces a run folder with `timings.csv` containing 3 rows, every column filled.

### 2.4 `phase2/run_nsight.ps1`

Wraps `ncu` for a small high-value set. Nsight runs are 10–60× slower than the underlying render, so do not profile the full grid.

- `-Variant` and `-RunDir` like the benchmark script.
- `-Mode` parameter: `quick` (one targeted-metrics pass on `1024x768_64spp`, ~2 min) | `full` (`--set full` capture on 3 configs, ~10 min). `verify.ps1` defaults to `quick`; the explicit baseline lock-down (Section 2.7) uses `full`.
- Targeted metrics list (Mode `quick`):
  ```
  gpu__time_duration.sum
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  smsp__warps_active.avg.pct_of_peak_sustained_active
  smsp__thread_inst_executed_per_inst_executed.ratio   # divergence proxy
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__throughput.avg.pct_of_peak_sustained_active
  lts__t_sectors_lookup_hit.sum
  lts__t_sectors.sum
  sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active   # XU pipe (RT-related on Ampere)
  smsp__pcsamp_warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active
  smsp__pcsamp_warps_issue_stalled_short_scoreboard.avg.pct_of_peak_sustained_active
  ```
- Output: `<RunDir>/nsight/<config>_metrics.csv` (Mode quick) or `<config>_full.ncu-rep` (Mode full).
- Also copies active `scene.h` and current git hash into the nsight folder.

**Permissions note:** First `ncu` invocation may fail with `ERR_NVGPUCTRPERM`. One-time fix: NVIDIA Control Panel → Developer → Manage GPU Performance Counters → "Allow access to all users." Reboot if it doesn't take effect.

**Acceptance:** `.\run_nsight.ps1 -Variant baseline -Mode quick` produces `<run>/nsight/1024x768_64spp_metrics.csv` opening cleanly in Excel.

### 2.5 `phase2/scripts/compare.py`

Tiny Python helper called by `run_benchmark.ps1`. Skimage-based.

```
python compare.py <render.png> <reference.png>
# stdout (one line):  SSIM=0.987 PSNR=34.2
```

Behavior:
- Auto-resizes reference to match render dimensions (Lanczos).
- Returns SSIM ≥ 0 even on mismatched sizes/format — never crashes the pipeline.
- Has a `--threshold` flag: exits non-zero if SSIM falls below it. `verify.ps1` uses this for hard regression detection.

`pip install scikit-image numpy pillow`. WSL2 is fine if Windows Python is annoying.

### 2.6 `phase2/verify.ps1` — the unified entry point

This is the script every later step calls. Build it last, after 2.2–2.5 work standalone.

**Behavior:**

```powershell
.\verify.ps1 -Variant <n> [-SsimThreshold 0.99] [-FullNsight] [-Tag <text>] [-SkipNsight]
```

Steps in order:

1. Run `recompile.ps1` (incremental). On failure: print build log tail, exit non-zero.
2. Create `runs/<timestamp>_<variant>/` (the run dir threaded into all subsequent calls via `-RunDir`).
3. Snapshot `scene.h` and current git commit hash into the run dir. Write `verify_info.txt` with variant name, SSIM threshold, host, GPU, driver version.
4. Run `run_benchmark.ps1 -Variant <n> -Configs quick -RunDir <runs/...>`.
5. Read every row of the resulting `timings.csv`. **If any SSIM < threshold: print FAIL, list the offending configs and their SSIM, exit non-zero.**
6. Unless `-SkipNsight` was passed: run `run_nsight.ps1 -Variant <n> -Mode quick -RunDir <runs/...>` (or `-Mode full` if `-FullNsight`).
7. Print PASS, the headline Mrays/s and SSIM for `1024x768_256spp`, and the run folder path. Exit 0.

**Crucial properties:**

- **Idempotent and self-contained.** After every commit during the optimization stack, the only command you run is `.\verify.ps1 -Variant <whatever>`. No manual benchmarking, no manual SSIM check, no manual Nsight invocation.
- **Returns deterministic exit codes.** PASS = 0, regression = 1, build failure = 2, harness bug = 3. Lets you wire it into git hooks later if you want.
- **Reusable for every optimization in Section 3 and beyond.** Adding RR, sphere tess, tile launches, NEE, denoiser — all use the same harness. Adding a future scene is a single `-Variant <new_scene>_baseline` invocation.

**Acceptance criteria for the whole harness (this is the gate before Section 3):**

- [ ] `.\verify.ps1 -Variant baseline` runs end-to-end in < 5 minutes from a clean tree.
- [ ] On a deliberately broken commit (force-clobber a render to black), it prints FAIL and points at the offending config.
- [ ] On the existing untouched code, it prints PASS and writes a complete run folder with timings + nsight metrics + SSIM/PSNR + scene snapshot + git hash.
- [ ] The `-Tag` flag appears in the run folder name so multiple verify runs at the same variant don't collide.

### 2.7 Locking down the baseline

**Before any optimization changes:**

1. `.\verify.ps1 -Variant baseline -FullNsight -Tag freeze`. Then layer on the full benchmark grid and full Nsight pass into the *same* run folder:
   ```powershell
   .\run_benchmark.ps1 -Variant baseline -Configs full -RunDir <baseline run dir>
   .\run_nsight.ps1   -Variant baseline -Mode full   -RunDir <baseline run dir>
   ```
2. Note wall-time and Mrays/s for `1024x768_1024spp` — this is the headline number for the deck.
3. Commit. Tag `optimized-baseline-frozen`.

**Do not touch `shaders.cu` or `main.cpp` until the baseline is frozen.**

---

## 3. Optimization Stack (sequential — `verify.ps1` between each)

After Section 2 is done, every step in this section follows the same loop:

1. Make the code change.
2. `git commit -m "<variant>: <what>"`
3. `.\verify.ps1 -Variant <variant>`
4. If PASS: continue to next step. If FAIL: `git revert HEAD` and try again.

The harness handles the SSIM/PSNR check, the timing CSV, and the Nsight capture automatically. **Do not run benchmarks or Nsight manually during this section** — that's how data inconsistency creeps in.

For the full benchmark grid + full Nsight pass per variant (needed for the deck), append:

```powershell
.\run_benchmark.ps1 -Variant <v> -Configs full -RunDir <runs/...>
.\run_nsight.ps1   -Variant <v> -Mode    full -RunDir <runs/...>
```

into the same run folder `verify.ps1` just created. Or do this only at the end for the variants that ship to the deck.

### Order of operations

| # | Optimization | Est. effort | Expected gain | SSIM target |
|---|---|---|---|---|
| 1 | Russian Roulette | 30 min | 10–25% time reduction | ≥ 0.99 vs baseline |
| 2 | Bounce-depth cap probe (analysis only, no code) | 30 min | informs cap choice | n/a |
| 3 | Sphere tessellation (= old Phase 3) | 2 hr | -10% to +10% (uncertain) | ≥ 0.99 vs baseline |
| 4 | Tile launches | 1 hr | memory drops to <2 GB at 4K | identical |
| 5 | NEE (optional) | 2 hr | 4–10× variance reduction → fewer SPP | match at sufficient SPP |
| 6 | Denoiser (optional) | 1 hr | 1-spp output usable | n/a (different metric) |

### 3.1 Russian Roulette

Edit `shaders.cu` closest-hit. After computing `throughput` for the bounce, if `prd.depth >= 5`:

```
float p_continue = max(throughput.x, throughput.y, throughput.z);   // luminance approx
if (rnd(seed) >= p_continue) { prd.done = 1; return; }
prd.throughput *= 1.0 / p_continue;   // unbiased compensation
```

Notes:
- Threshold depth 5 matches smallpt and cu-smallpt. Confirm cu-smallpt's exact threshold by checking `phase1/src/kernel.cu` before settling — match it for fair comparison.
- The fixed `< 20` bounce loop in raygen stays as a hard ceiling but RR will terminate most paths well before then.
- Move `prd.depth` increment to happen at the right point in the loop (probably already done).

**Verify:** `.\verify.ps1 -Variant rr`. Harness will fail if SSIM < 0.99 (typical RR bias signature). If it fails: the throughput compensation `*= 1.0 / p_continue` is missing or in the wrong place. Re-derive against smallpt's `radiance()`.

### 3.2 Bounce-depth ablation (analysis, no permanent change)

Temporarily expose `MAX_DEPTH` as a CLI flag (or just hardcode and rebuild — faster). Run:

```
for depth in 1, 2, 3, 4, 6, 8, 12, 20:
    render 1024x768_1024spp, log time and SSIM vs baseline
```

Plot time-vs-depth and SSIM-vs-depth. Pick the smallest depth where SSIM stays ≥ 0.99 (probably 6–8 for Cornell box). **Do not bake the lower cap in permanently** for the optimized version — RR handles termination correctly. The plot is for the deck.

### 3.3 Sphere tessellation (= old Phase 3)

This is the largest engineering chunk. Reference: existing wall tessellation in `scenes/scene_default.h`.

**Implementation outline:**

- New helper `tessellate_sphere(center, radius, lat, lon, verts_out, tri_idx_out)` in `scene_default.h`. Lat/lon UV sphere, default 64×64 = 8192 triangles per sphere.
- Replace `g_spheres` array with a list of tessellated meshes. SBT layout: each sphere's triangles share one hit group record via `sbtIndexOffsetBuffer`, just like walls.
- Move the 3 sphere meshes into the existing triangle GAS (single GAS). Drop the second sphere GAS. Drop the IAS in favor of single-GAS-as-traversable. (Or keep IAS for future flexibility — slightly slower but architecturally cleaner. Ship single-GAS for the deck unless it costs > 30 min.)
- Update `pipelineCompileOptions.usesPrimitiveTypeFlags` to TRIANGLE only.
- Drop the built-in sphere IS module and its hit-group variant.
- Simplify `compute_normal()` in `shaders.cu`: remove the sphere branch. Single triangle face-normal path.
- Update `traversableGraphFlags` if dropping the IAS: switch to `ALLOW_SINGLE_GAS`.

**Risk:** glass and mirror at 8K triangles will *look slightly different* from analytic spheres (faceting at edges). Bump to 128×128 if needed. Higher = slower BVH build but better looking.

**Verify:** `.\verify.ps1 -Variant rr_tess -SsimThreshold 0.98` (looser threshold because sphere geometry actually changed). Then visual inspection of glass refraction + mirror reflection — facet artifacts? If yes, bump tessellation finer and re-verify.

**If this slips by 2+ hours, abort and revert. The presentation is fine without it.**

### 3.4 Tile-based launches

The 18 GB memory issue is per-launch state scaling with launch size. Fix by tiling.

**Implementation:**
- In `main.cpp`, replace the single `optixLaunch(W, H)` with a loop over tiles (e.g. 512×512 each).
- Pass tile origin as part of `Params`. Raygen offsets its pixel index by tile origin.
- Accumulation buffer is full-size; tiles write to their slice.
- Loop over tiles for one launch worth of work, then advance `subframe_index`.

**Verify:** `.\verify.ps1 -Variant rr_tess_tile`. Then for the memory claim specifically:

```powershell
.\run_benchmark.ps1 -Variant rr_tess_tile -Configs mem -RunDir <run dir from verify>
```

The `mem` config (`4096x3072_1024spp`) should now show GPU peak memory < 2 GB in the CSV. Time may go up or down vs untiled — both are fine, log it.

### 3.5 NEE — optional, do only if 3.1–3.4 finish with ≥ 4 hours left

Direct light sampling against the area light (the Cornell light is the radius-600 sphere protruding through the ceiling — though after 3.3 it's a triangle mesh).

For the sphere/triangle-mesh light:
- Sample a random triangle on the light mesh (or a point on the analytic disc — simpler if you keep the analytic light). Actually: easier to sample the bounding disc (ceiling cutout) since the visible portion is approximately a disc.
- Cast a shadow ray. If it hits the light, add `throughput * Le * geometric_term * pdf_inverse` to radiance.
- Add MIS weights or do the simpler "NEE only at diffuse hits, regular path tracing for specular" — the latter is cleaner for 1-day implementation.

**Risk: lots of math to get right, easy to introduce bias. NEE bias only shows up with sufficient SPP — the harness's `quick` configs may pass while a 1024-spp render visibly biases. After `verify.ps1 -Variant rr_tess_tile_nee` passes, also run `run_benchmark.ps1 -Configs full -RunDir ...` and check SSIM at `1024x768_1024spp` ≥ 0.98.**

**If NEE bug found: revert. Don't ship a buggy optimization.**

### 3.6 Denoiser — optional

OptiX `OptixDenoiser`. Basic LDR mode, no albedo/normal AOVs (those are the next step up).

- Initialize denoiser with image dimensions during context setup.
- After accumulating samples, run denoiser on the accumulation buffer.
- Output the denoised result alongside the raw output for comparison.

**Use case for the deck:** render at 1, 4, 16 spp, denoise, compare visually to 1024 spp ground truth. The "1 spp denoised vs 1024 spp raw" image is a *very* good slide.

---

## 4. What to Measure (the data the deck depends on)

For each variant in `{baseline, rr, rr_tess, rr_tess_tile, [+nee, +denoiser]}`:

### 4.1 Benchmark grid (run via `run_benchmark.ps1 -Configs full`)

Existing grid is fine. The cross-variant CSV will be the centerpiece.

### 4.2 Nsight metrics (run via `run_nsight.ps1`)

3 configs × 5 variants = 15 captures. Manageable.

### 4.3 Targeted experiments (one-offs)

Run these *once* for the deck:

| Experiment | Purpose | Time |
|---|---|---|
| Bounce depth ablation (Section 3.2) | Justify max-depth choice and show per-bounce cost | 20 min |
| Memory scaling: 1080p, 1440p, 2160p, 3K, 4K at 256 spp, single-launch | Show the 18 GB problem before tiling, then re-run after | 30 min |
| Strong scaling: hold pixels constant (`1024x768`), vary spp = 64, 256, 1024, 4096 | Show throughput stability (Mrays/s flat) | 20 min |
| Weak scaling: hold spp constant (256), vary resolution from 512x384 to 4096x3072 | Confirm linear scaling in pixels | 20 min |
| Cross-variant ssim trace (`baseline.png` vs `rr.png` vs `rr_tess.png` vs ...) | Demonstrate equivalence at each stage | 10 min |

### 4.4 Phase 1 comparison

Phase 1 cu-smallpt rerun on Windows is a separate task (you said you'd handle it in another conversation). The optimization plan needs **one** number from Phase 1: Mrays/s on the headline config (`1024x768_256spp`) on Windows native, with `nvidia-smi` confirming the same RTX 3080 Ti is in use. This goes into the cross-phase summary table in slide 3.

---

## 5. Correctness Verification

**The harness handles this automatically.** `verify.ps1` runs after every commit, fails the build on regression, and writes the artifact trail to a per-run folder. Do not check correctness manually.

The only manual checks worth doing are:

1. After `rr_tess` (sphere tessellation): open the rendered PNG and inspect glass refraction + mirror reflection edges for visible facets.
2. After `rr_tess_tile_nee` (if NEE landed): visually compare 1024-spp render to baseline 1024-spp render — bias from NEE shows up as overall brightness drift in indirect-lit regions (e.g. the floor under the glass sphere).
3. After every variant ships: glance at the run folder to confirm `timings.csv`, `nsight/*.csv`, scene snapshot, and git hash are all present.

---

## 6. Slide Outline (8 slides, 6 minutes ≈ 45 sec/slide)

This is the path tracer talk you've spent the project building toward. Lean on visuals; the audience won't read text.

### Slide 1 — Problem & algorithm (45s)

- One sentence: "Render a Cornell box via Monte Carlo path tracing, on the GPU, fast."
- The smallpt algorithm in 2 bullets: unidirectional MC path tracer, recursive radiance with diffuse/specular/refractive materials, RR termination.
- Complexity: O(W·H·SPP·d·N) — pixels × samples × bounces × per-ray intersect cost.
- Visual: a 4-bounce path traced through the Cornell box (small diagram).

### Slide 2 — Three implementations, three architectures (45s)

A 3-column table:

| | smallpt | cu-smallpt (Phase 1) | OptiX optimized (Phase 2+) |
|---|---|---|---|
| Hardware | Single CPU core | CUDA cores (no RT cores) | RT cores + CUDA cores |
| Intersection | Software ray-sphere | Software ray-sphere | Hardware ray-triangle |
| Acceleration | None (linear scan) | None (linear scan) | Hardware BVH |
| Mrays/s | <0.1 | ~35 | ~300+ |

### Slide 3 — The OptiX pipeline + RT core diagram (45s)

Two diagrams side-by-side:
- **Left:** OptiX pipeline graph (raygen → traversal → IS/CH/AH → miss). Color-code which boxes execute on RT cores vs SM cores.
- **Right:** Ampere SM block diagram with the RT core highlighted. Explain "BVH traversal + ray-triangle test = fixed-function silicon."

**This slide is the "why is it faster" answer. Spend 10 extra seconds here.**

### Slide 4 — Engineering challenges (45s)

The 1e5-radius wall sphere precision saga (3 images: phase 1 sphere walls / phase 2 1e5 banding / phase 2 1e4 isocontours). Wall tessellation as the fix. One sentence on the ceiling banding bug → wall tessellation grid.

This sells the project as actual engineering, not just plumbing.

### Slide 5 — Optimization stack (60s)

Bar chart: Mrays/s for `baseline → rr → rr_tess → rr_tess_tile → [nee → denoiser]`. Each bar labeled with what was added.

Below: time per frame at 1080p, same configurations. Audience sees the stack of wins.

### Slide 6 — Where does the time go? (45s)

Nsight breakdown for the optimized version:
- Pie chart of stall reasons OR a bar chart of: warp occupancy, RT core utilization, memory bandwidth utilization.
- One callout: "ray divergence at deep bounces is the remaining bottleneck."
- Mention: SER (Ada+) would address this; we're on Ampere.

### Slide 7 — Image quality + memory (45s)

Top row: side-by-side renders at equal SPP (smallpt vs cu-smallpt vs optimized). They should look identical. SSIM numbers underneath.

Bottom row: memory plot. Single-launch peak GPU memory at 4K (the 18 GB problem) vs tiled (< 2 GB).

If the denoiser is in: a third row showing 1 spp + denoiser vs 1024 spp ground truth.

### Slide 8 — What's next + summary (30s)

Brief:
- Achievements: 9× over Phase 1, all-RT-core scene, memory-bounded launches.
- Future work: NEE/MIS, wavefront restructure, ReSTIR for real-time, SER on Ada+.
- Closing line: "Real-time path tracing is one denoiser away from this codebase."

### Slide build checklist

- All renders go in `phase2/results/<run>/renders/` with consistent naming.
- All charts: matplotlib via the `compare.py` helper, exported to PNG, dropped into PowerPoint.
- Nsight screenshots captured from the `--set full` `.ncu-rep` files (Nsight Compute GUI → File → Export image of the section view).

---

## 7. Risks and Decision Points

| Risk | Trigger | Mitigation |
|---|---|---|
| Sphere tessellation breaks the build for >2 hr | Section 3.3 starts before lunch, still failing by mid-afternoon | Revert. Ship `rr_tess_tile` without tessellation, lean on the wall-tessellation story for the geometric depth. |
| RR introduces visible bias | SSIM < 0.99 after Section 3.1 | Most likely the throughput-compensation `*= 1/p` is missing or in the wrong place. Re-derive against smallpt's `radiance()`. |
| NEE consumes 4+ hours | Section 3.5 still buggy after lunch | Cut. Ship without NEE. NEE on the "future work" slide is fine. |
| Nsight Compute permission errors on Windows | First `ncu` call returns "ERR_NVGPUCTRPERM" | Run elevated PowerShell. NVIDIA Control Panel → Developer → Manage GPU Performance Counters → "Allow access to all users." |
| 18 GB memory issue persists after tiling | Unexpected | Profile tile-loop allocations. Likely one allocation that doesn't scale with tile size — search for full-resolution buffers other than `accum_buffer`. |
| Phase 1 Windows port fails (separate conversation) | Phase 1 numbers unavailable for slide 2 | Cite WSL2 numbers, note in the deck "OS difference negligible since cu-smallpt is compute-bound; ±5%." |

---

## 8. Definition of Done

The optimized version is shippable for the deck when:

- [ ] `recompile.ps1`, `run_benchmark.ps1`, `run_nsight.ps1` all work on the headline config without manual intervention.
- [ ] Variants `baseline`, `rr`, `rr_tess` (or `rr_tile` if tessellation dropped), `rr_tess_tile` all have benchmark CSV + Nsight metrics CSV in `runs/`.
- [ ] SSIM ≥ 0.98 across the optimization chain (relaxed for the geometry-changed step).
- [ ] One memory plot showing tile launches fix the 18 GB issue.
- [ ] One bar chart showing Mrays/s improvement across the variant chain.
- [ ] One image-quality side-by-side at matched SPP.
- [ ] All 8 slide assets exported as PNG and dropped into the deck.

---

## 9. Future Work / Aspirational (do not start unless explicitly told)

**Ada+ only:**
- **Shader Execution Reordering (SER)** — call `optixReorder()` after the trace in raygen, sort rays by hit material before continuing. On Ada/Hopper this regroups warps and wins back ~30% from divergence. **Strict requirement: RTX 6000 Ada or newer (RTX 4000 series, L40, etc.). Not available on RTX 3080 Ti.** If SCC RTX 6000 Ada becomes available, this is the single most valuable optimization remaining.

**Ampere-compatible but expensive:**
- **Wavefront path tracer** — split the megakernel raygen into stages (generate, intersect, shade, compact) with sort-by-material between stages. Production-grade. Multi-day implementation.
- **MIS (Multiple Importance Sampling)** — combine BSDF sampling with NEE light sampling. Standard variance reduction.
- **ReSTIR DI/GI** — temporal sample reuse for real-time. Multi-week implementation but landing point for actual real-time path tracing.
- **Proper QMC sequences (Sobol/Halton)** — better convergence per spp than uniform random.

These all become candidate Phase 4 directions for follow-up work.

---

## 10. Hand-off Notes for Next Conversation

If this plan needs to be picked up cold:

- Start by reading `phase2/docs/STATUS.md` for state of the codebase.
- This file (`optimization_plan.md`) is authoritative; `plan.md` is historical.
- The reference render is at `phase3/reference/4096x3072_4096spp.png`.
- No code in this plan; all code goes in commits and gets tagged. Don't re-read ancient commits — read HEAD.
- Open question if it comes up: did the Phase 1 Windows port get done? Mrays/s number for `1024x768_256spp` on Windows is the only Phase 1 datum the deck needs.
