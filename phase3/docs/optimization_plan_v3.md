# EC527 Phase 3 — Final Push Plan (v3)
## Finishing the project + answering the professor's slide questions

**Owner:** Hudson Reynolds
**Course:** EC527 (BU, Spring 2026)
**Plan version:** v3, supersedes v2.
**Status going in:** phase 3 builds, renders, runs at 564–589 Mrays/s wall-clock across the 3-config quick set on RTX 3080 Ti. Nsight Compute capture works but names the kernel `optixLaunch` (not raygen) on Ampere, which limits some metrics.

### Changes in v3 vs v2

1. The phase 3 sphere-tess + tiling change is **shipped** (not "to do"). v3 is about finalization, not new variants.
2. Replaces "Build the test harness" with "Extend the harness to answer the professor's specific questions." Most of the harness exists.
3. Adds Phase 1 / phase 2 / phase 3 comparison data collection as a first-class deliverable.
4. Adds the Nsight Compute caveat explicitly (kernel named `optixLaunch`, deprecated-launcher path) and what the deck says about it.
5. NEE and bounce-depth ablation are in scope as half-day work. Denoiser is on the future-work slide.

---

## 0. The professor's questions and where each one gets answered

| Professor question | Answer source | Slide |
|---|---|---|
| Description of the problem | Cornell-box Monte Carlo path tracing | 1 |
| Serial code / algorithm | smallpt 99-line + OpenMP-multithreaded smallpt → cu-smallpt → OptiX | 2 |
| Complexity | O(W·H·SPP·d·log N) with BVH, vs O(W·H·SPP·d·N) linear | 2 |
| Where does time go | Wall-clock: BVH build / tile loop / readback. Inside the launch: Nsight DRAM 46% vs SM 23% (memory-leaning bound) | 6 |
| Arithmetic intensity | Per-bounce shading FLOPs / bytes-touched, plus DRAM/SM ratio from Nsight | 6 |
| Primary data structures | Triangle GAS (BVH), float4 accum buffer, SBT, per-ray PRD, HitGroupData | 3 |
| Memory reference pattern | Coherent for primary rays, divergent for secondary; accum buffer streamed once per pixel; SBT pulled per hit | 3 |
| Modified the algorithm to run in parallel? | Yes — pixels are embarrassingly parallel; recursion → iterative bounce loop because OptiX disallows recursive trace | 4 |
| How were data + computation partitioned? | (Pixel × spp) → CUDA threads through OptiX raygen; tiled across host launches for memory bounding | 4 |
| Optimizations + problems | Wall tess (precision), sphere tess (single GAS), GAS compaction, DISABLE_ANYHIT, RR, payload-as-pointer, tiled launches | 5 |
| Experiments + results | Mrays/s bar chart, memory plot, image-quality side-by-side | 7 |

This table is the source of truth for the deck. Everything in the plan below feeds it.

---

## 1. System Context

### Hardware / OS

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 3080 Ti (SM 8.6, Ampere, 80 SMs, 12 GB GDDR6X) |
| Driver | 596.21 |
| OS | Windows 11 (build, run, profile all native — no WSL2) |
| CUDA Toolkit | 12.8 |
| OptiX SDK | 8.1.0 |
| Host compiler | MSVC 19.44 (VS 2022 Build Tools) |
| Arch | `-arch=sm_86` |

**Key Ampere fact for the deck:** No SER (Ada+ only). `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` is set during ncu capture; on Ampere this changes nothing observable about perf because there's no SER to disable.

### What's shipped (phase 3 delta vs phase 2)

| Optimization | Status | Notes |
|---|---|---|
| Russian Roulette (depth>4, max-of-albedo) | shipped, was in phase 2 | inherited |
| Wall tessellation (16×16 + skirts) | shipped, phase 2 | precision fix for ceiling banding |
| Iterative bounce loop (no recursion) | shipped, phase 2 | OptiX requirement |
| Payload pointer split (2× uint32) | shipped, phase 2 | inherited |
| `DISABLE_ANYHIT` per geometry flag | shipped, phase 2 | |
| GAS compaction | shipped, phase 2 | `OPTIX_BUILD_FLAG_ALLOW_COMPACTION` |
| `PREFER_FAST_TRACE` | shipped, phase 2 | |
| **Sphere tessellation (256×256 lat/lon)** | **shipped, phase 3** | mirror, glass, light → triangles |
| **Single triangle GAS, no IAS** | **shipped, phase 3** | `traversableGraphFlags = ALLOW_SINGLE_GAS` |
| **TRIANGLE-only primitive type flag** | **shipped, phase 3** | drops sphere intersection module |
| **Tile-based launches (512×512)** | **shipped, phase 3** | bounded per-launch state |

### What's NOT shipped (and why)

| Item | Status | Rationale |
|---|---|---|
| NEE / MIS | not shipped | Half-day budget could hit it. §3 below decides go/no-go. |
| OptiX denoiser | not shipped | Future-work slide. |
| Bounce-depth ablation | not shipped | 30-min experiment. §4 schedules it. |
| Wavefront restructure | not shipped | Multi-day. Future work. |
| SER | not applicable | Ada+ only. |
| SBT consolidation (9 records → 1 + material LUT) | not shipped | Marginal win; ICache pressure not measured. Future work. |

### Current numbers (3-config quick set, sphere tess 256×256)

| Resolution | SPP | Time (ms) | Mrays/s |
|---|---|---|---|
| 512×384 | 64 | 21.36 | 589.08 |
| 1024×768 | 256 | 352.82 | 570.62 |
| 2048×1536 | 256 | 1427.86 | 564.00 |

vs phase 2 baseline: **~300 Mrays/s** at 1024×768_64spp. **Roughly ~1.9× over phase 2** at the headline config when the deck is built. (Need to confirm with a clean phase 2 run on the same machine — see §2.1.)

---

## 2. The "Nsight Compute on Ampere" honesty slide

### What's actually happening

When ncu runs your binary with `OPTIX_FORCE_DEPRECATED_LAUNCHER=1`, the kernel it sees and reports is named `optixLaunch`. This is the deprecated launcher path's outer kernel that wraps your raygen, not your raygen itself. Symptoms in your `1024x768_512spp_metrics.csv`:

- Time inflates from ~705 ms wall-clock to ~931 ms (4 tile launches summed: 269 + 349 + 162 + 149 ms). The 8545 ms reported in the log is wall-clock under ncu's replay-based metric collection — ncu replays each kernel several times and that's what blows wall-time up to 8.5 s.
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` = 23% — this is real for the launcher kernel but reflects launcher overhead, not your shader work.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` = 46% — probably trustworthy as a memory-system signal.
- `smsp__warps_active.avg.pct_of_peak_sustained_active` = 31% — real number, but for a kernel that includes launcher housekeeping.
- `smsp__pcsamp_warps_issue_stalled_*` = `n/a` — pc-sampling doesn't run under deprecated launcher capture. **This is the metric we'd most want and can't have on Ampere.**
- `smsp__thread_inst_executed_per_inst_executed.ratio` = 24.4 / 32 — moderate divergence, ~76% of threads active per warp on average. **Useful and trustworthy.**
- `sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active` = 4.6% — XU is the transcendental pipe (sqrt, sin, cos). Confirms not transcendental-bound.

### What the deck says

A short slide titled "Profiling caveat" that says: on Ampere with no SER, ncu sees the deprecated launcher kernel, not the raygen. Wall-clock and the metrics that make sense at the launcher level (DRAM throughput, thread-instructions ratio) are reported. PC-sampling stall reasons are unavailable on this hardware. SER on Ada+ would let ncu name the raygen kernel directly.

You can fold this into the "where does time go" slide if 6 minutes is tight.

### Worth a quick try (10 minutes)

A single `--replay-mode application` ncu run instead of the default kernel replay. Application replay re-runs the whole exe per metric pass, which sometimes captures the OptiX raygen more cleanly. Worth one shot:

```powershell
ncu --replay-mode application --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all -- .\build\Release\optix_smallpt.exe --width 1024 --height 768 --spp 64 --output replay_test.ppm --ptx .\build\shaders.ptx
```

If the kernel comes through with a different name and you get pc-sampling, great. If not, ship with the caveat slide.

---

## 3. Decision: NEE go/no-go (~30 min decision, half-day implementation)

NEE (Next Event Estimation) at every diffuse hit samples the light directly with a shadow ray, and weights the BSDF sample correctly. For Cornell box with one big light, this is a **very large variance reduction** — 4× fewer samples for the same noise is plausible.

### Why it might pay

Your light is a single sphere at the top. Direct lighting dominates the image. Every diffuse bounce is currently sampling cosine-weighted hemisphere and hoping to hit the light eventually. NEE replaces that with: shoot a shadow ray to a point on the light, MIS-weight, continue path with BSDF sample for indirect.

### Why it might not pay (in 6 minutes of slide time)

- The deck is graded on optimizations, and NEE is more of a *variance-reduction* technique than a per-ray-faster optimization. The story "NEE reduces noise per sample at the cost of extra rays per bounce" is harder to fit in 30 seconds than "we tessellated the spheres into the GAS."
- One bias risk per implementation pass.
- The current optimization story (Mrays/s 35 → 300 → 570) is already strong.

### Decision rule

**Implement NEE only if §4's data collection finishes by lunch.** Otherwise ship without NEE, mention it on the future-work slide, and use the freed time to run the bounce-depth ablation and to polish the deck.

### If implementing

A single CH-shader change (40 lines, in `shaders.cu`):

1. At every diffuse hit, sample a point on the light sphere (uniform-area sample on the visible hemisphere from the hit).
2. Cast a shadow ray with `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT` and a separate "any-hit-only" ray type. Or reuse the main ray type and gate on `prd->shadow_only` flag.
3. If unoccluded, add `throughput * brdf * Le * geom_term / pdf` to radiance.
4. Continue the path with the existing BSDF sample (no MIS for now — use balance heuristic later if time permits, or just use NEE-only for direct + BSDF for indirect).

**A separate ray type adds a second hit-group set and a second miss program — that's a bigger change than one CH-shader patch.** Cleaner option: use the same ray type, set `prd->is_shadow=1`, in CH check that flag and immediately set `prd->done=1` if hit, otherwise let miss set radiance. The shadow ray's `tmax` is set to just-short-of-light to avoid the light occluding itself.

Acceptance criteria: SSIM ≥ 0.97 vs reference at matched SPP; visible noise reduction in 64-spp render; Mrays/s drops by ~30–50% but image quality wins.

If NEE breaks: revert. Don't sink half a day.

---

## 4. Data collection (this is the bulk of the work — 4 hours)

The current `timings.csv` has only phase 3 numbers. The deck needs phase 2 baseline numbers in the same CSV format, on the same machine, captured the same way.

### 4.1 Phase 2 baseline rerun (30 min)

Build phase 2 fresh (separate `build/` from phase 3) and run the same 3 quick configs. Append to a new `phase2_timings.csv` with the same schema. This is the bar-chart input for slide 5.

```powershell
cd C:\Users\hudsonre\527project\phase2
powershell -ExecutionPolicy Bypass -File .\select_scene.ps1 -Name default
powershell -ExecutionPolicy Bypass -File .\run_phase2_benchmark.ps1
```

If `run_phase2_benchmark.ps1` writes to `results\phase2_optix_baseline\runs\<ts>\timings.csv`, copy that CSV alongside phase 3 for plotting.

**Acceptance:** A CSV row for `optix_phase2,default,1024x768,256,<ms>,<Mrays/s>,sm_86,<ts>` exists, and the time is in the 400–500 ms range (phase 2 baseline at ~300 Mrays/s would put 1024×768_256spp at ~600 ms).

### 4.2 Phase 3 full grid (1 hour)

Right now you have `quick` (3 configs). Run `-Full` to get the 9-config grid, including 4K runs that test memory.

```powershell
cd C:\Users\hudsonre\527project\phase3
powershell -ExecutionPolicy Bypass -File .\run_phase3_benchmark.ps1 -Full
```

This gives you weak-scaling (resolution at fixed spp) and strong-scaling (spp at fixed resolution) data for the deck.

**Acceptance:** 9 rows in `timings.csv`. Mrays/s should be flat-ish across configs (~550–590) — confirms the renderer is throughput-stable.

### 4.3 Memory measurement at 4K (30 min)

The 18 GB → "<2 GB" claim from v2 needs an actual number. The exe doesn't print peak GPU memory. Quickest path: run the 4K config and watch nvidia-smi.

Add to `main.cpp` (4 lines, before exit):

```cpp
size_t free_b = 0, total_b = 0;
cudaMemGetInfo(&free_b, &total_b);
fprintf(stderr, "GPU memory: %zu MB used / %zu MB total\n",
        (total_b - free_b) / (1024*1024), total_b / (1024*1024));
```

This reports right before cleanup, which is when peak should be near max. Run the 4K_1024spp config and write the number on the memory-plot slide. If the number is too low because allocations were already freed, instrument inside the tile loop instead.

Alternative: in another terminal during the 4K run, `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > mem_log.csv`. Take max.

**Acceptance:** A peak-memory number for phase 3 at 4K_1024spp. Compare to phase 2's documented 18 GB.

### 4.4 Bounce-depth ablation (30 min)

Cheapest "extra data" win for the deck. Add a `--max-depth` CLI flag to `main.cpp` (default 20) and parameterize the bounce loop. Re-run 1024×768_1024spp at depth ∈ {2, 4, 6, 8, 12, 20}. Plot Mrays/s and SSIM-vs-reference. RR makes high depth nearly free; the curve will plateau around 6–8.

```cpp
int max_depth = 20;
// add to argv parsing: --max-depth N
// pass to params via Params::max_bounces, used in raygen loop instead of literal 20
```

**Acceptance:** 6 data points, plotted on slide 5 or 6. The plateau visualizes "RR makes the cap cosmetic."

### 4.5 Per-phase wall-clock breakdown (15 min)

The exe currently times only the tile loop. The deck wants "where does the time go," so wrap each phase in `cudaEvent_t` pairs:

- BVH build (lines 224–234 of `main.cpp`).
- Compaction (lines 232–234).
- SBT construction + accum buffer alloc (335–384).
- Tile loop (already timed).
- Readback (468).

Print all five to stderr. Build is amortized over the whole render — for a 4K_1024spp run it's <2% of total. For a 512×384_64spp run it might be 20%. **Both numbers are slide-worthy.**

**Acceptance:** 5 timings printed per run. Pie chart on slide 6.

### 4.6 Nsight Compute, just the metrics that mean something (20 min)

Modify the metric set in `run_nsight.ps1` to drop the `n/a` ones and add the ones that actually answer slide questions on Ampere:

```powershell
$Metrics = @(
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "lts__t_sectors.sum",
    "lts__t_sectors_lookup_hit.sum",
    # New: arithmetic-intensity-relevant
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
) -join ","
```

The three new `smsp__sass_thread_inst_executed_*` metrics give you a real FLOP count for the SM-side work. Combined with `dram__bytes.sum` you get an empirical AI: FLOPs / bytes. This is the closest thing to a defensible AI number you can produce, even if it doesn't include RT-core work.

Re-run quick capture for both phase 2 and phase 3:

```powershell
cd C:\Users\hudsonre\527project\phase3
powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 -Variant phase2_baseline -ExePath ..\phase2\build\Release\optix_smallpt.exe -PtxPath ..\phase2\build\shaders.ptx -Mode quick
powershell -ExecutionPolicy Bypass -File .\run_nsight.ps1 -Variant phase3_tess_tile -ExePath .\build\Release\optix_smallpt.exe -PtxPath .\build\shaders.ptx -Mode quick
```

**Acceptance:** Two `*_metrics.csv` files, one per variant, with FLOP-count metrics populated.

### 4.7 Application-replay attempt (10 min, may fail, low risk)

One shot at getting raygen-named metrics:

```powershell
ncu --replay-mode application --kernel-name "regex:(?i)raygen" --metrics gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed --target-processes all -- .\build\Release\optix_smallpt.exe --width 1024 --height 768 --spp 64 --output replay_test.ppm --ptx .\build\shaders.ptx
```

Without `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` set, application replay may capture the modern OptiX launch path. If `No kernels were profiled`, set the env var and retry. If still no raygen, accept it and move on.

**Acceptance:** Either a working raygen capture (great — use it on slide 6), or a confirmed dead-end (caveat slide).

### 4.8 SSIM correctness check (15 min)

The `compare.py` referenced in v1 isn't in the uploads. Quick standalone:

```python
# scripts/ssim_compare.py
import sys
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
a = np.array(Image.open(sys.argv[1]).convert('RGB'))
b = np.array(Image.open(sys.argv[2]).convert('RGB'))
print(f"SSIM: {ssim(a, b, channel_axis=-1):.4f}")
```

Compare phase 3 1024×768_1024spp against phase 2's 1024×768_1024spp downsampled from `reference/4096x3072_4096spp.png`. Want SSIM ≥ 0.97 (relaxed because sphere geometry changed from analytic to tessellated).

**Acceptance:** SSIM number printed, image side-by-side ready for slide 7.

### 4.9 Phase 1 placeholder

You're rerunning phase 1 separately. The plan reserves a row in the deck's bar chart for phase 1 Mrays/s on this machine at `1024x768_256spp`. Drop the number in when you have it. Until then the slide can show "Phase 1 (cu-smallpt): TBD on RTX 3080 Ti, ~35 Mrays/s WSL2" with a note. Don't block the deck on this.

---

## 5. Final code changes (small, after data collection — 1 hour)

These are quality-of-life and "while you're in there" items. Don't touch the renderer until §4 finishes. Bottom-of-the-list polish:

### 5.1 Clean up the `is_sphere` analytic-normal branch (15 min, decide)

`compute_normal()` still returns the analytic outward normal for spheres because tessellation can show faceting on close-up reflections. With sphere tess at 256×256 (130k tris/sphere), faceting should be invisible from the camera position. Two options:

- **(a) Drop the analytic-normal branch entirely.** Pure triangle pipeline. Cleaner story for slide 3 ("everything is one triangle GAS"). Risk: silhouette edges or extreme glancing reflections show facets. Need to visually verify at 1024×768_1024spp.
- **(b) Keep the branch and explain on slide 5.** "We use the analytic normal at sphere hits even though the geometry is tessellated, because shading-normal precision matters more than geometry-normal precision for glossy surfaces."

**Recommended: (b).** Keep the branch. It's defensible engineering, it's what production renderers do (smooth normals on tessellated meshes are a thing), and dropping it risks visible facet artifacts you'd have to chase mid-data-collection.

### 5.2 Comment cleanup (5 min)

`shaders.cu` line 170 has `// changed from 130.0f to try to fix ghost ring, also tried gaze in place of d. neither worked.` — if the ghost ring is gone now (sphere tess might have eaten it), reword to `// origin = eye + d * 130.0f advances ray to image plane (smallpt convention)`. If still present, expand the comment with current understanding and note it as a known artifact for the slide deck.

### 5.3 Add `--max-depth` CLI flag (10 min)

Required for §4.4 ablation. Parameterize the literal `20` in raygen via a Params field.

### 5.4 Add memory measurement print (5 min)

`cudaMemGetInfo` call at the end of run, for §4.3.

### 5.5 Add per-phase event timers (15 min)

For §4.5. cudaEventRecord around BVH build, compaction, SBT setup, tile loop, readback. Print all to stderr.

### 5.6 Don't touch — known-good

Don't modify: tile loop, raygen, CH shader, sphere tessellation parameters, tile size. These work and the deck depends on them not regressing. **No "while I'm in there" optimizations during data-collection week.**

---

## 6. Updated test scripts

### 6.1 `run_phase3_benchmark.ps1` — additions

Already mostly good. Two adds:

1. **`-MaxDepth N` parameter** for the ablation. Pass through to exe as `--max-depth`.
2. **Memory log capture.** When config name contains "4096", spawn a background `nvidia-smi` logger and record peak.

These are the only changes. Don't re-architect.

### 6.2 `run_nsight.ps1` — additions

Already good. Three changes:

1. **Metric set updated** per §4.6.
2. **Add `-ApplicationReplay` switch** that adds `--replay-mode application` to the ncu invocation. For one-shot §4.7 attempt. Default off.
3. **Print derived AI** at the end of quick mode: `(2 * fadd + 2 * fmul + 4 * ffma) / dram__bytes.sum`. (FFMA counts as 2 FLOPs; fadd and fmul as 1 each — the "2 *" prefactor here is: sum the per-thread instruction count, which ncu reports as already-multiplied-by-active-threads.) Sanity-check the number is in the 0.5–10 FLOP/byte range.

### 6.3 New `scripts/ssim_compare.py`

Per §4.8. ~10 lines. Don't overthink it.

### 6.4 New `scripts/build_charts.py`

Reads phase2 + phase3 timings.csv, produces three PNGs:

1. `mrays_bar.png` — phase 1 / phase 2 / phase 3 Mrays/s at 1024×768_256spp.
2. `weak_scaling.png` — phase 3 Mrays/s vs resolution at fixed spp.
3. `bounce_depth.png` — phase 3 Mrays/s and SSIM vs max-depth.

matplotlib, ~80 lines total. Output to `results/charts/`.

---

## 7. Slide outline (7 content slides, 6 minutes ≈ 50 sec/slide)

### Slide 1 — Problem (40s)
- "Render a Cornell box via Monte Carlo path tracing on the GPU. Smallpt's algorithm in 99 lines was the starting point. We made it 30,000× faster on the same machine."
- Visual: smallpt 99-line code thumbnail on the left, your phase 3 render on the right.

### Slide 2 — Algorithm + complexity (50s)
- 4 bullets:
  - smallpt: serial recursive `radiance()`, ray-sphere tests against 9 spheres, RR after depth 4.
  - Multithreaded smallpt: same algorithm, OpenMP over rows.
  - cu-smallpt (phase 1): one CUDA thread per (pixel, sample), software ray-sphere intersection, no BVH.
  - OptiX (phase 3): one OptiX thread per (pixel, sample), hardware BVH traversal, hardware ray-triangle intersection on RT cores.
- Complexity:
  - smallpt: O(W·H·SPP·d·N) per-ray, where N = primitive count.
  - phase 3: O(W·H·SPP·d·log N), BVH log factor.
  - On 9 primitives the log doesn't matter; on 395k triangles (your tessellated scene) it matters enormously.
- Visual: side-by-side complexity formula.

### Slide 3 — Data structures + memory pattern (50s)
- Four data structures:
  - **Triangle GAS** — the BVH itself, 395k triangles, ~16 MB compacted on device.
  - **Accum buffer** — `float4[W*H]`, single allocation, 12 MB at 1024×768.
  - **SBT (Shader Binding Table)** — 9 hit-group records (6 walls + 3 spheres), each carries material + emission + albedo + center.
  - **PRD (per-ray-data)** — 80 bytes, lives in raygen stack, passed to CH via 2× uint32 packed pointer.
- Memory pattern:
  - **Primary rays:** spatially coherent, BVH traversal is cache-friendly on RT cores.
  - **Secondary rays:** divergent after first diffuse bounce; this is the bottleneck.
  - **Accum buffer:** one write per pixel, streamed.
  - **Per Nsight: L2 hit rate ~91%** (12.4 / 13.7 G sectors). Bandwidth-leaning bound, not compute.

### Slide 4 — Parallelization (40s)
- "Pixels are embarrassingly parallel; samples per pixel are too. We map (pixel × spp) to OptiX threads."
- "OptiX disallows recursive trace from CH shaders, so the recursive `radiance()` becomes an iterative bounce loop in raygen, with PRD accumulating throughput and radiance across bounces."
- "Tiled launches: 512×512 tile × N tiles. Single full-image accum buffer on device, host loop sets `tile_origin` and re-launches. Bounds OptiX per-launch state."
- Visual: pseudocode showing the bounce loop.

### Slide 5 — Optimizations (60s)
The big slide. Each bullet is ~6 seconds.
- Wall tessellation (16×16 + skirts) — fixes float32 precision banding from 1e5-radius wall-sphere phase 1.
- Russian Roulette after depth 4 — kills low-throughput paths early, ~40% perf win.
- Iterative bounce loop — required by OptiX.
- Payload pointer packing into 2× uint32 — keeps PRD in registers, avoids global-memory roundtrip.
- DISABLE_ANYHIT — opaque geometry, skips any-hit invocation.
- GAS compaction + PREFER_FAST_TRACE — 30% smaller BVH, optimal traversal.
- **Sphere tessellation (phase 3)** — mirror, glass, light → 256×256 lat/lon → 130k tris each. Single triangle GAS, no IAS, no built-in sphere intersection module. Drops a CH program group, drops a primitive type, drops the IAS dereference.
- **Tiled launches (phase 3)** — bounds per-launch device state. 4K_1024spp went from 18 GB → < 2 GB.
- Visual: bar chart Mrays/s phase1 → phase2 → phase3.

### Slide 6 — Where time goes + arithmetic intensity (50s)
- Pie chart from §4.5: BVH build / SBT setup / tile loop / readback. Shows tile loop dominates at any meaningful resolution.
- Inside the tile loop:
  - SM throughput 23%, DRAM 46% from Nsight. **Memory-leaning.**
  - Thread instructions ratio 24/32 — ~76% warp utilization, modest divergence.
  - XU (transcendental) pipe 4.6% — not transcendental-bound.
- Arithmetic intensity:
  - Empirical from §4.6: ~`X` FLOPs/byte (will fill in once Nsight FLOP metrics run).
  - Back-of-envelope per bounce: ~50 FLOPs of shading (RNG + onb + sample + scatter), ~100 bytes touched (PRD update + HitGroupData read). Roughly 0.5 FLOP/byte SM-side. RT-core ray-triangle work doesn't show in SM FLOPs.
  - Conclusion: **memory-leaning, not compute-bound** on Ampere. SER on Ada+ would help the divergence side.
- Profiling caveat (1 line): on Ampere with no SER, ncu sees the deprecated launcher kernel; pc-sampling stall reasons are unavailable.

### Slide 7 — Results (50s)
- Top: Mrays/s bar chart, smallpt → cu-smallpt → phase 2 → phase 3. Numbers on bars.
- Middle: image side-by-side at matched SPP. SSIM 0.97+. They look identical.
- Bottom: memory plot — phase 2 single-launch peak vs phase 3 tiled peak at 4K. The 18 GB → < 2 GB win.
- One sentence: "9× phase 1 → 1.9× more in phase 3, all without touching the algorithm."

### Engineering challenges slide (skippable in delivery, 30s if shown)
- The 1e5-radius wall-sphere precision saga (phase 1 → phase 2). Three thumbnails: phase 1 sphere walls / 1e4 isocontours / 1e5 banding / tessellated walls fix.
- One sentence: "this is what made the phase-2 tessellated-walls + skirts approach the right answer."

### Future work (10s)
- "Ada+ SER, NEE+MIS, denoiser, wavefront restructure, ReSTIR for real-time."

---

## 8. Execution timeline (half-day)

If you start at 9 AM:

- **9:00–9:15** Run §4.1 (phase 2 baseline rerun, kicks off in background).
- **9:15–10:00** §5.3, §5.4, §5.5 (CLI flag + memory print + event timers in `main.cpp`). Rebuild.
- **10:00–10:15** §4.5 quick run to confirm event timers print clean numbers.
- **10:15–11:15** §4.2 phase 3 full grid (1 hour wall-clock for 9 configs including 4K).
- **11:15–11:45** §4.4 bounce-depth ablation (parameterized by §5.3).
- **11:45–12:00** §4.3 memory measurement at 4K.
- **12:00–12:30** Lunch + decide NEE go/no-go based on slack remaining.
- **12:30–13:30** Either: NEE implementation (§3) — or — Nsight runs (§4.6, §4.7) + SSIM check (§4.8).
- **13:30–14:00** Charts (§6.4). Three PNGs.
- **14:00 onwards** Deck.

If everything goes sideways and you only get 2 hours: do §4.1, §4.2, §4.5, §4.8. Skip ablation, skip memory plot, skip NEE. The bar chart + SSIM + a memory note is enough for the deck.

---

## 9. Risks

| Risk | Trigger | Mitigation |
|---|---|---|
| Phase 2 baseline doesn't rebuild cleanly | Old build/ stale | `Remove-Item phase2/build -Recurse; .\select_scene.ps1 -Name default` to force regen |
| Memory measurement at 4K is misleading because it's measured post-cleanup | `cudaMemGetInfo` after `cudaFree` shows little | Move the print into the tile loop (after first launch). Or just use `nvidia-smi` external logger. |
| `--max-depth` parameterization breaks RR | RR uses `prd.depth > 4` literal | Make the RR threshold a separate `Params` field too, default 4. Or just leave RR threshold hard-coded and only parameterize the loop cap. |
| ncu application replay still doesn't show raygen | Ampere just won't do it | Caveat slide. Move on. |
| Phase 3 full grid takes >2 hours | 4K runs slow | Drop the 4096_1024spp config from the full grid. 4096_256spp is enough for memory-plot purposes. |
| NEE introduces visible bias | SSIM < 0.95 after NEE | Revert. The unNEE'd version is good. |

---

## 10. Definition of Done

The deck is shippable when:

- [ ] Phase 2 + phase 3 timings.csv files both populated, same machine, same scene.
- [ ] Phase 3 full 9-config grid run.
- [ ] Bounce-depth ablation: 6 data points, plot.
- [ ] Memory at 4K_1024spp measured for phase 3, ideally for phase 2 too.
- [ ] Per-phase wall-clock breakdown for at least 1024×768_256spp.
- [ ] SSIM ≥ 0.97 phase 3 vs phase 2 reference at matched SPP.
- [ ] Nsight quick capture for phase 2 + phase 3 with FLOP metrics.
- [ ] (optional) Application-replay attempt run, result documented either way.
- [ ] All 7 slides + engineering-challenges slide drafted with assets.
- [ ] Phase 1 number landed when phase 1 rerun completes.

---

## 11. Hand-off

If picking this up cold:

- Phase 3 source is in `phase3/`. Phase 2 in `phase2/`. Both build with `select_scene.ps1 -Name default`.
- Reference render at `phase3/reference/4096x3072_4096spp.png`.
- `run_nsight.ps1` MUST set `OPTIX_FORCE_DEPRECATED_LAUNCHER=1` for ncu to see anything.
- The kernel name in ncu output will be `optixLaunch`, not raygen. This is expected on Ampere.
- The deck's Mrays/s headline bar is `1024x768_256spp` across phases.
- The optimization story is wall-clock-driven. Nsight is supporting evidence with the caveat slide.
