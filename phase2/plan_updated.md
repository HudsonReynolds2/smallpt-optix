# EC527 Ray Tracing Project Plan
## CUDA Baseline → OptiX 8 Path Tracer (Spheres → Triangle Mesh)

**Solo project | Platform: RTX 3080 Ti, SM 8.6 (Ampere)**

---

## Overview

This project ports and extends [cu-smallpt](https://github.com/matt77hias/cu-smallpt), a CUDA implementation of the smallpt path tracer, through three progressive phases:

1. **Phase 1** — CUDA baseline: build, characterize, and benchmark cu-smallpt as-is
2. **Phase 2** — OptiX 8 with built-in primitives: walls as triangles, mirror/glass/light as built-in spheres, hardware BVH traversal
3. **Phase 3** — OptiX 8 fully triangle-based: tessellate the remaining spheres into triangle meshes, all geometry on RT cores

All three phases implement the same path tracing algorithm (unidirectional Monte Carlo path tracing, Russian roulette termination, diffuse/specular/refractive materials). Structural differences introduced by the OptiX programming model are documented explicitly. Results from every phase are saved incrementally to `results/`.

---

## Environment

Two platforms are used. Phase 1 ran on WSL2; Phases 2/3 run on Windows native because the WSL2 NVIDIA driver stack does not expose a working OptiX runtime (`libnvoptix.so.1` is a 9.9KB stub, `OPTIX_ERROR_LIBRARY_NOT_FOUND` / `ENTRY_SYMBOL_NOT_FOUND` could not be resolved). Both platforms target the same physical GPU at SM 8.6.

| Component | Phase 1 (WSL2) | Phases 2 & 3 (Windows) |
|---|---|---|
| GPU | RTX 3080 Ti (SM 8.6, Ampere) | same |
| Driver | 596.21 | same |
| CUDA Toolkit | 12.0 (apt) | 12.8 |
| OptiX SDK | — | 8.1.0 |
| Host compiler | GCC 13.3 | MSVC 19.44 (VS 2022 Build Tools) |
| OS | Ubuntu 24.04 (WSL2) | Windows 11 |

**Tooling notes:**
- VS 2022 Developer PowerShell is required on Windows. VS 2026 is unsupported by CUDA 12.8 and breaks CMake CUDA toolset detection.
- CUDA MSBuild integration files have to be copied into the VS 2022 Build Tools directory:
  ```
  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions\*
    -> C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\
  ```
- Nsight Compute hardware counters (`ncu --set full`) are blocked under WSL2 (`ERR_NVGPUCTRPERM`). For Phases 2/3 we use the Windows-side Nsight Compute GUI or `ncu` from the Windows shell.

---

## Results Directory Structure

Create at the start, populate throughout, never delete intermediate results.

```
results/
  phase1_cuda_baseline/
    renders/
      720p_256spp.png
      1080p_1024spp.png
      4k_4096spp.png
    timings.csv
    nsight/
    notes.md
  phase2_optix_spheres/
    renders/
      diagnostic/         # precision experiment renders (1e4, 1e5 wall spheres)
    timings.csv
    nsight/
    notes.md
  phase3_optix_triangles/
    renders/
    timings.csv
    nsight/
    notes.md
  differences.md          # cross-phase analysis for the report
```

**timings.csv format (same for all phases):**
```
phase,resolution,spp,time_ms,mrays_sec,notes
cuda_baseline,1920x1080,1024,4312.5,487.2,sm_86 arch
```

---

## Phase 1 — CUDA Baseline (WSL2) ✅ Complete

### Goals
- Build cu-smallpt targeting SM 8.6
- Parameterize resolution and SPP via CLI
- Benchmark across resolution/SPP configurations
- Establish reference renders for image-quality comparison

### Status
Done on WSL2. Reference render at 1280×720, 256 spp = 6787 ms (34.76 Mrays/s) is the Phase 2/3 image-quality target (`ref_720p.png`).

### Notes for the report
- Phase 1 numbers come from a different host OS than Phases 2/3 because of the WSL2/OptiX issue. Both runs target the same physical GPU at SM 8.6, and cu-smallpt is compute-bound (no graphics interop), so within ~5% the WSL2 vs. Windows wobble is negligible compared to the 8.5×+ gap to Phase 2.
- Nsight Compute hardware counters were not capturable on WSL2; we report only event-timing throughput for Phase 1. Compare-against numbers in the report should use end-to-end Mrays/s rather than counter-derived metrics.

---

## Phase 2 — OptiX 8 Path Tracer (Built-in Primitives)

### Goals
- Port the path tracer to the modern OptiX 7+/8 API
- Match Phase 1's image quality at the same SPP
- Benchmark same configurations, compare to Phase 1
- Document structural differences from the CUDA version

### Scope deviation from the original plan
The original plan called for **custom sphere intersection** with all 9 smallpt primitives (6 wall spheres + mirror + glass + light) kept as analytic spheres. That changed for two reasons.

**1. OptiX 8 ships a hardware-friendly built-in sphere primitive.** Using `OPTIX_BUILD_INPUT_TYPE_SPHERES` is faster than rolling our own intersection program, since it cooperates with the BVH traversal hardware. We use the built-in.

**2. The smallpt 1e5-radius wall trick fails in OptiX 8 due to float32 precision.** smallpt builds Cornell box walls as spheres with radius `1e5` whose surface locally approximates a plane. The OptiX 8 hardware sphere intersector is float32 only, and at radius 1e5 the quadratic root finder produces visibly banded t-values. The artifacts manifest as rectangular tile patterns on flat walls and a faint ghost back-wall sphere ([reference scene comparison](#)). pocketpt (a related GLSL smallpt port) hit the same wall and worked around it by using `double` precision in the sphere intersection — which we cannot do because the intersector is hardware.

A diagnostic experiment confirmed precision was the cause. Wall radius was scaled down from 1e5 to 1e4 with everything else held fixed: the rectangular banding became visible concentric isocontours of the spherical surface, exactly as expected when the curvature is barely-resolved by the float32 t-solver. See `results/phase2_optix_spheres/renders/diagnostic/walls_1e4.png` and `walls_1e5.png`.

**Resulting design.** The 6 walls become **flat triangles** (12 triangles total, 2 per quad) at the actual room boundaries. Mirror, glass, and light remain **built-in spheres** (their radii are 16.5–600, well within the precision envelope). This is closer to how every real-world Cornell box renderer is built.

### Acceleration structure

Mixing primitive types in a single GAS is brittle in OptiX 8. We use two GASes under one IAS:

| GAS | Primitive type | Count | SBT range |
|---|---|---|---|
| Triangle GAS | Built-in triangles | 12 | records 0..11 |
| Sphere GAS | Built-in spheres | 3 | records 12..14 |
| IAS | 2 instances, identity transforms | — | — |

The IAS instance for the sphere GAS uses `sbtOffset = NUM_TRIANGLES = 12` so the sphere records sit immediately after the triangle records.

### Pipeline structure

| Shader | Role | Notes |
|---|---|---|
| Ray Generation | Per-pixel iterative bounce loop, accumulates radiance | One launch handles all spp |
| Closest-Hit | Material eval + scatter direction; branches on `optixGetPrimitiveType()` for normal | Shared by triangles and spheres |
| Miss | Marks path as done (background is black) | |
| Intersection | None for triangles (hardware); built-in OptiX sphere IS module for spheres | No custom IS shader needed |
| Any-Hit | Not used | |

### Key algorithmic difference: recursion → iteration

cu-smallpt uses recursive `radiance()` calls. OptiX does not support recursive `optixTrace()` from within a closest-hit shader, so the bounce loop is restructured into the ray generation shader. Mathematically equivalent.

```cpp
// In RayGen shader:
float3 throughput = {1,1,1};
float3 radiance   = {0,0,0};
for (int bounce = 0; bounce < MAX_DEPTH && !prd.done; bounce++) {
    optixTrace(..., payload);
    // payload.radiance and payload.throughput updated by closest-hit / miss
}
```

### Required OptiX flags (lessons learned)

These bit us during development; documenting so they don't again:

- `OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS` is **mandatory** on any GAS whose closest-hit shader calls `optixGetSphereData` or `optixGetTriangleVertexData`. Without it those calls return undefined values silently (unless validation mode is on, in which case OptiX throws `OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS`).
- `optixGetObjectRayOrigin` / `optixGetObjectRayDirection` are **IS/AH only**. Closest-hit must use `optixGetWorldRay*` and either compute object-space hit position via a single transform, or do shading entirely in world space.
- `pipelineCompileOptions.usesPrimitiveTypeFlags` must include both `TRIANGLE` and `SPHERE` since the pipeline now sees both.
- `pipelineCompileOptions.numAttributeValues` must be at least 2 (built-in triangle barycentrics use 2 attributes). A value of 1 from an earlier sphere-only iteration silently breaks triangle hits.
- `traversableGraphFlags` must be `ALLOW_SINGLE_LEVEL_INSTANCING` once we have an IAS over GASes (was `ALLOW_SINGLE_GAS` in the sphere-only version).
- `optixPipelineSetStackSize` last argument is `maxTraversableGraphDepth`; must be 2 for IAS→GAS.

### Steps

**2.1 OptiX SDK install (Windows)**
Download installer from https://developer.nvidia.com/optix and install to `C:\ProgramData\NVIDIA Corporation\OptiX SDK 8.1.0\`.

**2.2 Project layout**
```
phase2/
  CMakeLists.txt
  src/
    main.cpp           # host: context, GASes, IAS, pipeline, SBT, launch
    shaders.cu         # raygen + closest-hit + miss (compiled to PTX)
    shared.h           # Params / PRD / HitGroupData definitions
  results/ -> ../results/phase2_optix_spheres/
```

**2.3 Build (VS 2022 Developer PowerShell)**
```powershell
cd phase2\build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**2.4 Run**
```powershell
.\Release\optix_smallpt.exe --width 1280 --height 720 --spp 256 `
    --output ..\..\results\phase2_optix_spheres\renders\720p_256spp.ppm `
    --ptx ..\shaders.ptx
python -c "from PIL import Image; Image.open('...\720p_256spp.ppm').save('...\720p_256spp.png')"
```

**2.5 Benchmark grid**
Run all configs, log to `results/phase2_optix_spheres/timings.csv`:

| Resolution | SPP |
|---|---|
| 1280×720 | 256, 1024 |
| 1920×1080 | 256, 1024, 4096 |
| 3840×2160 | 256, 1024 |

**2.6 Profile**
```powershell
ncu --set full -o results\phase2_optix_spheres\nsight\optix_p2_1080p_1024spp `
    .\optix_smallpt.exe --width 1920 --height 1080 --spp 1024 `
    --output tmp.ppm --ptx ..\shaders.ptx
```

Phase 2 exercises RT cores for BVH traversal across all geometry, and for triangle intersection on the walls. The 3 spheres traverse on RT cores but their intersection runs as the built-in sphere IS program on shader cores — not the HW triangle intersector. Note this when discussing RT-core utilization in the report; the cleanest "all geometry on RT cores" measurement is Phase 3.

---

## Phase 3 — OptiX 8, Fully Triangle-Based

### Why this phase exists

Phase 2 is **not** fully RT-core accelerated, despite using OptiX. The RT core on Ampere does two things in hardware: (1) BVH traversal, and (2) ray-triangle intersection (fixed-function silicon). Anything that isn't a triangle gets #1 but not #2 — the intersection math runs as a shader program (an IS program) on the regular SM cores.

Built-in spheres (`OPTIX_BUILD_INPUT_TYPE_SPHERES`) are a special case: NVIDIA ships an optimized sphere intersection routine bundled with the driver, and it's faster than a hand-rolled custom IS program because it's integrated with the traversal pipeline. But it is still an IS program running on shader cores — **not** the hardware triangle intersector. So Phase 2's three spheres (mirror, glass, light) take a hybrid path: HW traversal + SW intersection. The 12 wall triangles take the full HW path.

This muddies the headline claim. Phase 1 → Phase 2 shows ~9× from "software CUDA path tracer" to "OptiX with mostly-RT-core geometry," but you can't cleanly attribute that delta to the RT cores when part of the scene bypasses the triangle intersector. Phase 3 puts every primitive on the same hardware path so the comparison is unambiguous, and so Nsight Compute's RT-core utilization numbers reflect the entire scene.

There's a secondary empirical question worth measuring: tessellating each sphere to ~8K triangles inflates the BVH and adds work per ray. Does the hardware triangle intersector still beat the built-in sphere IS once you account for that overhead? On Ampere the expected answer is yes, but it's worth verifying rather than assuming.

There's also a precision motivation. The Phase 2 wall-sphere experiment showed that analytic primitives at extreme scales (1e5 radius) fall outside the float32 hardware intersector's precision envelope. Tessellation sidesteps this entirely — every triangle is small and well-conditioned. Phase 3 finishes converting the scene to the representation that production path tracers actually use.

### Goals
- Replace the 3 remaining built-in spheres with tessellated triangle meshes
- Move *all* geometry onto hardware triangle intersection (RT cores)
- Drop the built-in sphere IS module entirely
- Benchmark same configurations, compare all three phases

### What changes from Phase 2

The walls are already triangles from Phase 2. The only thing that changes is the 3 sphere primitives.

| | Phase 2 | Phase 3 |
|---|---|---|
| Walls | 12 triangles (already) | 12 triangles (unchanged) |
| Mirror, glass, light | Built-in sphere primitive | Tessellated triangle mesh |
| Intersection | Hardware triangle + built-in sphere | Hardware triangle only |
| `usesPrimitiveTypeFlags` | TRIANGLE \| SPHERE | TRIANGLE only |
| Sphere hit group | Has built-in sphere IS module | Removed |
| Closest-hit | Branches on primitive type for normal | Single triangle path |

### Sphere tessellation

Each of the 3 spheres tessellated to a UV sphere mesh on the host before upload. Suggested subdivision: 64 latitude × 64 longitude = ~8,192 triangles per sphere = ~24,576 added triangles total. Smooth enough to be visually equivalent to the analytic sphere at this scene scale.

```cpp
void tessellate_sphere(float3 center, float radius, int lat, int lon,
                       std::vector<float3>& verts,
                       std::vector<uint3>& indices) {
    int base = (int)verts.size();
    for (int i = 0; i <= lat; i++) {
        float phi = M_PI * i / lat;
        for (int j = 0; j <= lon; j++) {
            float theta = 2.0f * M_PI * j / lon;
            verts.push_back(center + radius * make_float3(
                sinf(phi)*cosf(theta), cosf(phi), sinf(phi)*sinf(theta)));
        }
    }
    for (int i = 0; i < lat; i++) {
        for (int j = 0; j < lon; j++) {
            int a = base + i*(lon+1) + j;
            int b = a + 1;
            int c = a + (lon+1);
            int d = c + 1;
            indices.push_back({(uint)a,(uint)b,(uint)d});
            indices.push_back({(uint)a,(uint)d,(uint)c});
        }
    }
}
```

For shading, **face normals via `optixGetTriangleVertexData` + cross product** is the simplest path and matches what the Phase 2 closest-hit already does for walls. Smooth shading via per-vertex normals is a stretch goal — it would visually closer match the analytic sphere but adds complexity (extra buffer, weighted normals). Start with face normals, ship a smooth version if time permits.

### Steps

**3.1 Branch from Phase 2**
```powershell
git checkout -b phase3-triangle-mesh
```

**3.2 Replace sphere GAS with triangle GAS**
- Add `tessellate_sphere` to `main.cpp`
- Build a single triangle GAS containing both walls and tessellated spheres, OR keep 2 separate GASes (one walls, one tess-spheres) under the IAS. Single-GAS is simpler.
- SBT layout: each tessellated sphere's triangles share one material, so use `sbtIndexOffsetBuffer` to point all of sphere N's triangles at hit group record N.

**3.3 Update pipeline compile options**
- `usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE` only
- Remove the built-in sphere IS module construction
- Hit group can drop the sphere variant; one triangle hit group covers everything

**3.4 Simplify closest-hit**
- Drop the `optixGetPrimitiveType()` branch
- Triangle face-normal path is the only path

**3.5 Render and benchmark**
Same configurations as Phase 1/2. Compare image quality side-by-side at equal SPP.

```powershell
ncu --set full -o results\phase3_optix_triangles\nsight\optix_p3_1080p_1024spp `
    .\optix_smallpt_tri.exe --width 1920 --height 1080 --spp 1024 `
    --output tmp.ppm --ptx ..\shaders.ptx
```

For Phase 3, all geometry is on RT cores. Mention `sm__inst_executed_pipe_tensor_op_hmma` and any RT-related counters if Nsight surfaces them.

---

## Denoiser (Optional, Time Permitting)

OptiX ships a built-in AI denoiser (`OptixDenoiser`). If time allows after Phase 3:

- Render at low SPP (e.g. 64) with Phase 3
- Run the OptiX denoiser on the output
- Compare image quality (visually and via PSNR) against Phase 1 at high SPP
- Compare total time: `Phase3_64spp + denoise_time` vs. `Phase1_4096spp`

Strong result for the report if it works. Skip if time is short.

---

## Documented Differences (for Report)

Maintain `results/differences.md` as you go.

1. **Recursion → iteration.** Recursive `radiance()` replaced by iterative bounce loop in ray-gen. Mathematically equivalent estimator.
2. **Kernel structure.** Monolithic per-pixel CUDA kernel → multi-stage OptiX pipeline (raygen / closest-hit / miss / IS) invoked by the traversal engine.
3. **Acceleration structure.** Phase 1 — linear sphere list, O(N) per ray. Phases 2/3 — hardware-accelerated BVH (IAS over per-primitive-type GASes).
4. **Intersection hardware.** The RT core does BVH traversal and ray-triangle intersection in fixed-function hardware; non-triangle primitives use HW traversal but execute an IS program on shader cores for the intersection.
   - Phase 1: software ray-sphere on CUDA cores. No RT-core involvement at all.
   - Phase 2: walls on the full HW path (traversal + triangle intersection). Three spheres on a hybrid path — HW traversal + NVIDIA's built-in sphere IS program on shader cores. Faster than custom IS, but **not** the HW triangle intersector.
   - Phase 3: every primitive (walls + tessellated spheres) on the full HW path. All intersection on RT cores.
5. **Wall geometry.** Phase 1 uses radius-1e5 spheres for walls (smallpt's classic trick). Phases 2/3 use real flat triangles because the OptiX 8 hardware sphere intersector is float32 only and produces visible precision banding at radius 1e5. See diagnostic experiment in `results/phase2_optix_spheres/renders/diagnostic/`.
6. **Subpixel sampling.** Phase 1 uses smallpt's 2×2 stratified subpixel grid. Phases 2/3 simplify to a tent-filtered jitter per sample. Same converged image, slightly different convergence rate per spp.
7. **RNG.** All phases use a per-pixel LCG. Seeding is similar but not identical (thread indexing differs slightly between the CUDA kernel and the OptiX raygen).
8. **Memory layout.** Phase 1 — device array of `Sphere` structs. Phases 2/3 — SBT records per hit group, vertex/index/sphere buffers in OptiX-managed GAS storage.
9. **Platform.** Phase 1 on WSL2/GCC, Phases 2/3 on Windows/MSVC due to a missing OptiX runtime in the WSL2 driver stack.

---

## Benchmark Summary Template

| Phase | Resolution | SPP | Time (ms) | Mrays/s | Speedup vs P1 |
|---|---|---|---|---|---|
| 1 — CUDA baseline (WSL2) | 1280×720 | 256 | 6787 | 34.76 | 1× |
| 2 — OptiX walls+spheres (Windows) | 1280×720 | 256 | — | — | —× |
| 3 — OptiX triangles only (Windows) | 1280×720 | 256 | — | — | —× |
| 1 — CUDA baseline | 1920×1080 | 1024 | — | — | 1× |
| 2 — OptiX walls+spheres | 1920×1080 | 1024 | — | — | —× |
| 3 — OptiX triangles only | 1920×1080 | 1024 | — | — | —× |

Include side-by-side renders at equal SPP to show visual equivalence.

---

## Notes

- Commit and tag each phase: `phase1-done`, `phase2-done`, `phase3-done`.
- If a benchmark config takes >5 min, drop it and note in timings.csv.
- OptiX validation mode (`OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL`) is on for debug builds and saves enormous debugging time — leave it enabled while iterating, disable for benchmark runs.
- All three phases compile with `-arch=sm_86`.
