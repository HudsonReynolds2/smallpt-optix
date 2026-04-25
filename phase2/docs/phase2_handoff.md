# Phase 2 Handoff

Where things stand, what to do next, what not to repeat.

---

## State

OptiX 8.1 path tracer (`optix_smallpt`) on Windows. Cornell box matches Phase 1 reference. 1280×720 / 256 spp ≈ 740 ms (≈320 Mrays/s, ~9× faster than Phase 1's WSL2 number).

- 12 wall triangles + 3 built-in spheres (mirror, glass, light)
- Triangle GAS + Sphere GAS under one IAS
- One closest-hit shader for both, branches on `optixGetPrimitiveType()`
- Sources: `phase2/src/main.cpp`, `phase2/src/shaders.cu`, `phase2/src/shared.h`

---

## Build & run (VS 2022 Developer PowerShell)

```powershell
cd C:\Users\hudsonre\527project\phase2\build
cmake --build . --config Release

cd Release
.\optix_smallpt.exe --width 1280 --height 720 --spp 256 `
    --output ..\..\results\phase2_optix_spheres\renders\720p_256spp.ppm `
    --ptx ..\shaders.ptx
```

Convert PPM → PNG with `python -c "from PIL import Image; Image.open('x.ppm').save('x.png')"`.
Save them to the results folder.

---

## What's left for Phase 2

Note: This should all be in a script that can be re run and saves to a folder with a timestamp, and generates the pngs of each ppm for easy viewing.
1. Run the full benchmark grid (see `plan.md`):
   - 1280×720 @ 256, 1024
   - 1920×1080 @ 256, 1024, 4096
   - 3840×2160 @ 256, 1024
2. Save renders + append CSV rows to `results/phase2_optix_spheres/timings.csv`.
3. Run Nsight Compute (`ncu --set full -o ...`) on at least 1080p/1024spp.
4. Tag commit `phase2-done`.


---

## What not to redo

- **Don't bring back the radius-1e5 wall spheres.** OptiX 8's hardware sphere intersector is float32 only and produces precision banding at that scale. Real triangles are the correct fix and what every other Cornell box renderer uses. Diagnostic confirmation in `results/phase2_optix_spheres/renders/diagnostic/walls_1e4.png` and `walls_1e5.png`.
- **Don't drop `OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS` from either GAS.** Without it, `optixGetSphereData` and `optixGetTriangleVertexData` silently return garbage (no error unless validation mode is on).
- **Don't call `optixGetObjectRay*` from closest-hit.** IS/AH only. Use `optixGetWorldRay*` and a single transform if you need object-space coords.
- **Don't redo Phase 1 on Windows.** WSL2 vs Windows-native CUDA on the same GPU is within ~5% for compute-bound kernels; the gap to Phase 2 is so large the platform difference is a footnote. Once we are done, we can see if smallpgt gpu has much better native windows performance.

---

## Phase 3 plan (quick version)

Walls already triangles, so Phase 3 only changes the 3 small spheres:

1. Branch: `git checkout -b phase3-triangle-mesh`
2. Add `tessellate_sphere(center, radius, lat=64, lon=64)` to `main.cpp`. Generates ~8K triangles per sphere.
3. Replace sphere GAS construction with a triangle GAS containing the tessellated meshes. Either keep the IAS (one walls-GAS + one tess-spheres-GAS) or merge into a single triangle GAS — either works.
4. Pipeline compile: drop `OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE`, drop the built-in sphere IS module, drop the sphere hit-group program group.
5. Closest-hit: drop the primitive-type branch; triangle path only.
6. Build, render, compare against Phase 2 image at equal SPP (should match closely; sphere silhouettes show tessellation at extreme zoom but not at 1280×720).
7. Benchmark same grid as Phase 2, save to `results/phase3_optix_triangles/`.

Time estimate: 2–3 hours. Most of the host code is already structured for this.

---

## Files in `results/`

- `phase2_optix_spheres/renders/720p_256spp.{ppm,png}` — final correct render
- `phase2_optix_spheres/renders/diagnostic/walls_1e4.png` — precision experiment (small-radius walls, shows precision banding)
- `phase2_optix_spheres/renders/diagnostic/walls_1e5.png` — original buggy render kept as artifact
- `phase2_optix_spheres/timings.csv` — needs to be populated with the full grid

`ref_720p.png` from Phase 1 lives at the top level for quick visual diff.
