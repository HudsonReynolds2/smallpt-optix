#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Material types matching Phase 1
enum MaterialType : unsigned int {
    MAT_DIFFUSE    = 0,
    MAT_SPECULAR   = 1,
    MAT_REFRACTIVE = 2,
};

// Per-primitive SBT hit group data.
// Used for both triangles (walls) and built-in spheres (mirror, glass, light).
struct HitGroupData {
    float3       emission;
    float3       albedo;
    MaterialType material;
    int          is_sphere;     // 1 for tessellated sphere records, 0 for walls
    float3       sphere_center; // analytic center for smooth normals (zero for walls)
};

// Miss SBT data (nothing needed, background is black)
struct MissData {
    float3 bg_color;
};

// Ray gen SBT data (nothing needed, camera params go in Params)
struct RayGenData {};

// Launch params - passed to all shaders via constant memory
struct Params {
    float4*                accum_buffer;   // float4 accumulation buffer (RGBA, A unused)
    unsigned int           width;
    unsigned int           height;
    unsigned int           samples_per_launch; // samples this launch (we do 1 launch per spp)
    unsigned int           subframe_index;     // for RNG seeding

    // Tile origin for tiled launches (raygen adds this to launch index).
    // Both are 0 for single-launch (non-tiled) mode.
    unsigned int           tile_origin_x;
    unsigned int           tile_origin_y;

    // Maximum number of bounces per ray. Hardcoded literal `20` was used
    // before; now parameterized so we can run a depth ablation for the
    // EC527 deck without recompiling. RR (depth>4) makes high values nearly
    // free, but the ablation curve is informative.
    //
    // pipelineLinkOptions.maxTraceDepth must be >= this value at pipeline
    // creation time, so on the host side we still pass 20 to OptiX even
    // when we run with a smaller cap; the loop just exits sooner.
    unsigned int           max_bounces;

    // Camera (smallpt Cornell box)
    float3                 eye;
    float3                 cx;   // right vector scaled by fov/aspect
    float3                 cy;   // up vector scaled by fov

    OptixTraversableHandle handle;
};

// ---------------------------------------------------------------------------
// Per-ray payload (PRD)
//
// PHASE 3 v4: PRD is no longer a host-visible struct. The fields previously
// stored in a stack-allocated PRD (which the compiler spilled to local memory
// because we passed its address through the optixTrace call as a u64 pointer)
// now live entirely in OptiX payload REGISTERS. This is a pure register-level
// optimization -- no algorithmic change.
//
// Why: Nsight Compute on the previous build showed
//   - DRAM 44.95% of peak (memory-bound)
//   - Occupancy 31.23% (register-pressure limited)
// The PRD pointer-through-payload pattern was forcing every prd-> field
// access in CH/miss to round-trip through L1/L2 (and frequently DRAM, given
// the L2 hit rate). Moving the fields to payload registers eliminates those
// loads/stores entirely and frees the registers that had been holding the
// PRD's stack slot, which raises occupancy.
//
// Payload register layout (15 total). The hot fields (radiance, throughput,
// seed, depth, done) are in/out across the trace call; origin/direction are
// outputs from CH that raygen reads to issue the next bounce.
//
//   0..2   radiance (float3) -- in: accumulator so far; out: same + emission
//   3..5   throughput (float3) -- in: current; out: post-RR, post-albedo
//   6      seed (uint32) -- in/out (RR + material sampling consume entropy)
//   7      depth (uint32) -- in: bounce depth before hit; out: depth + 1
//   8      done (uint32) -- in: 0; out: 1 if path terminated (miss or RR)
//   9..11  next ray origin (float3) -- output only (CH writes hit_pos)
//   12..14 next ray direction (float3) -- output only (CH writes scatter dir)
//
// Pipeline: numPayloadValues must be >= 15 in OptixPipelineCompileOptions.
// On OptiX 8 the per-trace cap is 32, so 15 is comfortably within budget.
//
// Floats are passed via __float_as_uint / __uint_as_float for bitwise round
// trips through the integer payload slots.
// ---------------------------------------------------------------------------
