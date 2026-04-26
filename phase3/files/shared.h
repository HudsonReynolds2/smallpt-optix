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
// Triangles only in phase 3 (spheres got tessellated into the same GAS).
struct HitGroupData {
    float3       emission;
    float3       albedo;
    MaterialType material;
};

// Miss SBT data (nothing needed, background is black)
struct MissData {
    float3 bg_color;
};

// Ray gen SBT data (nothing needed, camera params go in Params)
struct RayGenData {};

// Launch params - passed to all shaders via constant memory
//
// PHASE 3 CHANGES:
//   - Added tile_origin_x / tile_origin_y so the host can launch a smaller
//     (tile_w, tile_h, 1) grid covering one tile of the full image, and
//     the raygen offsets its pixel index by the tile origin to write into
//     the right slice of the full-resolution accumulation buffer.
//   - The accum_buffer is allocated once at full image size; tiles each
//     write to their own region. No per-tile allocations.
struct Params {
    float4*                accum_buffer;   // float4 accumulation buffer (full image, RGBA, A unused)
    unsigned int           width;          // full image width
    unsigned int           height;         // full image height
    unsigned int           samples_per_launch; // samples this launch (we do 1 launch per spp)
    unsigned int           subframe_index;     // for RNG seeding

    // Tile origin within the full image (added in phase 3).
    // optixGetLaunchIndex() returns coords within this tile only.
    unsigned int           tile_origin_x;
    unsigned int           tile_origin_y;

    // Camera (smallpt Cornell box)
    float3                 eye;
    float3                 cx;   // right vector scaled by fov/aspect
    float3                 cy;   // up vector scaled by fov

    OptixTraversableHandle handle;
};

// Per-ray payload - passed as pointer split across 2 uint32 registers
struct PRD {
    float3       radiance;
    float3       throughput;
    float3       origin;
    float3       direction;
    unsigned int seed;
    int          depth;
    int          done;
};
