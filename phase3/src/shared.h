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
