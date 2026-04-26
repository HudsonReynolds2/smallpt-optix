#include <optix.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "shared.h"

// vec_math helpers inline - avoid sutil dependency
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ float3 operator*(float s, float3 a) { return make_float3(a.x*s, a.y*s, a.z*s); }
__device__ __forceinline__ float3 operator/(float3 a, float s) { return make_float3(a.x/s, a.y/s, a.z/s); }
__device__ __forceinline__ float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
__device__ __forceinline__ float3& operator*=(float3& a, float3 b) { a.x*=b.x; a.y*=b.y; a.z*=b.z; return a; }
__device__ __forceinline__ float3& operator*=(float3& a, float s)  { a.x*=s;   a.y*=s;   a.z*=s;   return a; }
__device__ __forceinline__ float3& operator/=(float3& a, float s)  { a.x/=s;   a.y/=s;   a.z/=s;   return a; }
__device__ __forceinline__ float dot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ __forceinline__ float3 normalize(float3 v) {
    float inv = 1.0f / sqrtf(dot(v,v)); return v * inv;
}
__device__ __forceinline__ float fmaxf3(float3 v) { return fmaxf(v.x, fmaxf(v.y, v.z)); }

// Launch params in constant memory
extern "C" { __constant__ Params params; }

// ---------------------------------------------------------------------------
// Payload pointer packing (PRD passed as pointer split into 2 x uint32)
// ---------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int u32_from_ptr_lo(PRD* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<unsigned long long>(ptr));
}
static __forceinline__ __device__ unsigned int u32_from_ptr_hi(PRD* ptr) {
    return static_cast<unsigned int>(reinterpret_cast<unsigned long long>(ptr) >> 32);
}
static __forceinline__ __device__ PRD* ptr_from_u32(unsigned int lo, unsigned int hi) {
    return reinterpret_cast<PRD*>(
        static_cast<unsigned long long>(lo) | (static_cast<unsigned long long>(hi) << 32)
    );
}

// ---------------------------------------------------------------------------
// RNG helpers (LCG, same style as optixPathTracer sample)
// ---------------------------------------------------------------------------
static __forceinline__ __device__ unsigned int lcg(unsigned int& seed) {
    seed = 1664525u * seed + 1013904223u;
    return seed;
}
static __forceinline__ __device__ float rnd(unsigned int& seed) {
    return static_cast<float>(lcg(seed) & 0x00FFFFFF) / 0x01000000;
}

// ---------------------------------------------------------------------------
// Sampling utilities
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 cosine_sample_hemisphere(float u1, float u2) {
    const float r   = sqrtf(u1);
    const float phi = 2.0f * CUDART_PI_F * u2;
    return make_float3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u1)));
}

static __forceinline__ __device__ void onb_from_normal(float3 n, float3& u, float3& v) {
    // Build orthonormal basis from normal (Duff et al. style)
    float3 up = fabsf(n.x) > 0.1f ? make_float3(0,1,0) : make_float3(1,0,0);
    u = normalize(cross(up, n));
    v = cross(n, u);
}

// ---------------------------------------------------------------------------
// Specular helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 reflect(float3 d, float3 n) {
    return d - 2.0f * dot(n, d) * n;
}

static __forceinline__ __device__ float schlick(float cos_theta, float n1, float n2) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    float c = 1.0f - cos_theta;
    return r0 + (1.0f - r0) * c*c*c*c*c;
}

// Returns refracted direction or reflected if TIR. pr = weight for this choice.
static __forceinline__ __device__ float3 refract_or_reflect(
    float3 d, float3 n, float n_out, float n_in, float& pr, unsigned int& seed)
{
    float3 d_re = reflect(d, n);
    bool out_to_in = dot(n, d) < 0.0f;
    float3 nl = out_to_in ? n : make_float3(-n.x,-n.y,-n.z);
    float  nn = out_to_in ? n_out/n_in : n_in/n_out;
    float  cos_theta = dot(d, nl);  // negative
    float  cos2_phi  = 1.0f - nn*nn*(1.0f - cos_theta*cos_theta);

    if (cos2_phi < 0.0f) { pr = 1.0f; return d_re; } // TIR

    float3 d_tr = normalize(nn*d - nl*(nn*cos_theta + sqrtf(cos2_phi)));
    float  cos_fresnel = out_to_in ? -cos_theta : dot(d_tr, n);
    float  Re   = schlick(cos_fresnel, n_out, n_in);
    float  p_Re = 0.25f + 0.5f * Re;

    if (rnd(seed) < p_Re) {
        pr = Re / p_Re;
        return d_re;
    } else {
        pr = (1.0f - Re) / (1.0f - p_Re);
        return d_tr;
    }
}

// ---------------------------------------------------------------------------
// optixTrace wrapper
// ---------------------------------------------------------------------------
static __forceinline__ __device__ void trace(
    OptixTraversableHandle handle,
    float3 origin, float3 direction,
    float tmin, float tmax,
    PRD* prd)
{
    unsigned int lo = u32_from_ptr_lo(prd);
    unsigned int hi = u32_from_ptr_hi(prd);
    optixTrace(
        handle, origin, direction, tmin, tmax,
        0.0f,                      // ray time
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,                   // SBT offset, stride, miss index
        lo, hi);
}

// ---------------------------------------------------------------------------
// Ray Generation shader
// Iterative path tracing loop (no recursion - OptiX doesn't support recursive
// optixTrace calls from within CH shaders).
// Equivalent to Phase 1's recursive Radiance() but restructured as a loop.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();

    const unsigned int x = params.tile_origin_x + idx.x;
    const unsigned int y = params.tile_origin_y + idx.y;

    unsigned int seed = (y * params.width + x) * 1973u + params.subframe_index * 9277u;
    lcg(seed); // warm up

    float3 L = make_float3(0.0f, 0.0f, 0.0f);

    const unsigned int spp        = params.samples_per_launch;
    const unsigned int max_bounce = params.max_bounces;

    for (unsigned int s = 0; s < spp; ++s) {
        // Tent filter (matching Phase 1 exactly)
        float u1 = 2.0f * rnd(seed);
        float u2 = 2.0f * rnd(seed);
        float dx = (u1 < 1.0f) ? sqrtf(u1) - 1.0f : 1.0f - sqrtf(2.0f - u1);
        float dy = (u2 < 1.0f) ? sqrtf(u2) - 1.0f : 1.0f - sqrtf(2.0f - u2);

        // Map pixel to [-0.5, 0.5] with jitter, then into world space
        float fx = ((float)x + 0.5f + dx) / (float)params.width  - 0.5f;
        float fy = ((float)y + 0.5f + dy) / (float)params.height - 0.5f;

        float3 d = params.cx * fx + params.cy * fy;
        float3 gaze = make_float3(0.0f, -0.042612f, -1.0f);
        float gnorm = sqrtf(dot(gaze,gaze));
        gaze = gaze / gnorm;
        d = d + gaze;

        // origin = eye + d * 130.0f advances the ray to the image plane
        // (smallpt convention; smallpt uses the same 130.0 offset). Earlier
        // experiments with `gaze` in place of `d` produced an over-zoomed
        // view; smaller offsets re-introduced an aperture-edge ghost ring.
        float3 ray_origin    = params.eye + d * 130.0f;
        float3 ray_direction = normalize(d);

        // Path trace
        PRD prd;
        prd.radiance   = make_float3(0.0f, 0.0f, 0.0f);
        prd.throughput = make_float3(1.0f, 1.0f, 1.0f);
        prd.origin     = ray_origin;
        prd.direction  = ray_direction;
        prd.seed       = seed;
        prd.depth      = 0;
        prd.done       = 0;

        for (unsigned int bounce = 0; bounce < max_bounce && !prd.done; ++bounce) {
            trace(params.handle, prd.origin, prd.direction, 1e-4f, 1e16f, &prd);
        }

        seed = prd.seed;
        L += prd.radiance * (1.0f / spp);
    }

    // Clamp and accumulate
    L.x = fminf(fmaxf(L.x, 0.0f), 1.0f);
    L.y = fminf(fmaxf(L.y, 0.0f), 1.0f);
    L.z = fminf(fmaxf(L.z, 0.0f), 1.0f);

    unsigned int pixel = (params.height - 1 - y) * params.width + x; // flip Y
    if (params.subframe_index == 0) {
        params.accum_buffer[pixel] = make_float4(L.x, L.y, L.z, 1.0f);
    } else {
        float4 prev = params.accum_buffer[pixel];
        float t = 1.0f / (float)(params.subframe_index + 1);
        params.accum_buffer[pixel] = make_float4(
            prev.x + t * (L.x - prev.x),
            prev.y + t * (L.y - prev.y),
            prev.z + t * (L.z - prev.z),
            1.0f
        );
    }
}

// ---------------------------------------------------------------------------
// Miss shader - background is black (no environment light in Cornell box)
// ---------------------------------------------------------------------------
extern "C" __global__ void __miss__ms() {
    unsigned int lo = optixGetPayload_0();
    unsigned int hi = optixGetPayload_1();
    PRD* prd = ptr_from_u32(lo, hi);
    prd->done = 1;
}

// ---------------------------------------------------------------------------
// Compute surface normal.
//   - Sphere: analytic outward normal (hit_pos - sphere_center). Returned
//     UNFLIPPED so refract_or_reflect can use dot(N, ray_dir) to decide
//     whether the ray is entering or exiting the medium. The MAT_SPECULAR
//     and MAT_DIFFUSE branches do their own flip-toward-ray locally.
//
//     We keep the analytic normal even though the sphere geometry is now
//     tessellated. Why: the geometry is triangles for traversal speed
//     (single GAS, no IAS, no built-in sphere primitives), but shading uses
//     the analytic normal for sharp glossy reflections. This is the
//     standard "smooth-shaded tessellated sphere" pattern: triangle
//     intersection for t, analytic normal for N.
//
//   - Wall (triangle): face normal flipped to face the incoming ray, so
//     back-face hits along wall-skirt seams shade with the room-side
//     orientation rather than going black.
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 compute_normal(
    float3 hit_pos, float3 ray_dir, const HitGroupData* data)
{
    if (data->is_sphere) {
        // Outward geometric normal, NOT flipped to face the ray. Phase 2's
        // built-in-sphere path returned this same unflipped normal, and the
        // refraction code below depends on that contract.
        return normalize(hit_pos - data->sphere_center);
    }

    const unsigned int prim_idx    = optixGetPrimitiveIndex();
    const unsigned int sbt_gas_idx = optixGetSbtGASIndex();
    const OptixTraversableHandle gas = optixGetGASTraversableHandle();
    float3 verts[3];
    optixGetTriangleVertexData(gas, prim_idx, sbt_gas_idx, 0.0f, verts);
    float3 N_obj = normalize(cross(verts[1]-verts[0], verts[2]-verts[0]));
    float3 N = normalize(optixTransformNormalFromObjectToWorldSpace(N_obj));
    return dot(N, ray_dir) < 0.0f ? N : make_float3(-N.x, -N.y, -N.z);
}

// ---------------------------------------------------------------------------
// Closest-hit shader (shared by all geometry).
// Evaluates material, accumulates emission, computes scatter direction.
// ---------------------------------------------------------------------------
extern "C" __global__ void __closesthit__ch() {
    unsigned int lo = optixGetPayload_0();
    unsigned int hi = optixGetPayload_1();
    PRD* prd = ptr_from_u32(lo, hi);

    const HitGroupData* data = reinterpret_cast<const HitGroupData*>(optixGetSbtDataPointer());

    const float  t_hit   = optixGetRayTmax();
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 hit_pos = optixGetWorldRayOrigin() + t_hit * ray_dir;

    float3 N = compute_normal(hit_pos, ray_dir, data);

    prd->radiance += prd->throughput * data->emission;

    // Russian roulette on throughput BEFORE albedo multiply.
    if (prd->depth > 4) {
        float cont = fminf(fmaxf3(prd->throughput), 1.0f);
        if (rnd(prd->seed) >= cont) { prd->done = 1; return; }
        prd->throughput /= cont;
    }

    prd->throughput *= data->albedo;

    // Scatter
    float3 new_dir;
    switch (data->material) {

    case MAT_SPECULAR: {
        // Use the same ray-oriented normal as diffuse so that secondary rays
        // hitting the mirror sphere from inside (e.g. after a refraction
        // bounce) reflect correctly instead of firing into the front wall.
        float3 w = dot(N, ray_dir) < 0.0f ? N : make_float3(-N.x, -N.y, -N.z);
        new_dir = reflect(ray_dir, w);
        break;
    }

    case MAT_REFRACTIVE: {
        float pr;
        new_dir = refract_or_reflect(ray_dir, N, 1.0f, 1.5f, pr, prd->seed);
        prd->throughput *= pr;
        break;
    }

    default: { // MAT_DIFFUSE
        // Orient the shading hemisphere so it faces the incoming ray.
        float3 w = dot(N, ray_dir) < 0.0f ? N : make_float3(-N.x,-N.y,-N.z);
        float3 u, v;
        onb_from_normal(w, u, v);
        float3 sample = cosine_sample_hemisphere(rnd(prd->seed), rnd(prd->seed));
        new_dir = normalize(sample.x * u + sample.y * v + sample.z * w);
        break;
    }
    }

    prd->origin    = hit_pos;
    prd->direction = new_dir;
    prd->depth++;
}
