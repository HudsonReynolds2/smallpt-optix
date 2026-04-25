#pragma once

// SCENE_NAME: default
//
// The standard Cornell box scene used by phase 2. Matches phase 1
// (cu-smallpt) output pixel-for-pixel at 4:3 aspect ratios. Walls are
// tessellated triangle meshes (visible room face + extension skirts) so
// large flat surfaces don't suffer float32 t-value precision loss when
// they fight a nearby curved surface for a closest-hit (see the precision
// note further down).

#include "shared.h"

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Cornell box scene description
//
// Visible-plane coordinates (unchanged from canonical smallpt):
//   left wall:   x = 1
//   right wall:  x = 99
//   floor:       y = 0
//   ceiling:     y = 81.6
//   back wall:   z = 0
//   front wall:  z = 300 (behind the camera; black, never visible directly)
//
// ---------------------------------------------------------------------------
// Why each wall is extended past the visible room (skirts):
//
// Phase 1's wall spheres have radius 1e5, so they're effectively infinite
// planes. Phase 2 originally used quads sized exactly to the room. That
// left gaps at the seams between adjacent walls; primary rays at extreme
// camera angles escaped past the geometry and struck the radius-600 light
// sphere extending above the ceiling, producing bright artifacts and
// contaminated reflections.
//
// Fix: each wall is widened (in the two directions parallel to its plane)
// past the room bounds, so escape paths get filled by the same wall
// material that would have been there in phase 1's infinite-plane scene.
//
// ---------------------------------------------------------------------------
// Why WALL_EXT is 500 and not 1e5:
//
// A previous version used WALL_EXT = 1e5 (matching phase 1's sphere radii).
// That broke the ceiling cutout where the light sphere protrudes: OptiX's
// float32 triangle intersector loses precision over very large triangles,
// and a single ~2e5-wide ceiling triangle had t-error larger than the
// sphere/ceiling separation, so the ceiling won the closest-hit test
// instead of the light sphere and the disc disappeared.
//
// 500 units is roughly 5x the room dimensions: large enough to swallow
// every camera-frustum escape and every secondary-bounce escape, while
// small enough that float precision over the triangles stays clean.
//
// ---------------------------------------------------------------------------
// Why the walls are TESSELLATED (not single quads):
//
// Even after dropping WALL_EXT from 1e5 to 500, the visible portion of
// each wall is still a single ~100x170 unit triangle pair. With the light
// sphere protruding only ~0.27 units below the ceiling, the float32
// t-value error along that 170-unit triangle was occasionally larger than
// 0.27 in pixels near where the sphere meets the ceiling. Result: a band
// of pixels where the ceiling won the closest-hit race instead of the
// light sphere, producing a visible horizontal banding artifact across
// the ceiling.
//
// The fix is straightforward: subdivide each wall into a grid of small
// triangles. When the largest triangle a primary ray can intersect is
// only ~6 units across instead of 170 units, the relative t-error per
// intersection drops by an order of magnitude and the closest-hit test
// resolves cleanly. This is also what every production renderer does --
// real Cornell box scenes are not single huge quads.
//
// Tessellation scheme (per wall):
//   - Visible region (the actual room face) is split into a uniform NxN
//     grid of cells (default N=16, so 16x16 = 256 cells = 512 triangles).
//   - Each of the four extension skirts (left/right/top/bottom edges
//     beyond the room) is one large quad (= 2 triangles).
//   - Each of the four extension corners is one large quad (= 2 triangles).
//   - Total: NxN + 8 quads per wall.
//
// The grid is built so that visible-region cells share their boundary
// vertices with the adjacent skirt cells (no T-junctions, watertight).
// The skirts can stay coarse because they are never visible to primary
// rays -- they only catch secondary rays escaping through wall seams,
// where precision near the room boundary is irrelevant.
// ---------------------------------------------------------------------------

#define WALL_EXT          500.0f
#define WALL_TESS_N       16     // visible-region grid resolution per side

// A wall is an axis-aligned rectangular face of the room with extension
// skirts. The plane is fixed by (const_axis, const_value); the two
// in-plane axes are (u_axis, v_axis), with the visible region spanning
// [u_min, u_max] x [v_min, v_max]. normal_sign is +1 or -1 along
// const_axis (which side of the plane the room is on).
//
// CCW winding (when viewed from inside the room, i.e. from the side the
// normal points to) is encoded by ordering quad vertices as:
//     (u_min, v_min) -> (u_max, v_min) -> (u_max, v_max) -> (u_min, v_max)
// then flipping if normal_sign is negative.
struct WallDef {
    int          const_axis;     // 0=x, 1=y, 2=z
    float        const_value;
    int          u_axis;         // 0=x, 1=y, 2=z (perpendicular to const_axis)
    int          v_axis;         // 0=x, 1=y, 2=z (perpendicular to both)
    float        u_min, u_max;   // visible-region bounds along u_axis
    float        v_min, v_max;   // visible-region bounds along v_axis
    int          normal_sign;    // +1 or -1: which side the room is on
    float3       emission;
    float3       albedo;
    MaterialType material;
};

static const WallDef g_walls[] = {
    // Left wall (plane x=1), red, normal +x. u=z, v=y.
    { 0, 1.0f,    2, 1,    0.0f, 170.0f,   0.0f, 81.6f,   +1,
      {0,0,0}, {0.75f, 0.25f, 0.25f}, MAT_DIFFUSE },

    // Right wall (plane x=99), blue, normal -x. u=z, v=y.
    { 0, 99.0f,   2, 1,    0.0f, 170.0f,   0.0f, 81.6f,   -1,
      {0,0,0}, {0.25f, 0.25f, 0.75f}, MAT_DIFFUSE },

    // Back wall (plane z=0), white, normal +z. u=x, v=y.
    { 2, 0.0f,    0, 1,    1.0f, 99.0f,    0.0f, 81.6f,   +1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Front wall (plane z=300), black, normal -z. u=x, v=y.
    // z=300 is past eye.z=295.6 so ray origins computed as eye+d*130 are
    // always behind this wall; otherwise rays reflecting off the mirror
    // sphere's bottom would strike z=170 as a valid forward hit and
    // return black, unlike phase 1 where the camera is inside the front
    // wall sphere (radius 1e5 centered at z=-99999.83, surface at z=170.17).
    { 2, 300.0f,  0, 1,    1.0f, 99.0f,    0.0f, 81.6f,   -1,
      {0,0,0}, {0,0,0}, MAT_DIFFUSE },

    // Floor (plane y=0), white, normal +y. u=x, v=z.
    { 1, 0.0f,    0, 2,    1.0f, 99.0f,    0.0f, 170.0f,  +1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Ceiling (plane y=81.6), white, normal -y. u=x, v=z.
    { 1, 81.6f,   0, 2,    1.0f, 99.0f,    0.0f, 170.0f,  -1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },
};
static const int NUM_WALLS = sizeof(g_walls) / sizeof(g_walls[0]);

// ---------------------------------------------------------------------------
// Build the triangle list for a single wall.
//
// Layout: a (N+2) x (N+2) grid of vertices covering the extended quad,
// with grid lines at:
//     u: [u_min - WALL_EXT, u_min, u_min + (u_max-u_min)/N, ..., u_max,
//         u_max + WALL_EXT]
//     v: same scheme.
// Two triangles per cell, written as 6 standalone vertices (no index
// buffer -- matches main.cpp's existing layout for simplicity).
//
// CCW winding when viewed from the room side: depends on normal_sign.
// ---------------------------------------------------------------------------
static inline float3 make_axis_point(const WallDef& w, float u, float v) {
    float xyz[3];
    xyz[w.const_axis] = w.const_value;
    xyz[w.u_axis]     = u;
    xyz[w.v_axis]     = v;
    return make_float3(xyz[0], xyz[1], xyz[2]);
}

static inline void tessellate_wall(const WallDef& w, int N,
                                   std::vector<float3>& verts_out)
{
    // Build 1D coordinate arrays along u and v (length N+3).
    // Indices 0 and N+2 are the skirt extents; indices 1..N+1 are the
    // uniform subdivision of the visible region (so cells [0..N) are the
    // visible grid, and the outermost cells on each side are the skirts).
    float u_coord[WALL_TESS_N + 3];
    float v_coord[WALL_TESS_N + 3];
    u_coord[0]     = w.u_min - WALL_EXT;
    u_coord[N + 2] = w.u_max + WALL_EXT;
    v_coord[0]     = w.v_min - WALL_EXT;
    v_coord[N + 2] = w.v_max + WALL_EXT;
    for (int i = 0; i <= N; ++i) {
        float t = (float)i / (float)N;
        u_coord[i + 1] = w.u_min + t * (w.u_max - w.u_min);
        v_coord[i + 1] = w.v_min + t * (w.v_max - w.v_min);
    }

    // Determine the orientation flip. With unit basis vectors e_u, e_v, e_n
    // along (u_axis, v_axis, const_axis), the axis triple (e_u, e_v, e_n) is
    // right-handed (cross(e_u, e_v) = +e_n) when it's a cyclic permutation of
    // (x,y,z). That happens iff (v_axis - u_axis) mod 3 == 1.
    //   - Right-handed and normal_sign=+1   -> CCW winding emits tri pointing +const_axis: OK
    //   - Right-handed and normal_sign=-1   -> need to flip
    //   - Left-handed  and normal_sign=+1   -> need to flip
    //   - Left-handed  and normal_sign=-1   -> already pointing the right way
    const bool right_handed = ((w.v_axis - w.u_axis + 3) % 3) == 1;
    const bool standard_winding = (right_handed == (w.normal_sign > 0));

    // For each (N+2) x (N+2) cell, emit two triangles.
    // Standard order is:
    //   tri 0: (p00, p10, p11)
    //   tri 1: (p00, p11, p01)
    // where cross(p10-p00, p11-p00) gives +cross(e_u, e_v). When that
    // direction matches the room-side normal we keep this order; otherwise
    // we reverse the per-triangle vertex order.
    const int total_cells = (N + 2) * (N + 2);
    verts_out.reserve(verts_out.size() + total_cells * 6);

    for (int j = 0; j < N + 2; ++j) {
        for (int i = 0; i < N + 2; ++i) {
            float u0 = u_coord[i],     u1 = u_coord[i + 1];
            float v0 = v_coord[j],     v1 = v_coord[j + 1];

            float3 p00 = make_axis_point(w, u0, v0);
            float3 p10 = make_axis_point(w, u1, v0);
            float3 p11 = make_axis_point(w, u1, v1);
            float3 p01 = make_axis_point(w, u0, v1);

            if (standard_winding) {
                verts_out.push_back(p00); verts_out.push_back(p10); verts_out.push_back(p11);
                verts_out.push_back(p00); verts_out.push_back(p11); verts_out.push_back(p01);
            } else {
                verts_out.push_back(p00); verts_out.push_back(p11); verts_out.push_back(p10);
                verts_out.push_back(p00); verts_out.push_back(p01); verts_out.push_back(p11);
            }
        }
    }
}

// Triangles per wall after tessellation: (N+2)^2 cells x 2 tris/cell.
static inline int triangles_per_wall(int N) { return (N + 2) * (N + 2) * 2; }

// ---------------------------------------------------------------------------
// Build the full scene triangle list.
//
// verts_out:        flat vertex array, 3 verts per triangle (no index buffer)
// tri_wall_idx_out: parallel array of length num_triangles, giving the
//                   wall index (0..NUM_WALLS-1) each triangle belongs to.
//                   Used as the SBT index offset so all triangles of one
//                   wall share that wall's hit group record.
// ---------------------------------------------------------------------------
static inline void build_wall_geometry(int N,
                                       std::vector<float3>&    verts_out,
                                       std::vector<uint32_t>& tri_wall_idx_out)
{
    verts_out.clear();
    tri_wall_idx_out.clear();
    const int tris_per_wall = triangles_per_wall(N);
    tri_wall_idx_out.reserve(NUM_WALLS * tris_per_wall);

    for (int w = 0; w < NUM_WALLS; ++w) {
        tessellate_wall(g_walls[w], N, verts_out);
        for (int t = 0; t < tris_per_wall; ++t)
            tri_wall_idx_out.push_back((uint32_t)w);
    }
}

struct SphereDef {
    float        radius;
    float3       center;
    float3       emission;
    float3       albedo;
    MaterialType material;
};

// Sphere definitions match canonical smallpt / cu-smallpt exactly.
static const SphereDef g_spheres[] = {
    { 16.5f, {27.0f,        16.5f,  47.0f}, {0,0,0},      {0.999f,0.999f,0.999f}, MAT_SPECULAR    }, // Mirror
    { 16.5f, {73.0f,        16.5f,  78.0f}, {0,0,0},      {0.999f,0.999f,0.999f}, MAT_REFRACTIVE  }, // Glass
    { 600.f, {50.0f, 681.6f-0.27f, 81.6f},  {12,12,12},   {0,0,0},                MAT_DIFFUSE     }, // Light
};
static const int NUM_SPHERES = sizeof(g_spheres) / sizeof(g_spheres[0]);

// SBT layout: one hit group record per wall (NUM_WALLS), then one per sphere.
// Triangles within a wall share a record via sbtIndexOffsetBuffer.
static const int NUM_HG_RECORDS = NUM_WALLS + NUM_SPHERES;
