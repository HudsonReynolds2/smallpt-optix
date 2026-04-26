#pragma once

// SCENE_NAME: default
//
// Phase 3 scene: walls AND spheres are tessellated triangle meshes.
// The mirror, glass, and light spheres -- which were OptiX built-in
// spheres in phase 2 -- are now lat/lon UV-tessellated triangle meshes,
// so the entire scene is one triangle GAS, no IAS.

#include "shared.h"

#include <cstdint>
#include <vector>
#include <cmath>

#ifndef SPHERE_TESS_LAT
#define SPHERE_TESS_LAT 64
#endif
#ifndef SPHERE_TESS_LON
#define SPHERE_TESS_LON 64
#endif

// (Wall tessellation explanation preserved from phase 2 below.)
//
// ---------------------------------------------------------------------------
// Why each wall is extended past the visible room (skirts):
// Phase 1's wall spheres have radius 1e5, so they're effectively infinite
// planes. Phase 2 originally used quads sized exactly to the room. That
// left gaps at the seams between adjacent walls; primary rays at extreme
// camera angles escaped past the geometry and struck the radius-600 light
// sphere extending above the ceiling.
//
// Why WALL_EXT is 500: large enough to swallow escapes, small enough that
// float precision over the triangles stays clean.
//
// Why walls are TESSELLATED: float32 t-error along a 170-unit triangle
// was occasionally larger than the light-sphere protrusion, producing
// horizontal banding on the ceiling. Subdividing brings the largest tri
// to ~6 units and resolves the closest-hit cleanly.
// ---------------------------------------------------------------------------

#define WALL_EXT          500.0f
#define WALL_TESS_N       16

struct WallDef {
    int          const_axis;
    float        const_value;
    int          u_axis;
    int          v_axis;
    float        u_min, u_max;
    float        v_min, v_max;
    int          normal_sign;
    float3       emission;
    float3       albedo;
    MaterialType material;
};

static const WallDef g_walls[] = {
    { 0, 1.0f,    2, 1,    0.0f, 170.0f,   0.0f, 81.6f,   +1,
      {0,0,0}, {0.75f, 0.25f, 0.25f}, MAT_DIFFUSE },
    { 0, 99.0f,   2, 1,    0.0f, 170.0f,   0.0f, 81.6f,   -1,
      {0,0,0}, {0.25f, 0.25f, 0.75f}, MAT_DIFFUSE },
    { 2, 0.0f,    0, 1,    1.0f, 99.0f,    0.0f, 81.6f,   +1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },
    { 2, 300.0f,  0, 1,    1.0f, 99.0f,    0.0f, 81.6f,   -1,
      {0,0,0}, {0,0,0}, MAT_DIFFUSE },
    { 1, 0.0f,    0, 2,    1.0f, 99.0f,    0.0f, 170.0f,  +1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },
    { 1, 81.6f,   0, 2,    1.0f, 99.0f,    0.0f, 170.0f,  -1,
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },
};
static const int NUM_WALLS = sizeof(g_walls) / sizeof(g_walls[0]);

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

    const bool right_handed = ((w.v_axis - w.u_axis + 3) % 3) == 1;
    const bool standard_winding = (right_handed == (w.normal_sign > 0));

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

static inline int triangles_per_wall(int N) { return (N + 2) * (N + 2) * 2; }

// ---------------------------------------------------------------------------
// Sphere definitions (matched to canonical smallpt)
// ---------------------------------------------------------------------------
struct SphereDef {
    float        radius;
    float3       center;
    float3       emission;
    float3       albedo;
    MaterialType material;
};

static const SphereDef g_spheres[] = {
    { 16.5f, {27.0f,        16.5f,  47.0f}, {0,0,0},      {0.999f,0.999f,0.999f}, MAT_SPECULAR    },
    { 16.5f, {73.0f,        16.5f,  78.0f}, {0,0,0},      {0.999f,0.999f,0.999f}, MAT_REFRACTIVE  },
    { 600.f, {50.0f, 681.6f-0.27f, 81.6f},  {12,12,12},   {0,0,0},                MAT_DIFFUSE     },
};
static const int NUM_SPHERES = sizeof(g_spheres) / sizeof(g_spheres[0]);

// ---------------------------------------------------------------------------
// Sphere tessellation: standard lat/lon UV sphere.
//
// lat_segments rows of latitude (poles at top/bottom), lon_segments columns
// of longitude. The poles are degenerate triangle fans; the body is a grid
// of quads each split into 2 triangles. CCW winding when viewed from
// outside the sphere -> outward-facing normals via cross(e1,e2) in
// compute_normal (which then auto-orients via dot(N, ray_dir)).
//
// Outputs raw vertex stream (no index buffer), 3 verts/triangle, matching
// the wall layout.
// ---------------------------------------------------------------------------
static inline int triangles_per_sphere(int lat, int lon) {
    // Top cap: lon triangles. Bottom cap: lon triangles.
    // Body: (lat - 2) rows of lon quads = (lat - 2) * lon * 2 triangles.
    return 2 * lon + (lat - 2) * lon * 2;
}

static inline float3 sphere_point(float3 center, float radius,
                                  int i_lat, int i_lon,
                                  int lat_segments, int lon_segments)
{
    // Latitude theta in [0, pi]: 0 = north pole (+y), pi = south pole (-y).
    float theta = (float)i_lat * (float)M_PI / (float)lat_segments;
    float phi   = (float)i_lon * 2.0f * (float)M_PI / (float)lon_segments;

    float sin_t = sinf(theta);
    float cos_t = cosf(theta);
    float sin_p = sinf(phi);
    float cos_p = cosf(phi);

    return make_float3(
        center.x + radius * sin_t * cos_p,
        center.y + radius * cos_t,
        center.z + radius * sin_t * sin_p
    );
}

static inline void tessellate_sphere(float3 center, float radius,
                                     int lat_segments, int lon_segments,
                                     std::vector<float3>& verts_out)
{
    const int tris = triangles_per_sphere(lat_segments, lon_segments);
    verts_out.reserve(verts_out.size() + tris * 3);

    float3 north = make_float3(center.x, center.y + radius, center.z);
    float3 south = make_float3(center.x, center.y - radius, center.z);

    // Top cap (i_lat = 0 is north pole, i_lat = 1 is first ring below).
    for (int j = 0; j < lon_segments; ++j) {
        float3 a = sphere_point(center, radius, 1, j,     lat_segments, lon_segments);
        float3 b = sphere_point(center, radius, 1, j + 1, lat_segments, lon_segments);
        // CCW from outside: north, b, a (so cross(b-north, a-north) points outward)
        verts_out.push_back(north);
        verts_out.push_back(b);
        verts_out.push_back(a);
    }

    // Body: rings between i_lat=1 and i_lat=lat_segments-1.
    for (int i = 1; i < lat_segments - 1; ++i) {
        for (int j = 0; j < lon_segments; ++j) {
            float3 p00 = sphere_point(center, radius, i,     j,     lat_segments, lon_segments);
            float3 p10 = sphere_point(center, radius, i,     j + 1, lat_segments, lon_segments);
            float3 p11 = sphere_point(center, radius, i + 1, j + 1, lat_segments, lon_segments);
            float3 p01 = sphere_point(center, radius, i + 1, j,     lat_segments, lon_segments);
            // Quad split into two CCW-from-outside triangles
            verts_out.push_back(p00); verts_out.push_back(p10); verts_out.push_back(p11);
            verts_out.push_back(p00); verts_out.push_back(p11); verts_out.push_back(p01);
        }
    }

    // Bottom cap.
    for (int j = 0; j < lon_segments; ++j) {
        float3 a = sphere_point(center, radius, lat_segments - 1, j,     lat_segments, lon_segments);
        float3 b = sphere_point(center, radius, lat_segments - 1, j + 1, lat_segments, lon_segments);
        // CCW from outside: south, a, b
        verts_out.push_back(south);
        verts_out.push_back(a);
        verts_out.push_back(b);
    }
}

// ---------------------------------------------------------------------------
// Build the full scene triangle list.
// PHASE 3: walls THEN spheres in one flat array. SBT index per triangle
// goes 0..NUM_WALLS-1 for walls, then NUM_WALLS..NUM_WALLS+NUM_SPHERES-1
// for spheres, so each surface still has its own hit group record.
// ---------------------------------------------------------------------------
static inline void build_scene_geometry(int wall_N, int sph_lat, int sph_lon,
                                        std::vector<float3>&    verts_out,
                                        std::vector<uint32_t>& tri_sbt_idx_out)
{
    verts_out.clear();
    tri_sbt_idx_out.clear();

    const int wall_tris = triangles_per_wall(wall_N);
    const int sph_tris  = triangles_per_sphere(sph_lat, sph_lon);
    tri_sbt_idx_out.reserve(NUM_WALLS * wall_tris + NUM_SPHERES * sph_tris);

    for (int w = 0; w < NUM_WALLS; ++w) {
        tessellate_wall(g_walls[w], wall_N, verts_out);
        for (int t = 0; t < wall_tris; ++t)
            tri_sbt_idx_out.push_back((uint32_t)w);
    }

    for (int s = 0; s < NUM_SPHERES; ++s) {
        tessellate_sphere(g_spheres[s].center, g_spheres[s].radius,
                          sph_lat, sph_lon, verts_out);
        for (int t = 0; t < sph_tris; ++t)
            tri_sbt_idx_out.push_back((uint32_t)(NUM_WALLS + s));
    }
}

// SBT layout: one hit group record per wall, then one per sphere.
static const int NUM_HG_RECORDS = NUM_WALLS + NUM_SPHERES;