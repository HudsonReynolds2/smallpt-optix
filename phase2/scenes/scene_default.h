#pragma once

// SCENE_NAME: default
//
// The standard Cornell box scene used by phase 2. Matches phase 1
// (cu-smallpt) output pixel-for-pixel at 4:3 aspect ratios. Walls are flat
// triangles widened past the room bounds (WALL_EXT) so escape paths around
// the camera frustum and at wall seams are closed.

#include "shared.h"

// ---------------------------------------------------------------------------
// Cornell box scene description
//
// Walls are flat quads (12 triangles total) instead of radius-1e5 spheres.
//
// Visible-plane coordinates (unchanged from canonical smallpt):
//   left wall:   x = 1
//   right wall:  x = 99
//   floor:       y = 0
//   ceiling:     y = 81.6
//   back wall:   z = 0
//   front wall:  z = 170 (behind the camera; black, never visible directly)
//
// ---------------------------------------------------------------------------
// Why each wall is extended past the visible room:
//
// Phase 1's wall spheres have radius 1e5, so they're effectively infinite
// planes. Phase 2 originally used quads sized exactly to the room. That left
// gaps at the seams between adjacent walls; primary rays at extreme camera
// angles escaped past the geometry and struck the radius-600 light sphere
// extending above the ceiling, producing bright artifacts and contaminated
// reflections.
//
// Fix: each wall is widened (in the two directions parallel to its plane)
// past the room bounds, so escape paths get filled by the same wall material
// that would have been there in phase 1's infinite-plane scene.
//
// ---------------------------------------------------------------------------
// Why WALL_EXT is 500 and not 1e5:
//
// A previous version used WALL_EXT = 1e5 (matching phase 1's sphere radii).
// That broke the ceiling cutout where the light sphere protrudes: the light
// only protrudes 0.27 units below the ceiling at y=81.33, and OptiX's float32
// triangle intersector loses precision over very large triangles. With a
// ~2e5-wide ceiling triangle, the t-value error at the disc was larger than
// 0.27, so the ceiling won the closest-hit test instead of the light sphere
// and the disc disappeared.
//
// 500 units is roughly 5x the room dimensions: large enough to swallow every
// camera-frustum escape and every secondary-bounce escape, while small
// enough that float precision over the triangles stays clean.
// ---------------------------------------------------------------------------

struct WallDef {
    float3       v0, v1, v2, v3; // quad corners (CCW when viewed from inside)
    float3       emission;
    float3       albedo;
    MaterialType material;
};

#define WALL_EXT 500.0f

static const WallDef g_walls[] = {
    // Left wall (plane x=1), red, normal +x. Extended in y and z.
    { {1.0f, 0.0f - WALL_EXT, 0.0f - WALL_EXT},
      {1.0f, 81.6f + WALL_EXT, 0.0f - WALL_EXT},
      {1.0f, 81.6f + WALL_EXT, 170.0f + WALL_EXT},
      {1.0f, 0.0f - WALL_EXT, 170.0f + WALL_EXT},
      {0,0,0}, {0.75f, 0.25f, 0.25f}, MAT_DIFFUSE },

    // Right wall (plane x=99), blue, normal -x. Extended in y and z.
    { {99.0f, 0.0f - WALL_EXT, 170.0f + WALL_EXT},
      {99.0f, 81.6f + WALL_EXT, 170.0f + WALL_EXT},
      {99.0f, 81.6f + WALL_EXT, 0.0f - WALL_EXT},
      {99.0f, 0.0f - WALL_EXT, 0.0f - WALL_EXT},
      {0,0,0}, {0.25f, 0.25f, 0.75f}, MAT_DIFFUSE },

    // Back wall (plane z=0), white, normal +z. Extended in x and y.
    { {99.0f + WALL_EXT, 0.0f - WALL_EXT,  0.0f},
      {99.0f + WALL_EXT, 81.6f + WALL_EXT, 0.0f},
      {1.0f - WALL_EXT,  81.6f + WALL_EXT, 0.0f},
      {1.0f - WALL_EXT,  0.0f - WALL_EXT,  0.0f},
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Front wall (plane z=300), black, normal -z. Extended in x and y.
    // Pushed to z=300 (past eye.z=295.6) so that ray origins computed as
    // eye + d*130 always fall behind this wall; otherwise rays reflecting
    // off the mirror sphere's bottom struck z=170 as a valid forward hit
    // and returned black, unlike phase 1 where the camera is inside the
    // front wall sphere (radius 1e5 centered at z=-99999.83, surface at z=170.17).
    { {1.0f - WALL_EXT,  0.0f - WALL_EXT,  300.0f},
      {1.0f - WALL_EXT,  81.6f + WALL_EXT, 300.0f},
      {99.0f + WALL_EXT, 81.6f + WALL_EXT, 300.0f},
      {99.0f + WALL_EXT, 0.0f - WALL_EXT,  300.0f},
      {0,0,0}, {0,0,0}, MAT_DIFFUSE },

    // Floor (plane y=0), white, normal +y. Extended in x and z.
    { {1.0f - WALL_EXT,  0.0f, 170.0f + WALL_EXT},
      {99.0f + WALL_EXT, 0.0f, 170.0f + WALL_EXT},
      {99.0f + WALL_EXT, 0.0f, 0.0f - WALL_EXT},
      {1.0f - WALL_EXT,  0.0f, 0.0f - WALL_EXT},
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Ceiling (plane y=81.6), white, normal -y. Extended in x and z.
    { {1.0f - WALL_EXT,  81.6f, 0.0f - WALL_EXT},
      {99.0f + WALL_EXT, 81.6f, 0.0f - WALL_EXT},
      {99.0f + WALL_EXT, 81.6f, 170.0f + WALL_EXT},
      {1.0f - WALL_EXT,  81.6f, 170.0f + WALL_EXT},
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },
};
static const int NUM_WALLS     = sizeof(g_walls) / sizeof(g_walls[0]);
static const int NUM_TRIANGLES = NUM_WALLS * 2;

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
    // TEMPORARY: light sphere lowered so it protrudes through the ceiling by 5 units
    // instead of 0.27, to break ceiling/light coplanar precision fight that causes
    // horizontal banding on the ceiling. Original (canonical smallpt) line below.
    // { 600.f, {50.0f, 681.6f-0.27f, 81.6f},  {12,12,12},   {0,0,0},                MAT_DIFFUSE     }, // Light
    { 600.f, {50.0f, 681.6f-5.0f,  81.6f},  {12,12,12},   {0,0,0},                MAT_DIFFUSE     }, // Light (TEMP: -5.0f instead of -0.27f)
};
static const int NUM_SPHERES = sizeof(g_spheres) / sizeof(g_spheres[0]);

// Total SBT hit group records: triangles first (one per triangle), then spheres.
static const int NUM_HG_RECORDS = NUM_TRIANGLES + NUM_SPHERES;
