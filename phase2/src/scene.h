#pragma once

#include "shared.h"

// ---------------------------------------------------------------------------
// Cornell box scene description
//
// Walls are flat quads (12 triangles total) instead of radius-1e5 spheres.
//
// IMPORTANT: each wall quad is extended FAR beyond the visible room bounds.
// Phase 1's wall spheres have radius 1e5, so they're effectively infinite
// planes from the camera's perspective. Phase 2 originally used quads sized
// exactly to the room (x in [1,99], y in [0,81.6], z in [0,170]); this left
// gaps at the seams between adjacent walls that primary and secondary rays
// could escape through, hitting the giant radius-600 light sphere that
// extends far above the ceiling. Result: bright white wedges in the top
// corners and contaminated reflections inside the glass/mirror spheres.
//
// Visible-plane coordinates (unchanged from canonical smallpt):
//   left wall:   x = 1
//   right wall:  x = 99
//   floor:       y = 0
//   ceiling:     y = 81.6
//   back wall:   z = 0
//   front wall:  z = 170 (behind the camera; black, never visible directly)
//
// EXT is the half-extent each wall is widened by along the two directions
// orthogonal to its normal. Big enough to swallow any escaping ray.
//
// Spheres kept (small radii, hardware sphere is fine here):
//   mirror, glass, light
// ---------------------------------------------------------------------------

struct WallDef {
    float3       v0, v1, v2, v3; // quad corners (CCW when viewed from inside)
    float3       emission;
    float3       albedo;
    MaterialType material;
};

// Extension distance: walls are pushed this far beyond visible room bounds
// in the two directions orthogonal to their normal. 1e5 matches phase 1's
// sphere radii so escape paths are closed at the same scale.
#define WALL_EXT 1.0e5f

static const WallDef g_walls[] = {
    // Left wall (plane x=1), red, normal +x.
    // Extended in y and z far beyond the room.
    { {1.0f, 0.0f - WALL_EXT, 0.0f - WALL_EXT},
      {1.0f, 81.6f + WALL_EXT, 0.0f - WALL_EXT},
      {1.0f, 81.6f + WALL_EXT, 170.0f + WALL_EXT},
      {1.0f, 0.0f - WALL_EXT, 170.0f + WALL_EXT},
      {0,0,0}, {0.75f, 0.25f, 0.25f}, MAT_DIFFUSE },

    // Right wall (plane x=99), blue, normal -x.
    { {99.0f, 0.0f - WALL_EXT, 170.0f + WALL_EXT},
      {99.0f, 81.6f + WALL_EXT, 170.0f + WALL_EXT},
      {99.0f, 81.6f + WALL_EXT, 0.0f - WALL_EXT},
      {99.0f, 0.0f - WALL_EXT, 0.0f - WALL_EXT},
      {0,0,0}, {0.25f, 0.25f, 0.75f}, MAT_DIFFUSE },

    // Back wall (plane z=0), white, normal +z.
    // Extended in x and y.
    { {99.0f + WALL_EXT, 0.0f - WALL_EXT,  0.0f},
      {99.0f + WALL_EXT, 81.6f + WALL_EXT, 0.0f},
      {1.0f - WALL_EXT,  81.6f + WALL_EXT, 0.0f},
      {1.0f - WALL_EXT,  0.0f - WALL_EXT,  0.0f},
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Front wall (plane z=170), black, normal -z.
    // (Behind the camera after the d*130 origin push, but extended for
    // consistency in case secondary rays end up traveling toward +z.)
    { {1.0f - WALL_EXT,  0.0f - WALL_EXT,  170.0f},
      {1.0f - WALL_EXT,  81.6f + WALL_EXT, 170.0f},
      {99.0f + WALL_EXT, 81.6f + WALL_EXT, 170.0f},
      {99.0f + WALL_EXT, 0.0f - WALL_EXT,  170.0f},
      {0,0,0}, {0,0,0}, MAT_DIFFUSE },

    // Floor (plane y=0), white, normal +y.
    // Extended in x and z.
    { {1.0f - WALL_EXT,  0.0f, 170.0f + WALL_EXT},
      {99.0f + WALL_EXT, 0.0f, 170.0f + WALL_EXT},
      {99.0f + WALL_EXT, 0.0f, 0.0f - WALL_EXT},
      {1.0f - WALL_EXT,  0.0f, 0.0f - WALL_EXT},
      {0,0,0}, {0.75f, 0.75f, 0.75f}, MAT_DIFFUSE },

    // Ceiling (plane y=81.6), white, normal -y.
    // Extended in x and z. This is the critical one: the unextended ceiling
    // was the main escape path for rays heading up-and-outward, which then
    // hit the radius-600 light sphere above the room and produced the bright
    // white wedges in the top corners of the image.
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
    { 600.f, {50.0f, 681.6f-0.27f, 81.6f},  {12,12,12},   {0,0,0},                MAT_DIFFUSE     }, // Light
};
static const int NUM_SPHERES = sizeof(g_spheres) / sizeof(g_spheres[0]);

// Total SBT hit group records: triangles first (one per triangle), then spheres.
static const int NUM_HG_RECORDS = NUM_TRIANGLES + NUM_SPHERES;
