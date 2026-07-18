// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// SoftwareRasterizer.h -- dependency-light CPU rasterizer that renders a posed
// MHR human mesh to a neutral-grey RGBA image with a coverage alpha at the input
// frame resolution. This is the C++ equivalent of the reference Python pyrender
// path (visualization/renderer.py, Renderer.__call__(..., return_rgba=True)).
//
// Pipeline: perspective (pinhole) projection from (fx,fy=focal, cx,cy=center) ->
// triangle setup -> z-buffer -> perspective-correct barycentric interpolation ->
// per-vertex-normal Lambert (or view-stable matcap) neutral-grey shading ->
// coverage alpha with edge SUPERSAMPLING (SSAA) for clean silhouettes.
//
// COORDINATE CONVENTION (reconciles renderer.py into one transform):
//   The reference places the mesh at `verts`, rotates it 180deg about X (OpenGL
//   vs CV), sets camera_translation = cam_t with camera_translation[0] *= -1, and
//   views from the origin with an IntrinsicsCamera. All of that collapses to the
//   standard CV pinhole projection used by the pipeline's perspective_projection:
//       v_cam = verts + cam_t                       (camera at origin, +z forward)
//       u = focal * v_cam.x / v_cam.z + center.x    (image col, x right)
//       v = focal * v_cam.y / v_cam.z + center.y    (image row, y down)
//       depth = v_cam.z                             (metres, >0 in front)
//   The 180-X flip and the cam_t[0] sign in renderer.py are exactly what turn the
//   OpenGL camera math back into this CV projection; see tools/RASTER_REPORT.md.
//
// Dependency-light: only the C++ standard library + MeshTypes.h. Plain float math
// (no Eigen) to keep this track free of a hard Eigen dependency.
//
// OUTPUT COLOR SPACE: linear (no gamma / sRGB encoding). The base grey is a linear
// value; the OFX plugin owns any color-management / display transform.

#pragma once

#include "MeshTypes.h"

namespace hastur {

struct RasterOptions {
  // Neutral-grey albedo (linear). Reference demo default is ivory (1,1,0.9);
  // the clay look uses a mid grey ~0.6. Applied equally to R,G,B (neutral).
  float grey = 0.6f;

  // Ambient term added to every lit pixel (renderer.py ambient_light=(0.3,...)).
  float ambient = 0.3f;

  // Per-light diffuse weight for the (view-space) 3-point Raymond-derived rig.
  float light_weight = 0.42f;

  // Supersampling factor per axis for anti-aliased silhouettes (1 = off, 2 = 4x).
  int ssaa = 2;

  // Emit premultiplied alpha (OFX hosts default to premultiplied). If false,
  // colour is straight (un-associated) with coverage in the alpha channel.
  bool premultiply = false;

  // Shading model: false = Lambert (default clay), true = view-stable matcap.
  bool matcap = false;

  // Near clip in metres; triangles with any vertex at depth <= near are dropped.
  float near_z = 1e-3f;
};

// Renders `mesh` (verts + faces) under `cam` into a W x H RGBA image. Returns an
// RgbaImage sized W x H, row-major HWC, float [0,1]. Alpha is mesh coverage
// (1 where the mesh covers a pixel, fractional on anti-aliased edges, 0 outside).
RgbaImage Render(const Mesh& mesh, const Camera& cam, int W, int H,
                 const RasterOptions& opt);

}  // namespace hastur
