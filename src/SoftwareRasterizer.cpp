// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
//
// SoftwareRasterizer.cpp -- see SoftwareRasterizer.h for the contract and the
// coordinate-convention derivation.

#include "SoftwareRasterizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace hastur {
namespace {

struct Vec3 {
  float x, y, z;
};

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 normalize(const Vec3& v) {
  float n = std::sqrt(dot(v, v));
  if (n < 1e-20f) return {0.0f, 0.0f, 0.0f};
  float inv = 1.0f / n;
  return {v.x * inv, v.y * inv, v.z * inv};
}

// The 3-point Raymond-derived directional rig, expressed in VIEW space (camera at
// the origin, +z into the scene, y down). Directions point from the surface TOWARD
// each light, so a surface facing the camera (normal ~ -z) is the brightest.
// Derived from renderer.py create_raymond_lights (thetas=pi/6, phis=0,2pi/3,4pi/3)
// with the OpenGL->CV z sign flipped. This is a view-stable approximation of the
// reference rig (favouring the clay look + correct silhouette over exact lights).
constexpr Vec3 kLights[3] = {
    {0.5f, 0.0f, -0.8660254f},
    {-0.25f, 0.4330127f, -0.8660254f},
    {-0.25f, -0.4330127f, -0.8660254f},
};

// Per-vertex normals: area-weighted accumulation of face normals, then normalize.
// Computed in the mesh/vertex frame, which equals the camera frame up to the pure
// translation cam_t (translation does not rotate normals).
std::vector<Vec3> ComputeVertexNormals(const std::vector<float>& verts,
                                        const std::vector<int32_t>& faces) {
  const int nv = static_cast<int>(verts.size() / 3);
  const int nf = static_cast<int>(faces.size() / 3);
  std::vector<Vec3> nrm(nv, Vec3{0.0f, 0.0f, 0.0f});
  auto V = [&](int i) -> Vec3 {
    return {verts[3 * i], verts[3 * i + 1], verts[3 * i + 2]};
  };
  for (int f = 0; f < nf; ++f) {
    int i0 = faces[3 * f], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
    Vec3 a = V(i0), b = V(i1), c = V(i2);
    Vec3 fn = cross(b - a, c - a);  // magnitude ~ 2*area -> area weighting
    for (int idx : {i0, i1, i2}) {
      nrm[idx].x += fn.x;
      nrm[idx].y += fn.y;
      nrm[idx].z += fn.z;
    }
  }
  for (auto& n : nrm) n = normalize(n);
  return nrm;
}

// Neutral-grey shade scalar for an (already camera-oriented) normal.
inline float ShadeScalar(const Vec3& n, const Vec3& view_to_cam,
                         const RasterOptions& opt) {
  if (opt.matcap) {
    // View-stable matcap: hemisphere lit from the camera direction.
    float d = std::max(0.0f, dot(n, view_to_cam));
    return opt.ambient + (1.0f - opt.ambient) * d;
  }
  float s = opt.ambient;
  for (const Vec3& L : kLights) s += opt.light_weight * std::max(0.0f, dot(n, L));
  return s;
}

}  // namespace

RgbaImage Render(const Mesh& mesh, const Camera& cam, int W, int H,
                 const RasterOptions& opt) {
  RgbaImage out;
  out.width = W;
  out.height = H;
  out.data.assign(static_cast<size_t>(W) * H * 4, 0.0f);
  if (W <= 0 || H <= 0 || !mesh.faces || mesh.verts.empty()) return out;

  const int ss = std::max(1, opt.ssaa);
  const int SW = W * ss, SH = H * ss;
  const std::vector<int32_t>& faces = *mesh.faces;
  const std::vector<float>& verts = mesh.verts;
  const int nv = static_cast<int>(verts.size() / 3);
  const int nf = static_cast<int>(faces.size() / 3);

  std::vector<Vec3> vnrm = ComputeVertexNormals(verts, faces);

  // Project every vertex to supersampled screen space. Supersampling multiplies
  // the pixel grid by ss, so the intrinsics scale by ss as well.
  const float fx = cam.focal * ss;
  const float fy = cam.focal * ss;
  const float cx = cam.center[0] * ss;
  const float cy = cam.center[1] * ss;
  const Vec3 ct{cam.cam_t[0], cam.cam_t[1], cam.cam_t[2]};

  struct PV {
    float sx, sy;  // supersampled screen position
    float z;       // camera-space depth (metres, >0 in front)
    Vec3 ncam;     // camera-space normal (== vertex normal; translation only)
    bool ok;       // in front of the near plane
  };
  std::vector<PV> pv(nv);
  for (int i = 0; i < nv; ++i) {
    Vec3 vc{verts[3 * i] + ct.x, verts[3 * i + 1] + ct.y, verts[3 * i + 2] + ct.z};
    PV& p = pv[i];
    p.ncam = vnrm[i];
    p.z = vc.z;
    p.ok = vc.z > opt.near_z;
    if (p.ok) {
      float inv = 1.0f / vc.z;
      p.sx = fx * vc.x * inv + cx;
      p.sy = fy * vc.y * inv + cy;
    } else {
      p.sx = p.sy = 0.0f;
    }
  }

  // Supersampled framebuffer: linear RGB + coverage flag + depth (as 1/z, larger
  // is nearer for a stable perspective-correct z test).
  const size_t npix = static_cast<size_t>(SW) * SH;
  std::vector<float> fb_r(npix, 0.0f), fb_g(npix, 0.0f), fb_b(npix, 0.0f);
  std::vector<uint8_t> fb_cov(npix, 0);
  std::vector<float> fb_invz(npix, -std::numeric_limits<float>::infinity());

  const float inv_fx = 1.0f / fx;
  const float inv_fy = 1.0f / fy;

  for (int f = 0; f < nf; ++f) {
    const int i0 = faces[3 * f], i1 = faces[3 * f + 1], i2 = faces[3 * f + 2];
    const PV &a = pv[i0], &b = pv[i1], &c = pv[i2];
    if (!a.ok || !b.ok || !c.ok) continue;  // simple near-plane reject

    // Edge-function signed area (in supersampled pixels).
    const float area = (b.sx - a.sx) * (c.sy - a.sy) - (c.sx - a.sx) * (b.sy - a.sy);
    if (std::fabs(area) < 1e-9f) continue;
    const float inv_area = 1.0f / area;

    // Bounding box, clamped to the framebuffer.
    int minx = static_cast<int>(std::floor(std::min({a.sx, b.sx, c.sx})));
    int maxx = static_cast<int>(std::ceil(std::max({a.sx, b.sx, c.sx})));
    int miny = static_cast<int>(std::floor(std::min({a.sy, b.sy, c.sy})));
    int maxy = static_cast<int>(std::ceil(std::max({a.sy, b.sy, c.sy})));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, SW - 1);
    maxy = std::min(maxy, SH - 1);
    if (minx > maxx || miny > maxy) continue;

    const float ia = 1.0f / a.z, ib = 1.0f / b.z, ic = 1.0f / c.z;

    for (int py = miny; py <= maxy; ++py) {
      const float ypc = py + 0.5f;
      for (int px = minx; px <= maxx; ++px) {
        const float xpc = px + 0.5f;
        // Barycentric via edge functions (winding-agnostic: divide by signed area).
        float w0 = (b.sx - xpc) * (c.sy - ypc) - (c.sx - xpc) * (b.sy - ypc);
        float w1 = (c.sx - xpc) * (a.sy - ypc) - (a.sx - xpc) * (c.sy - ypc);
        float w2 = (a.sx - xpc) * (b.sy - ypc) - (b.sx - xpc) * (a.sy - ypc);
        const bool inside =
            (w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0);
        if (!inside) continue;
        w0 *= inv_area;
        w1 *= inv_area;
        w2 *= inv_area;

        // Perspective-correct depth: 1/z is linear in screen space.
        const float invz = w0 * ia + w1 * ib + w2 * ic;
        const size_t idx = static_cast<size_t>(py) * SW + px;
        if (invz <= fb_invz[idx]) continue;  // farther or equal -> keep existing
        const float z = 1.0f / invz;

        // Perspective-correct normal.
        Vec3 n{
            (w0 * a.ncam.x * ia + w1 * b.ncam.x * ib + w2 * c.ncam.x * ic) * z,
            (w0 * a.ncam.y * ia + w1 * b.ncam.y * ib + w2 * c.ncam.y * ic) * z,
            (w0 * a.ncam.z * ia + w1 * b.ncam.z * ib + w2 * c.ncam.z * ic) * z,
        };
        n = normalize(n);

        // Reconstruct the camera-space position of this pixel to orient the normal
        // toward the camera (double-sided: always shade the visible front face).
        Vec3 pcam{(xpc - cx) * inv_fx * z, (ypc - cy) * inv_fy * z, z};
        if (dot(n, pcam) > 0.0f) {  // normal points away from camera -> flip
          n.x = -n.x;
          n.y = -n.y;
          n.z = -n.z;
        }
        Vec3 view_to_cam = normalize(Vec3{-pcam.x, -pcam.y, -pcam.z});

        float shade = ShadeScalar(n, view_to_cam, opt);
        float g = opt.grey * shade;
        g = std::min(1.0f, std::max(0.0f, g));

        fb_invz[idx] = invz;
        fb_r[idx] = g;
        fb_g[idx] = g;
        fb_b[idx] = g;
        fb_cov[idx] = 1;
      }
    }
  }

  // Box-downsample the supersampled buffer to the output resolution.
  const float inv_samples = 1.0f / static_cast<float>(ss * ss);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float sr = 0.0f, sg = 0.0f, sb = 0.0f;
      int cov = 0;
      for (int sy = 0; sy < ss; ++sy) {
        for (int sx = 0; sx < ss; ++sx) {
          const size_t si =
              static_cast<size_t>(y * ss + sy) * SW + (x * ss + sx);
          if (fb_cov[si]) {
            sr += fb_r[si];
            sg += fb_g[si];
            sb += fb_b[si];
            ++cov;
          }
        }
      }
      const float alpha = cov * inv_samples;
      float r, g, b;
      if (opt.premultiply) {
        // Premultiplied: colour already weighted by coverage.
        r = sr * inv_samples;
        g = sg * inv_samples;
        b = sb * inv_samples;
      } else if (cov > 0) {
        // Straight: average over covered subsamples only.
        const float invc = 1.0f / static_cast<float>(cov);
        r = sr * invc;
        g = sg * invc;
        b = sb * invc;
      } else {
        r = g = b = 0.0f;
      }
      const size_t o = (static_cast<size_t>(y) * W + x) * 4;
      out.data[o + 0] = r;
      out.data[o + 1] = g;
      out.data[o + 2] = b;
      out.data[o + 3] = alpha;
    }
  }
  return out;
}

}  // namespace hastur
