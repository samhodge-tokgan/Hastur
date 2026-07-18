// Copyright the Hastur authors.
// SPDX-License-Identifier: LicenseRef-SAM-License
#include "CameraSolver.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace hastur {

namespace {

// fix_aspect_ratio(scale, aspect_ratio=w/h): if w > h*r -> (w, w/r) else (h*r, h).
std::array<float, 2> FixAspectRatio(std::array<float, 2> s, float r) {
  const float w = s[0], h = s[1];
  if (w > h * r) return {w, w / r};
  return {h * r, h};
}

// PyTorch antialiased separable resample weights for one output dimension.
// Mirrors aten _compute_weights_aa (bilinear/linear filter), align_corners=false.
// For output index i: sample window [xmin, xmin+xsize) of the input with the
// normalized triangle weights. scale = input/output.
struct AAWeights {
  int out_size = 0;
  std::vector<int> xmin;    // window start per output index
  std::vector<int> xsize;   // window length per output index
  std::vector<double> w;    // flattened weights, out_size * max_xsize
  int stride = 0;           // max_xsize
};

double LinearFilter(double x) {
  if (x < 0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

AAWeights ComputeAAWeights(int in_size, int out_size) {
  AAWeights r;
  r.out_size = out_size;
  const double scale = static_cast<double>(in_size) / out_size;  // >=1 (downsample)
  const double support = (scale >= 1.0) ? 1.0 * scale : 1.0;     // interp_size/2 = 1
  const double invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  r.stride = static_cast<int>(std::ceil(2.0 * support)) + 2;
  r.xmin.resize(out_size);
  r.xsize.resize(out_size);
  r.w.assign(static_cast<size_t>(out_size) * r.stride, 0.0);
  for (int i = 0; i < out_size; ++i) {
    const double center = scale * (i + 0.5);
    int xmin = std::max(static_cast<int>(center - support + 0.5), 0);
    int xmax = std::min(static_cast<int>(center + support + 0.5), in_size);
    int xsize = xmax - xmin;
    double total = 0.0;
    for (int j = 0; j < xsize; ++j) {
      double ww = LinearFilter((j + xmin - center + 0.5) * invscale);
      r.w[static_cast<size_t>(i) * r.stride + j] = ww;
      total += ww;
    }
    if (total != 0.0)
      for (int j = 0; j < xsize; ++j) r.w[static_cast<size_t>(i) * r.stride + j] /= total;
    r.xmin[i] = xmin;
    r.xsize[i] = xsize;
  }
  return r;
}

}  // namespace

BBoxCS ProcessBBox(const BBox& box, float pad, int in_w, int in_h) {
  BBoxCS out;
  out.center = {(box.x0 + box.x1) * 0.5f, (box.y0 + box.y1) * 0.5f};
  out.scale = {(box.x1 - box.x0) * pad, (box.y1 - box.y0) * pad};
  // TopdownAffine: prior aspect_ratio 0.75, then to the model input ratio w/h.
  std::array<float, 2> s = FixAspectRatio(out.scale, 0.75f);
  s = FixAspectRatio(s, static_cast<float>(in_w) / in_h);
  out.square = s[0];  // in_w==in_h -> square; s[0]==s[1]
  return out;
}

CamInt DefaultCamInt(int W, int H, float fov_override_deg, float focal_override,
                     float px, float py) {
  const double diag = std::sqrt(static_cast<double>(W) * W + static_cast<double>(H) * H);
  double f;
  if (focal_override > 0.f)
    f = focal_override;
  else if (fov_override_deg > 0.f)
    f = diag / (2.0 * std::tan(fov_override_deg * M_PI / 180.0 / 2.0));
  else
    f = diag;  // prepare_batch.py default
  const double cx = (px >= 0.f) ? px : W / 2.0;
  const double cy = (py >= 0.f) ? py : H / 2.0;
  CamInt K{};
  K[0] = static_cast<float>(f);
  K[4] = static_cast<float>(f);
  K[2] = static_cast<float>(cx);
  K[5] = static_cast<float>(cy);
  K[8] = 1.f;
  return K;
}

void MakeAffine(const std::array<float, 2>& center, float square, int out,
                std::array<float, 6>& fwd, std::array<float, 6>& inv) {
  // get_warp_matrix(rot=0): maps the `square`-sided box centered at `center`
  // onto [0,out]. Pure isotropic scale a = out/square + translation.
  const double a = static_cast<double>(out) / square;
  const double tx = out * 0.5 - center[0] * a;
  const double ty = out * 0.5 - center[1] * a;
  fwd = {static_cast<float>(a), 0.f, static_cast<float>(tx),
         0.f, static_cast<float>(a), static_cast<float>(ty)};
  // Inverse (crop -> frame): a_inv = square/out.
  const double ai = 1.0 / a;
  inv = {static_cast<float>(ai), 0.f, static_cast<float>(-tx * ai),
         0.f, static_cast<float>(ai), static_cast<float>(-ty * ai)};
}

std::array<float, 3> ConditionInfo(const BBox& box, float pad, const CamInt& K,
                                   int in_w, int in_h) {
  BBoxCS cs = ProcessBBox(box, pad, in_w, in_h);
  const double f = K[0];
  const double cx = cs.center[0], cy = cs.center[1];
  const double kcx = K[2], kcy = K[5];
  return {static_cast<float>((cx - kcx) / f), static_cast<float>((cy - kcy) / f),
          static_cast<float>(cs.square / f)};
}

void RayCond(const BBox& box, float pad, const CamInt& K,
             std::array<float, 2 * kTokenGrid * kTokenGrid>& out, int in_w, int in_h) {
  BBoxCS cs = ProcessBBox(box, pad, in_w, in_h);
  std::array<float, 6> fwd, inv;
  MakeAffine(cs.center, cs.square, in_w, fwd, inv);
  // Diagonal scale + translation of the forward (frame->crop) affine.
  const double ax = fwd[0], tx = fwd[2];  // channel 0 (x)
  const double ay = fwd[4], ty = fwd[5];  // channel 1 (y)
  const double kcx = K[2], kfx = K[0];
  const double kcy = K[5], kfy = K[4];

  // Build the [2, H, W] linear ray field at full crop resolution, then apply
  // the exact PyTorch antialiased bilinear downsample by 1/patch (H,W -> 32,32).
  const int H = in_h, W = in_w;
  std::vector<double> chan0(static_cast<size_t>(H) * W);
  std::vector<double> chan1(static_cast<size_t>(H) * W);
  // ray_x(col) = ((col/ax - tx/ax) - kcx)/kfx ; ray_y(row) similarly.
  for (int c = 0; c < W; ++c) {
    double full_x = c / ax - tx / ax;
    double rx = (full_x - kcx) / kfx;
    for (int r = 0; r < H; ++r) chan0[static_cast<size_t>(r) * W + c] = rx;
  }
  for (int r = 0; r < H; ++r) {
    double full_y = r / ay - ty / ay;
    double ry = (full_y - kcy) / kfy;
    double* row = &chan1[static_cast<size_t>(r) * W];
    for (int c = 0; c < W; ++c) row[c] = ry;
  }

  const int G = kTokenGrid;
  AAWeights wr = ComputeAAWeights(H, G);  // rows: H -> G
  AAWeights wc = ComputeAAWeights(W, G);  // cols: W -> G

  auto downsample = [&](const std::vector<double>& in, double* dst) {
    // Pass 1: resample columns W -> G, keep H rows. tmp is [H, G].
    std::vector<double> tmp(static_cast<size_t>(H) * G);
    for (int r = 0; r < H; ++r) {
      const double* srow = &in[static_cast<size_t>(r) * W];
      for (int oc = 0; oc < G; ++oc) {
        const double* w = &wc.w[static_cast<size_t>(oc) * wc.stride];
        int xmin = wc.xmin[oc], xs = wc.xsize[oc];
        double acc = 0.0;
        for (int j = 0; j < xs; ++j) acc += w[j] * srow[xmin + j];
        tmp[static_cast<size_t>(r) * G + oc] = acc;
      }
    }
    // Pass 2: resample rows H -> G. out is [G, G].
    for (int orr = 0; orr < G; ++orr) {
      const double* w = &wr.w[static_cast<size_t>(orr) * wr.stride];
      int ymin = wr.xmin[orr], ys = wr.xsize[orr];
      for (int oc = 0; oc < G; ++oc) {
        double acc = 0.0;
        for (int j = 0; j < ys; ++j)
          acc += w[j] * tmp[static_cast<size_t>(ymin + j) * G + oc];
        dst[static_cast<size_t>(orr) * G + oc] = acc;
      }
    }
  };

  // Write channel 0 then channel 1 into out (float).
  std::vector<double> d0(static_cast<size_t>(G) * G), d1(static_cast<size_t>(G) * G);
  downsample(chan0, d0.data());
  downsample(chan1, d1.data());
  for (int i = 0; i < G * G; ++i) {
    out[i] = static_cast<float>(d0[i]);
    out[G * G + i] = static_cast<float>(d1[i]);
  }
}

Camera PerspectiveProjection(const std::array<float, 3>& pred_cam, const BBox& box,
                             float pad, const CamInt& K, float default_scale,
                             int in_w, int in_h) {
  BBoxCS cs = ProcessBBox(box, pad, in_w, in_h);
  // pred_cam[[0,2]] *= -1  (camera-system difference).
  const double s = -static_cast<double>(pred_cam[0]);
  const double tx = pred_cam[1];
  const double ty = -static_cast<double>(pred_cam[2]);
  const double bbox_size = cs.square;
  const double focal = K[0];
  const double bs = bbox_size * s * default_scale + 1e-8;
  const double tz = 2.0 * focal / bs;
  // USE_INTRIN_CENTER=true
  const double cx = 2.0 * (cs.center[0] - K[2]) / bs;
  const double cy = 2.0 * (cs.center[1] - K[5]) / bs;

  Camera cam;
  cam.focal = static_cast<float>(focal);
  cam.center = {K[2], K[5]};
  cam.cam_t = {static_cast<float>(tx + cx), static_cast<float>(ty + cy),
               static_cast<float>(tz)};
  return cam;
}

}  // namespace hastur
