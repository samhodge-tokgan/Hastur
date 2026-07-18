// Copyright the Hastur authors.
// SPDX-License-Identifier: Apache-2.0

#include "MhrModel.h"

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <vector>

namespace hastur {
namespace {

constexpr double kLn2 = 0.6931471805599453;
constexpr double kPi = 3.14159265358979323846;

using Vec3d = Eigen::Vector3d;

// ---- quaternion (xyzw) helpers, matching pymomentum / roma backends ----------
struct Quat {  // x, y, z, w
  double x, y, z, w;
};

inline Quat quatNormalize(const Quat& q) {
  double n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (n < 1e-12) n = 1e-12;
  return {q.x / n, q.y / n, q.z / n, q.w / n};
}

// Hamilton product q1 (x) q2, assumes normalized inputs.
inline Quat quatMul(const Quat& a, const Quat& b) {
  return {
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

// Rotate vector v by (assumed-normalized) quaternion q.
inline Vec3d quatRotate(const Quat& q, const Vec3d& v) {
  Vec3d axis(q.x, q.y, q.z);
  double r = q.w;
  Vec3d av = axis.cross(v);
  Vec3d aav = axis.cross(av);
  return v + 2.0 * (av * r + aav);
}

// euler XYZ (roll,pitch,yaw) -> quaternion (xyzw), matching quaternion.py.
inline Quat eulerXyzToQuat(double roll, double pitch, double yaw) {
  double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
  double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
  double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
  return {sr * cp * cy - cr * sp * sy,
          cr * sp * cy + sr * cp * sy,
          cr * cp * sy - sr * sp * cy,
          cr * cp * cy + sr * sp * sy};
}

// skel_state = [tx,ty,tz, qx,qy,qz,qw, s]. multiply(s1, s2) (parent . child).
struct Skel {
  Vec3d t;
  Quat q;
  double s;
};

inline Skel skelMultiply(const Skel& a, const Skel& b) {
  Quat qa = quatNormalize(a.q);
  Quat qb = quatNormalize(b.q);
  Skel r;
  r.t = a.t + a.s * quatRotate(qa, b.t);
  r.q = quatMul(qa, qb);
  r.s = a.s * b.s;
  return r;
}

// ---- roma: rotmat -> unitquat (SciPy decision-matrix method) -----------------
inline Quat rotmatToUnitQuat(const Eigen::Matrix3d& M) {
  double d0 = M(0, 0), d1 = M(1, 1), d2 = M(2, 2);
  double d3 = d0 + d1 + d2;
  double dm[4] = {d0, d1, d2, d3};
  int choice = 0;
  for (int c = 1; c < 4; ++c)
    if (dm[c] > dm[choice]) choice = c;

  double q[4];  // xyzw
  if (choice != 3) {
    int i = choice, j = (i + 1) % 3, k = (j + 1) % 3;
    q[i] = 1.0 - d3 + 2.0 * M(i, i);
    q[j] = M(j, i) + M(i, j);
    q[k] = M(k, i) + M(i, k);
    q[3] = M(k, j) - M(j, k);
  } else {
    q[0] = M(2, 1) - M(1, 2);
    q[1] = M(0, 2) - M(2, 0);
    q[2] = M(1, 0) - M(0, 1);
    q[3] = 1.0 + d3;
  }
  double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

inline double wrapPi(double a) {
  if (a < -kPi) a += 2 * kPi;
  if (a > kPi) a -= 2 * kPi;
  return a;
}

// roma: unitquat (xyzw) -> euler, intrinsic "ZYX". Returns [a0,a1,a2] in the
// order roma stacks them (Bernardes-Viollet / scipy algorithm, ZYX specialized).
inline std::array<double, 3> quatToEulerZYX(const Quat& quat, double eps = 1e-7) {
  // working convention "xyz", extrinsic=false; i=0,j=1,k=2; asymmetric; sign=+1.
  double qx = quat.x, qy = quat.y, qz = quat.z, qw = quat.w;
  double a = qw - qy;
  double b = qx + qz;
  double c = qy + qw;
  double d = qz - qx;

  std::array<double, 3> ang{0, 0, 0};
  ang[1] = 2.0 * std::atan2(std::hypot(c, d), std::hypot(a, b));
  double half_sum = std::atan2(b, a);
  double half_diff = std::atan2(d, c);
  // intrinsic: angle_first=2, angle_third=0
  ang[2] = half_sum - half_diff;
  ang[0] = half_sum + half_diff;

  bool case1 = std::abs(ang[1]) <= eps;
  bool case2 = std::abs(ang[1] - kPi) <= eps;
  if (case1 || case2) {
    ang[2] = 0.0;
    if (case1) ang[0] = 2.0 * half_sum;
    if (case2) ang[0] = 2.0 * half_diff;  // intrinsic sign=+1
  }
  // asymmetric: angles[angle_third=0] *= sign(=1); angles[1] -= pi/2
  ang[1] -= kPi / 2.0;
  for (int i = 0; i < 3; ++i) ang[i] = wrapPi(ang[i]);
  return ang;
}

// ---- mhr_utils: batchXYZfrom6D for a single 6-vector -> euler(3) -------------
inline std::array<double, 3> xyzFrom6D(const double p[6]) {
  Vec3d x_raw(p[0], p[1], p[2]);
  Vec3d y_raw(p[3], p[4], p[5]);
  Vec3d x = x_raw.normalized();
  Vec3d z = x.cross(y_raw).normalized();
  Vec3d y = z.cross(x);
  // matrix columns = [x, y, z]
  double m00 = x[0], m10 = x[1], m20 = x[2];
  double m11 = y[1], m21 = y[2];
  double m12 = z[1], m22 = z[2];
  double sy = std::sqrt(m00 * m00 + m10 * m10);
  std::array<double, 3> e;
  if (sy >= 1e-6) {
    e[0] = std::atan2(m21, m22);
    e[1] = std::atan2(-m20, sy);
    e[2] = std::atan2(m10, m00);
  } else {
    e[0] = std::atan2(-m12, m11);
    e[1] = std::atan2(-m20, sy);
    e[2] = 0.0;
  }
  return e;
}

// ---- rot6d_to_rotmat (Gram-Schmidt), columns b1,b2,b3 ------------------------
inline Eigen::Matrix3d rot6dToRotmat(const float g[6]) {
  Vec3d a1(g[0], g[1], g[2]);
  Vec3d a2(g[3], g[4], g[5]);
  Vec3d b1 = a1.normalized();
  Vec3d b2 = (a2 - b1.dot(a2) * b1).normalized();
  Vec3d b3 = b1.cross(b2);
  Eigen::Matrix3d R;
  R.col(0) = b1;
  R.col(1) = b2;
  R.col(2) = b3;
  return R;
}

// ---- body cont -> model params (133) index tables (mhr_utils.py) -------------
const int kBody3Dof[23][3] = {
    {0, 2, 4}, {6, 8, 10}, {12, 13, 14}, {15, 16, 17}, {18, 19, 20},
    {21, 22, 23}, {24, 25, 26}, {27, 28, 29}, {34, 35, 36}, {37, 38, 39},
    {44, 45, 46}, {53, 54, 55}, {64, 65, 66}, {85, 69, 73}, {86, 70, 79},
    {87, 71, 82}, {88, 72, 76}, {91, 92, 93}, {112, 96, 100}, {113, 97, 106},
    {114, 98, 109}, {115, 99, 103}, {130, 131, 132}};
const int kBody1Dof[58] = {
    1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52,
    56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89,
    90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120,
    121, 122, 123};
const int kBody1DofTrans[6] = {124, 125, 126, 127, 128, 129};
// mhr_param_hand_idxs (zeroed in body pose euler)
const int kHandParamIdx[54] = {
    62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97,
    98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
    113, 114, 115};

// hand dofs (compact_cont_to_model_params_hand)
const int kHandDofs[16] = {3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1};

void compactContToBody(const float cont[260], double body[133]) {
  for (int i = 0; i < 133; ++i) body[i] = 0.0;
  const int n3 = 23 * 3;   // 69
  const int n1 = 58;
  // 3dof: cont[0 : 138] as 23 x 6
  for (int t = 0; t < 23; ++t) {
    double p[6];
    for (int c = 0; c < 6; ++c) p[c] = cont[t * 6 + c];
    std::array<double, 3> e = xyzFrom6D(p);
    for (int c = 0; c < 3; ++c) body[kBody3Dof[t][c]] = e[c];
  }
  // 1dof: cont[138 : 138+116] as 58 x 2 (sin,cos) -> atan2(sin,cos)
  const int off1 = 2 * n3;  // 138
  for (int t = 0; t < n1; ++t) {
    double s = cont[off1 + t * 2 + 0];
    double co = cont[off1 + t * 2 + 1];
    body[kBody1Dof[t]] = std::atan2(s, co);
  }
  // trans: cont[254:260]
  const int offT = off1 + 2 * n1;  // 254
  for (int t = 0; t < 6; ++t) body[kBody1DofTrans[t]] = cont[offT + t];
}

void compactContToHand(const double hand_cont[54], double hand_mp[27]) {
  for (int i = 0; i < 27; ++i) hand_mp[i] = 0.0;
  int cpos = 0;   // position in cont (54)
  int mpos = 0;   // position in model params (27)
  for (int jd = 0; jd < 16; ++jd) {
    int k = kHandDofs[jd];
    if (k == 3) {
      double p[6];
      for (int c = 0; c < 6; ++c) p[c] = hand_cont[cpos + c];
      std::array<double, 3> e = xyzFrom6D(p);
      for (int c = 0; c < 3; ++c) hand_mp[mpos + c] = e[c];
      cpos += 6;
      mpos += 3;
    } else {  // k == 1 or 2: k sincos pairs -> k angles
      for (int a = 0; a < k; ++a) {
        double s = hand_cont[cpos + a * 2 + 0];
        double co = hand_cont[cpos + a * 2 + 1];
        hand_mp[mpos + a] = std::atan2(s, co);
      }
      cpos += 2 * k;
      mpos += k;
    }
  }
}

}  // namespace

// -----------------------------------------------------------------------------
MhrModel::MhrModel(std::shared_ptr<const MeshAssets> assets) : a_(std::move(assets)) {
  V_ = static_cast<int>(a_->block("base_shape").shape[0]);
  J_ = static_cast<int>(a_->block("joint_translation_offsets").shape[0]);
}

void MhrModel::DecodeInternal(const std::array<float, kParamDim>& pred,
                              std::array<float, 204>& model_params,
                              std::array<float, kShapeComps>& shape,
                              std::array<float, 889>& joint_params) const {
  int c = 0;
  // --- global 6D -> ZYX euler ---
  float g6[6];
  for (int i = 0; i < 6; ++i) g6[i] = pred[c + i];
  c += kGlobal6D;
  Eigen::Matrix3d R = rot6dToRotmat(g6);
  Quat gq = rotmatToUnitQuat(R);
  std::array<double, 3> grot = quatToEulerZYX(gq);

  // --- body continuous pose -> euler(133) ---
  float cont[260];
  for (int i = 0; i < 260; ++i) cont[i] = pred[c + i];
  c += kBodyCont;
  double body133[133];
  compactContToBody(cont, body133);
  // zero out hand euler params
  for (int i = 0; i < 54; ++i) body133[kHandParamIdx[i]] = 0.0;
  // zero jaw (last 3)
  body133[130] = body133[131] = body133[132] = 0.0;

  // --- shape / scale / hand PCA slices ---
  for (int i = 0; i < kShapeComps; ++i) shape[i] = pred[c + i];
  c += kShapeComps;
  float scale28[kScaleComps];
  for (int i = 0; i < kScaleComps; ++i) scale28[i] = pred[c + i];
  c += kScaleComps;
  float hand_pca[108];
  for (int i = 0; i < 108; ++i) hand_pca[i] = pred[c + i];
  c += 2 * kHandPCA;
  // expr slice zeroed / ignored.

  // --- scales = scale_mean + scale28 @ scale_comps(28,68) ---
  const float* scale_mean = a_->f32("scale_mean");
  const float* scale_comps = a_->f32("scale_comps");  // (28,68)
  double scales[68];
  for (int o = 0; o < 68; ++o) {
    double v = scale_mean[o];
    for (int k = 0; k < 28; ++k) v += static_cast<double>(scale28[k]) * scale_comps[k * 68 + o];
    scales[o] = v;
  }

  // --- full_pose(136) = [0,0,0, grot(3), body133[:130]] ---
  double full_pose[136];
  full_pose[0] = full_pose[1] = full_pose[2] = 0.0;
  full_pose[3] = grot[0];
  full_pose[4] = grot[1];
  full_pose[5] = grot[2];
  for (int i = 0; i < 130; ++i) full_pose[6 + i] = body133[i];

  // --- hand PCA -> model params, dropped into full_pose ---
  const float* hpm = a_->f32("hand_pose_mean");     // (54,)
  const float* hpc = a_->f32("hand_pose_comps");    // (54,54)
  const int32_t* hjl = a_->i32("hand_joint_idxs_left");
  const int32_t* hjr = a_->i32("hand_joint_idxs_right");
  for (int side = 0; side < 2; ++side) {
    const float* in = hand_pca + side * kHandPCA;   // 54 coeffs
    // cont = hand_pose_mean + in @ hand_pose_comps(54,54)
    double cont54[54];
    for (int o = 0; o < 54; ++o) {
      double v = hpm[o];
      for (int k = 0; k < 54; ++k) v += static_cast<double>(in[k]) * hpc[k * 54 + o];
      cont54[o] = v;
    }
    double hand_mp[27];
    compactContToHand(cont54, hand_mp);
    const int32_t* idx = side == 0 ? hjl : hjr;
    for (int i = 0; i < 27; ++i) full_pose[idx[i]] = hand_mp[i];
  }

  // --- model_params(204) = [full_pose(136), scales(68)] ---
  for (int i = 0; i < 136; ++i) model_params[i] = static_cast<float>(full_pose[i]);
  for (int i = 0; i < 68; ++i) model_params[136 + i] = static_cast<float>(scales[i]);

  // --- joint_parameters(889) = parameter_transform(889,249) @ [model_params, 0_45] ---
  const float* PT = a_->f32("parameter_transform");  // (889,249)
  for (int d = 0; d < 889; ++d) {
    double v = 0.0;
    const float* row = PT + d * 249;
    for (int n = 0; n < 204; ++n) v += static_cast<double>(row[n]) * model_params[n];
    // columns 204..248 multiply the zero identity coeffs -> skip
    joint_params[d] = static_cast<float>(v);
  }
}

void MhrModel::Decode(const std::array<float, kParamDim>& pred,
                      std::array<float, 204>& model_params,
                      std::array<float, kShapeComps>& shape) const {
  std::array<float, 889> jp;
  DecodeInternal(pred, model_params, shape, jp);
}

std::array<float, 889> MhrModel::JointParameters(
    const std::array<float, kParamDim>& pred) const {
  std::array<float, 204> mp;
  std::array<float, kShapeComps> sh;
  std::array<float, 889> jp;
  DecodeInternal(pred, mp, sh, jp);
  return jp;
}

Mesh MhrModel::Run(const std::array<float, kParamDim>& pred,
                   const float* pose_corrective) const {
  std::array<float, 204> model_params;
  std::array<float, kShapeComps> shape;
  std::array<float, 889> jp;
  DecodeInternal(pred, model_params, shape, jp);

  const int V = V_, J = J_;

  // --- identity rest pose = base_shape + sum_n shape_vectors[n] * shape[n] ---
  const float* base = a_->f32("base_shape");           // (V,3)
  const float* sv = a_->f32("shape_vectors");          // (45,V,3)
  std::vector<double> unposed(static_cast<size_t>(V) * 3);
  for (int v = 0; v < V; ++v) {
    unposed[v * 3 + 0] = base[v * 3 + 0];
    unposed[v * 3 + 1] = base[v * 3 + 1];
    unposed[v * 3 + 2] = base[v * 3 + 2];
  }
  for (int n = 0; n < kShapeComps; ++n) {
    double cn = shape[n];
    if (cn == 0.0) continue;
    const float* sb = sv + static_cast<size_t>(n) * V * 3;
    for (int v = 0; v < V * 3; ++v) unposed[v] += cn * sb[v];
  }
  // --- + pose-corrective offsets (cm, from the ONNX hook) ---
  if (pose_corrective) {
    for (int v = 0; v < V * 3; ++v) unposed[v] += pose_corrective[v];
  }

  // --- per-joint local skeleton state ---
  const float* jto = a_->f32("joint_translation_offsets");  // (J,3)
  const float* jpr = a_->f32("joint_prerotations");         // (J,4) xyzw
  std::vector<Skel> glob(J);
  for (int j = 0; j < J; ++j) {
    const float* jpar = &jp[j * 7];
    Skel s;
    s.t = Vec3d(jpar[0] + jto[j * 3 + 0], jpar[1] + jto[j * 3 + 1], jpar[2] + jto[j * 3 + 2]);
    Quat qe = eulerXyzToQuat(jpar[3], jpar[4], jpar[5]);
    Quat pre{jpr[j * 4 + 0], jpr[j * 4 + 1], jpr[j * 4 + 2], jpr[j * 4 + 3]};
    s.q = quatMul(pre, qe);  // prerotation . euler
    s.s = std::exp(static_cast<double>(jpar[6]) * kLn2);
    glob[j] = s;
  }

  // --- forward kinematics: prefix-multiply over pmi levels (double precision) ---
  const int64_t* pmi = a_->i64("pmi");                 // (2,266) row0=source,row1=target
  const int32_t* pmi_sizes = a_->i32("pmi_buffer_sizes");
  const int nlev = static_cast<int>(a_->block("pmi_buffer_sizes").shape[0]);
  const int64_t* src_row = pmi;             // 266
  const int64_t* tgt_row = pmi + 266;       // 266
  int col = 0;
  for (int lvl = 0; lvl < nlev; ++lvl) {
    int k = pmi_sizes[lvl];
    // read-all-then-write (torch gathers state1/state2 before index_copy_)
    std::vector<Skel> updated(k);
    for (int e = 0; e < k; ++e) {
      int64_t source = src_row[col + e];
      int64_t target = tgt_row[col + e];
      updated[e] = skelMultiply(glob[target], glob[source]);
    }
    for (int e = 0; e < k; ++e) glob[src_row[col + e]] = updated[e];
    col += k;
  }

  // --- linear blend skinning ---
  const float* ibp = a_->f32("inverse_bind_pose");     // (J,8)
  std::vector<Skel> jstate(J);
  for (int j = 0; j < J; ++j) {
    Skel inv{Vec3d(ibp[j * 8 + 0], ibp[j * 8 + 1], ibp[j * 8 + 2]),
             {ibp[j * 8 + 3], ibp[j * 8 + 4], ibp[j * 8 + 5], ibp[j * 8 + 6]},
             static_cast<double>(ibp[j * 8 + 7])};
    jstate[j] = skelMultiply(glob[j], inv);
  }

  const int64_t nnz = a_->block("skin_weights").shape[0];
  const int64_t* vidx = a_->i64("skin_vert_indices");
  const int32_t* jidx = a_->i32("skin_joint_indices");
  const float* sw = a_->f32("skin_weights");

  Mesh mesh;
  mesh.verts.assign(static_cast<size_t>(V) * 3, 0.0f);
  std::vector<double> skinned(static_cast<size_t>(V) * 3, 0.0);
  for (int64_t e = 0; e < nnz; ++e) {
    int v = static_cast<int>(vidx[e]);
    const Skel& js = jstate[jidx[e]];
    Quat q = quatNormalize(js.q);
    Vec3d p(unposed[v * 3 + 0], unposed[v * 3 + 1], unposed[v * 3 + 2]);
    Vec3d tp = js.t + quatRotate(q, js.s * p);
    double w = sw[e];
    skinned[v * 3 + 0] += w * tp[0];
    skinned[v * 3 + 1] += w * tp[1];
    skinned[v * 3 + 2] += w * tp[2];
  }

  // --- units (/100) + keypoint regression + camera-frame [1,2] flip ---
  // verts and joints in metres (before flip): value / 100.
  std::vector<double> vj(static_cast<size_t>(V + J) * 3);  // [verts; joints], metres
  for (int v = 0; v < V; ++v) {
    vj[v * 3 + 0] = skinned[v * 3 + 0] / 100.0;
    vj[v * 3 + 1] = skinned[v * 3 + 1] / 100.0;
    vj[v * 3 + 2] = skinned[v * 3 + 2] / 100.0;
  }
  mesh.joints.assign(static_cast<size_t>(J) * 3, 0.0f);
  for (int j = 0; j < J; ++j) {
    double x = glob[j].t[0] / 100.0, y = glob[j].t[1] / 100.0, z = glob[j].t[2] / 100.0;
    vj[(V + j) * 3 + 0] = x;
    vj[(V + j) * 3 + 1] = y;
    vj[(V + j) * 3 + 2] = z;
  }

  // keypoints = keypoint_mapping(308,18566) @ vj, take first 70.
  const float* KM = a_->f32("keypoint_mapping");  // (308, V+J)
  const int N = V + J;
  mesh.keypoints.assign(static_cast<size_t>(kNumKeypoints) * 3, 0.0f);
  for (int kp = 0; kp < kNumKeypoints; ++kp) {
    const float* row = KM + static_cast<size_t>(kp) * N;
    double ox = 0, oy = 0, oz = 0;
    for (int n = 0; n < N; ++n) {
      double w = row[n];
      if (w == 0.0) continue;
      ox += w * vj[n * 3 + 0];
      oy += w * vj[n * 3 + 1];
      oz += w * vj[n * 3 + 2];
    }
    // flip [1,2]
    mesh.keypoints[kp * 3 + 0] = static_cast<float>(ox);
    mesh.keypoints[kp * 3 + 1] = static_cast<float>(-oy);
    mesh.keypoints[kp * 3 + 2] = static_cast<float>(-oz);
  }

  // final verts / joints with [1,2] flip.
  for (int v = 0; v < V; ++v) {
    mesh.verts[v * 3 + 0] = static_cast<float>(vj[v * 3 + 0]);
    mesh.verts[v * 3 + 1] = static_cast<float>(-vj[v * 3 + 1]);
    mesh.verts[v * 3 + 2] = static_cast<float>(-vj[v * 3 + 2]);
  }
  for (int j = 0; j < J; ++j) {
    mesh.joints[j * 3 + 0] = static_cast<float>(vj[(V + j) * 3 + 0]);
    mesh.joints[j * 3 + 1] = static_cast<float>(-vj[(V + j) * 3 + 1]);
    mesh.joints[j * 3 + 2] = static_cast<float>(-vj[(V + j) * 3 + 2]);
  }

  mesh.faces = a_->faces();
  return mesh;
}

}  // namespace hastur
