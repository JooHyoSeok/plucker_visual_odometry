// Geometry core (port of vo_plucker.py sections 1-6).
//   T_cw = (R, t):  X_c = R X_w + t
//   Plücker L = (d, m), m = p x d, |d| = 1
//   camera frame:   d_c = R d,  m_c = R m + [t]x R d
//   image line:     l ∝ KL m_c,  KL = [[fy,0,0],[0,fx,0],[-fy cx,-fx cy, fx fy]]  (∝ K^-T)
#pragma once
#include <Eigen/Dense>
#include <optional>
#include <vector>

namespace vp {

using Mat3 = Eigen::Matrix3d;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using VecX = Eigen::VectorXd;
using Line = Eigen::Matrix<double, 6, 1>;

struct Camera {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  int W = 0, H = 0;
  Mat3 K, Kinv, KL;
  Camera() = default;
  Camera(double fx, double fy, double cx, double cy, int W, int H);
};

struct Pose {
  Mat3 R = Mat3::Identity();
  Vec3 t = Vec3::Zero();
};

Mat3 skew(const Vec3& v);
Mat3 so3_exp(const Vec3& w);
inline Vec3 center(const Pose& p) { return -p.R.transpose() * p.t; }
Mat3 orthonormalize(const Mat3& R);
Pose retract(const Pose& p, const Vec6& dx);  // left perturbation: T <- Exp(dx) T

Vec3 project_line(const Camera& c, const Line& L, const Pose& p);  // pixel image line

// triangulation (N views)
std::optional<Vec3> tri_point_multi(const Camera& c, const std::vector<Pose>& P, const std::vector<Vec2>& uv);
Vec4 backproject_plane(const Camera& c, const Pose& p, const Vec4& seg);
std::optional<Line> tri_line_multi(const Camera& c, const std::vector<Pose>& P, const std::vector<Vec4>& segs);
double parallax_deg(const Vec3& X, const Pose& a, const Pose& b);
double plane_angle_deg(const Camera& c, const Pose& a, const Vec4& sa, const Pose& b, const Vec4& sb);
bool point_ok(const Camera& c, const std::optional<Vec3>& X, const std::vector<Pose>& P,
              const std::vector<Vec2>& uv, double min_deg);
bool line_ok(const Camera& c, const std::optional<Line>& L, const std::vector<Pose>& P,
             const std::vector<Vec4>& segs, double line_min_deg);
std::vector<int> pick_frames(const std::vector<int>& frames, int k_max = 12);

// 2D-2D (normalized coords x2^T E x1 = 0): Sampson error [≈px, scaled by fx], 5-DoF refinement of (R, t/|t|)
VecX sampson(const Mat3& R, const Vec3& t, const std::vector<Vec2>& x1, const std::vector<Vec2>& x2, double fx);
Pose refine_relative_pose(Pose p, const std::vector<Vec2>& x1, const std::vector<Vec2>& x2, double fx,
                          int iters = 30);

// pose estimation against the map (points + lines)
struct MapObs {
  std::vector<Vec3> X;
  std::vector<Vec2> uv;
  std::vector<Line> L;
  std::vector<Vec4> seg;
};
VecX residuals(const Camera& c, const Pose& p, const MapObs& m);  // [point px (2 each), line endpoint dist px (2 each)]
double huber_cost(const VecX& r, double delta);
Pose estimate_pose(const Camera& c, Pose p, const MapObs& m, int iters = 40, double huber = 3.0);
// rotation + translation direction fixed, solve |t| only: t_k = base + s * that
double estimate_scale(const Camera& c, const Mat3& Rk, const Vec3& base, const Vec3& that, double s0,
                      const MapObs& m, int iters = 30, double huber = 3.0);

}  // namespace vp
