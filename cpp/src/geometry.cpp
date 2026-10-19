#include "vo_plucker/geometry.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vp {

namespace {
constexpr double kRad2Deg = 180.0 / M_PI;

Eigen::Matrix<double, 3, 4> proj_matrix(const Camera& c, const Pose& p) {
  Eigen::Matrix<double, 3, 4> P;
  P << c.K * p.R, c.K * p.t;
  return P;
}

VecX huber_weights(const VecX& r, double huber) {
  return r.cwiseAbs().unaryExpr([huber](double a) { return a <= huber ? 1.0 : huber / std::max(a, 1e-12); });
}
}  // namespace

Camera::Camera(double fx_, double fy_, double cx_, double cy_, int W_, int H_)
    : fx(fx_), fy(fy_), cx(cx_), cy(cy_), W(W_), H(H_) {
  K << fx, 0, cx, 0, fy, cy, 0, 0, 1;
  Kinv = K.inverse();
  KL << fy, 0, 0, 0, fx, 0, -fy * cx, -fx * cy, fx * fy;
}

Mat3 skew(const Vec3& v) {
  Mat3 S;
  S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
  return S;
}

Mat3 so3_exp(const Vec3& w) {
  const double th = w.norm();
  if (th < 1e-12) return Mat3::Identity() + skew(w);
  const Mat3 Kx = skew(w / th);
  return Mat3::Identity() + std::sin(th) * Kx + (1 - std::cos(th)) * Kx * Kx;
}

Mat3 orthonormalize(const Mat3& R) {
  // SO(3)로 다시 투영. constant-velocity 예측에서 수치 오차가 프레임마다 커지는 걸 막음
  Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Mat3 U = svd.matrixU(), V = svd.matrixV();
  Mat3 D = Mat3::Identity();
  D(2, 2) = (U * V.transpose()).determinant();
  return U * D * V.transpose();
}

Pose retract(const Pose& p, const Vec6& dx) {
  const Mat3 dR = so3_exp(dx.head<3>());
  return Pose{dR * p.R, dR * p.t + dx.tail<3>()};
}

Vec3 project_line(const Camera& c, const Line& L, const Pose& p) {
  const Vec3 d = p.R * L.head<3>();
  const Vec3 m = p.R * L.tail<3>() + p.t.cross(d);
  return c.KL * m;
}

std::optional<Vec3> tri_point_multi(const Camera& c, const std::vector<Pose>& P, const std::vector<Vec2>& uv) {
  Eigen::Matrix<double, Eigen::Dynamic, 4> A(2 * P.size(), 4);
  for (size_t i = 0; i < P.size(); ++i) {
    const auto Pm = proj_matrix(c, P[i]);
    A.row(2 * i) = uv[i](0) * Pm.row(2) - Pm.row(0);
    A.row(2 * i + 1) = uv[i](1) * Pm.row(2) - Pm.row(1);
  }
  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 4>> svd(A, Eigen::ComputeFullV);
  const Vec4 X = svd.matrixV().col(3);
  if (std::abs(X(3)) <= 1e-12) return std::nullopt;
  return Vec3(X.head<3>() / X(3));
}

Vec4 backproject_plane(const Camera& c, const Pose& p, const Vec4& seg) {
  // 이미지 선분 -> 카메라 중심과 그 선을 품는 평면 π = Pᵀl (|n| = 1로 정규화)
  const Vec3 l = Vec3(seg(0), seg(1), 1.0).cross(Vec3(seg(2), seg(3), 1.0));
  const Vec4 pi = proj_matrix(c, p).transpose() * l;
  return pi / pi.head<3>().norm();
}

std::optional<Line> tri_line_multi(const Camera& c, const std::vector<Pose>& P, const std::vector<Vec4>& segs) {
  // N개의 역투영 평면을 쌓은 Π (N×4)의 null space(2차원) = 직선 위 두 점 -> Plücker
  Eigen::Matrix<double, Eigen::Dynamic, 4> Pi(P.size(), 4);
  for (size_t i = 0; i < P.size(); ++i) Pi.row(i) = backproject_plane(c, P[i], segs[i]).transpose();
  Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 4>> svd(Pi, Eigen::ComputeFullV);
  const Vec4 A = svd.matrixV().col(3), B = svd.matrixV().col(2);
  const Vec3 d = A(3) * B.head<3>() - B(3) * A.head<3>();
  const Vec3 m = A.head<3>().cross(B.head<3>());
  const double nd = d.norm();
  if (nd < 1e-12) return std::nullopt;
  Line L;
  L << d, m;
  return Line(L / nd);
}

double parallax_deg(const Vec3& X, const Pose& a, const Pose& b) {
  const Vec3 u = X - center(a), v = X - center(b);
  const double cosang = u.dot(v) / (u.norm() * v.norm() + 1e-12);
  return std::acos(std::clamp(cosang, -1.0, 1.0)) * kRad2Deg;
}

double plane_angle_deg(const Camera& c, const Pose& a, const Vec4& sa, const Pose& b, const Vec4& sb) {
  const Vec3 n1 = backproject_plane(c, a, sa).head<3>(), n2 = backproject_plane(c, b, sb).head<3>();
  return std::asin(std::clamp(n1.cross(n2).norm(), 0.0, 1.0)) * kRad2Deg;
}

bool point_ok(const Camera& c, const std::optional<Vec3>& X, const std::vector<Pose>& P,
              const std::vector<Vec2>& uv, double min_deg) {
  if (!X) return false;
  if (parallax_deg(*X, P.front(), P.back()) < min_deg) return false;
  for (size_t i = 0; i < P.size(); ++i) {
    const Vec3 Xc = P[i].R * *X + P[i].t;
    if (Xc(2) < 0.5) return false;
    const Vec3 p = c.K * Xc;
    if ((p.head<2>() / p(2) - uv[i]).norm() > 4.0) return false;
  }
  return true;
}

bool line_ok(const Camera& c, const std::optional<Line>& L, const std::vector<Pose>& P,
             const std::vector<Vec4>& segs, double line_min_deg) {
  if (!L) return false;
  if (plane_angle_deg(c, P.front(), segs.front(), P.back(), segs.back()) < line_min_deg)
    return false;  // 선이 baseline과 거의 평행 -> degenerate
  const Vec3 d = L->head<3>(), m0 = L->tail<3>();
  const Vec3 p0 = d.cross(m0);  // 원점에 가장 가까운 직선 위 점 (|d|=1)
  for (size_t i = 0; i < P.size(); ++i) {
    const Vec4& sg = segs[i];
    // 관측 선분 중점의 광선과 직선의 최근접점이 카메라 앞에 있는지 (cheirality)
    const Vec3 C = center(P[i]);
    const Vec3 r = P[i].R.transpose() * (c.Kinv * Vec3((sg(0) + sg(2)) / 2, (sg(1) + sg(3)) / 2, 1.0));
    const Vec3 w0 = C - p0;
    const double a = r.dot(r), b = r.dot(d), cc = d.dot(d);
    const double den = a * cc - b * b;
    if (den < 1e-12) return false;
    const double lam = (b * d.dot(w0) - cc * r.dot(w0)) / den;  // 광선 파라미터
    if (lam < 0.5 || lam > 60) return false;
    const Vec3 l = project_line(c, *L, P[i]);
    const double nl = std::hypot(l(0), l(1));
    const double e1 = std::abs(l.dot(Vec3(sg(0), sg(1), 1.0))) / nl;
    const double e2 = std::abs(l.dot(Vec3(sg(2), sg(3), 1.0))) / nl;
    if (std::max(e1, e2) > 4.0) return false;
  }
  return true;
}

std::vector<int> pick_frames(const std::vector<int>& frames, int k_max) {
  // 관측 프레임이 많으면 처음/끝 포함해서 균등 샘플 (np.linspace(...).round() 와 동일: half-to-even)
  const int n = frames.size();
  if (n <= k_max) return frames;
  const double step = double(n - 1) / (k_max - 1);
  std::vector<int> idx;
  for (int i = 0; i < k_max; ++i)
    idx.push_back(int(std::nearbyint(i == k_max - 1 ? double(n - 1) : i * step)));
  std::sort(idx.begin(), idx.end());
  idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
  std::vector<int> out;
  for (int i : idx) out.push_back(frames[i]);
  return out;
}

VecX sampson(const Mat3& R, const Vec3& t, const std::vector<Vec2>& x1, const std::vector<Vec2>& x2, double fx) {
  const Mat3 E = skew(t) * R;
  VecX r(x1.size());
  for (size_t i = 0; i < x1.size(); ++i) {
    const Vec3 a(x1[i](0), x1[i](1), 1.0), b(x2[i](0), x2[i](1), 1.0);
    const Vec3 Ea = E * a, Eb = E.transpose() * b;
    r(i) = b.dot(Ea) / std::sqrt(Ea(0) * Ea(0) + Ea(1) * Ea(1) + Eb(0) * Eb(0) + Eb(1) * Eb(1)) * fx;
  }
  return r;
}

Pose refine_relative_pose(Pose p, const std::vector<Vec2>& x1, const std::vector<Vec2>& x2, double fx, int iters) {
  // Gauss-Newton on Sampson error, 수치 Jacobian. damping 1e-3: |t|=1 gauge 방향 처리
  Eigen::Matrix<double, Eigen::Dynamic, 6> J(x1.size(), 6);
  for (int it = 0; it < iters; ++it) {
    const VecX r = sampson(p.R, p.t, x1, x2, fx);
    for (int i = 0; i < 6; ++i) {
      Vec6 d = Vec6::Zero();
      d(i) = 1e-7;
      const Vec3 t2 = p.t + d.tail<3>();
      J.col(i) = (sampson(so3_exp(d.head<3>()) * p.R, t2 / t2.norm(), x1, x2, fx) - r) / 1e-7;
    }
    const Eigen::Matrix<double, 6, 6> A = J.transpose() * J + 1e-3 * Eigen::Matrix<double, 6, 6>::Identity();
    const Vec6 dx = -A.partialPivLu().solve(J.transpose() * r);
    p.R = so3_exp(dx.head<3>()) * p.R;
    p.t = (p.t + dx.tail<3>()).normalized();
  }
  return p;
}

VecX residuals(const Camera& c, const Pose& p, const MapObs& m) {
  const int np_ = m.X.size(), nl = m.L.size();
  VecX r(2 * np_ + 2 * nl);
  for (int i = 0; i < np_; ++i) {
    const Vec3 q = c.K * (p.R * m.X[i] + p.t);
    r(2 * i) = q(0) / q(2) - m.uv[i](0);
    r(2 * i + 1) = q(1) / q(2) - m.uv[i](1);
  }
  for (int i = 0; i < nl; ++i) {  // 관측 선분의 두 끝점에서 투영된 직선까지의 거리 [px]
    const Vec3 l = project_line(c, m.L[i], p);
    const double n = std::hypot(l(0), l(1));
    const Vec4& s = m.seg[i];
    r(2 * np_ + 2 * i) = (s(0) * l(0) + s(1) * l(1) + l(2)) / n;
    r(2 * np_ + 2 * i + 1) = (s(2) * l(0) + s(3) * l(1) + l(2)) / n;
  }
  return r;
}

double huber_cost(const VecX& r, double delta) {
  double c = 0;
  for (int i = 0; i < r.size(); ++i) {
    const double a = std::abs(r(i));
    c += a <= delta ? 0.5 * a * a : delta * (a - 0.5 * delta);
  }
  return c;
}

Pose estimate_pose(const Camera& c, Pose p, const MapObs& m, int iters, double huber) {
  // Levenberg-Marquardt on SE(3), Huber IRLS, 수치 Jacobian
  double lam = 1e-3;
  VecX r = residuals(c, p, m);
  double cost = huber_cost(r, huber);
  Eigen::Matrix<double, Eigen::Dynamic, 6> J(r.size(), 6);
  for (int it = 0; it < iters; ++it) {
    const double eps = 1e-6;
    for (int i = 0; i < 6; ++i) {
      Vec6 dx = Vec6::Zero();
      dx(i) = eps;
      J.col(i) = (residuals(c, retract(p, dx), m) - r) / eps;
    }
    const VecX w = huber_weights(r, huber);
    const Eigen::Matrix<double, 6, 6> Hm = J.transpose() * w.asDiagonal() * J;
    const Vec6 g = J.transpose() * w.cwiseProduct(r);
    Vec6 dx;
    while (true) {
      Eigen::Matrix<double, 6, 6> A = Hm;
      A.diagonal() += lam * (Hm.diagonal().array() + 1e-9).matrix();
      dx = -A.partialPivLu().solve(g);
      const Pose p1 = retract(p, dx);
      const VecX r1 = residuals(c, p1, m);
      const double c1 = r1.allFinite() ? huber_cost(r1, huber) : std::numeric_limits<double>::infinity();
      if (c1 < cost) {
        p = p1, r = r1, cost = c1;
        lam = std::max(lam / 3, 1e-7);
        break;
      }
      lam *= 4;
      if (lam > 1e6) return p;
    }
    if (dx.norm() < 1e-10) break;
  }
  return p;
}

double estimate_scale(const Camera& c, const Mat3& Rk, const Vec3& base, const Vec3& that, double s0,
                      const MapObs& m, int iters, double huber) {
  auto f = [&](double s) { return residuals(c, Pose{Rk, base + s * that}, m); };
  double s = s0, lam = 1e-3;
  VecX r = f(s);
  double cost = huber_cost(r, huber);
  for (int it = 0; it < iters; ++it) {
    const VecX J = (f(s + 1e-6) - r) / 1e-6;
    const VecX w = huber_weights(r, huber);
    const double Hs = J.dot(w.cwiseProduct(J)), g = J.dot(w.cwiseProduct(r));
    double ds;
    while (true) {
      ds = -g / (Hs * (1 + lam) + 1e-12);
      const VecX r1 = f(s + ds);
      const double c1 = r1.allFinite() ? huber_cost(r1, huber) : std::numeric_limits<double>::infinity();
      if (c1 < cost) {
        s += ds, r = r1, cost = c1;
        lam = std::max(lam / 3, 1e-7);
        break;
      }
      lam *= 4;
      if (lam > 1e6) return s;
    }
    if (std::abs(ds) < 1e-10) break;
  }
  return s;
}

}  // namespace vp
