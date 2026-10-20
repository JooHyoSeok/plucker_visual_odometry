// g2o types for Plücker lines in the orthonormal representation (Bartoli & Sturm 2005), as python/local_ba.py:
//   L = (d, m) <-> (U in SO(3), phi):  m = cos(phi) U[:,0],  d = sin(phi) U[:,1]  -> d.m = 0 always, 4-DoF update
//   U <- U Exp(theta), phi <- phi + dphi
// EdgeLineProjection: distances [px] of the two observed segment endpoints to the projected image line
//   l = KL m_c,  m_c = R m + [t]x R d  (T_cw = (R, t), g2o VertexSE3Expmap, left update T <- Exp(w, v) T)
// Analytic Jacobians:
//   de/dl = (s^T - e/n [l1 l2 0]) / n,  n = |l_12|
//   pose : dm_c/dw = -[m_c]x,  dm_c/dv = -[d_c]x          (d_c = R d)
//   line : dm/dtheta = -cos(phi) U [e1]x,  dd/dtheta = -sin(phi) U [e2]x,  dm/dphi = -sin(phi) u1,  dd/dphi = cos(phi) u2
//          dm_c = R dm + [t]x R dd
// 라인 좌표 원점 c (세계): 정점 라인은 c 기준 (m' = m - c x d), 엣지는 t' = R c + t 로 같은 투영. 세계 원점에서 먼 라인은
//   |m| 이 커서 phi ~ 0 -> d(거리)/d(phi) = 1/sin^2(phi) 가 수십만, 헤시안이 커져 LM 초기 lambda (최대 대각 x 1e-5) 가 포즈 갱신을
//   누름 (KITTI 06 은 원점에서 수백 VO 단위). local BA 는 라인마다 첫 관측 키프레임 카메라 중심을 c 로
#pragma once
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/types/sba/vertex_se3_expmap.h>

#include <cmath>

#include "vo_plucker/geometry.h"

namespace vp {

struct LineOrth {
  Mat3 U = Mat3::Identity();
  double phi = M_PI / 4;
};

inline LineOrth to_orth(const Line& L) {
  const Vec3 d = L.head<3>(), m = L.tail<3>();
  const double nd = d.norm(), nm = m.norm();
  LineOrth o;
  Vec3 u1 = nm > 1e-12 ? Vec3(m / nm) : Vec3(d.unitOrthogonal());  // 원점을 지나는 직선: m = 0 -> 임의 수직 방향
  const Vec3 u2 = d / nd;
  u1 = (u1 - u1.dot(u2) * u2).normalized();
  o.U << u1, u2, u1.cross(u2);
  o.phi = std::atan2(nd, nm);
  return o;
}

inline Line shift_line(const Line& L, const Vec3& c) {  // 원점을 c 로 옮긴 같은 라인: m' = m - c x d
  Line S = L;
  S.tail<3>() -= c.cross(L.head<3>());
  return S;
}

inline Line from_orth(const LineOrth& o) {  // |d| = 1 (vo_plucker 규약)
  const Vec3 m = std::cos(o.phi) * o.U.col(0), d = std::sin(o.phi) * o.U.col(1);
  Line L;
  L << d / d.norm(), m / d.norm();
  return L;
}

class VertexLineOrth : public g2o::BaseVertex<4, LineOrth> {
 public:
  void setToOriginImpl() override { _estimate = LineOrth{}; }
  void oplusImpl(const double* u) override {
    _estimate.U = _estimate.U * so3_exp(Vec3(u[0], u[1], u[2]));
    _estimate.phi += u[3];
  }
  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
};

class EdgeLineProjection : public g2o::BaseBinaryEdge<2, Vec4, VertexLineOrth, g2o::VertexSE3Expmap> {
 public:
  using Base = g2o::BaseBinaryEdge<2, Vec4, VertexLineOrth, g2o::VertexSE3Expmap>;
  using Base::linearizeOplus;  // 작업공간을 받는 오버로드도 보이게
  Mat3 KL = Mat3::Identity();
  Vec3 c = Vec3::Zero();  // 라인 좌표 원점 (세계)

  void computeError() override {
    Vec3 l, mc, dc;
    project(l, mc, dc);
    const double n = std::hypot(l(0), l(1));
    const Vec4& s = _measurement;
    _error << (s(0) * l(0) + s(1) * l(1) + l(2)) / n, (s(2) * l(0) + s(3) * l(1) + l(2)) / n;
  }

  void linearizeOplus() override {
    Vec3 l, mc, dc;
    project(l, mc, dc);
    const double n = std::hypot(l(0), l(1));
    const Vec4& s = _measurement;
    Eigen::Matrix<double, 2, 3> dedl;
    for (int k = 0; k < 2; ++k) {
      const Vec3 sk(s(2 * k), s(2 * k + 1), 1.0);
      const double e = sk.dot(l) / n;
      dedl.row(k) = (sk.transpose() - (e / n) * Vec3(l(0), l(1), 0).transpose()) / n;
    }
    const Eigen::Matrix<double, 2, 3> A = dedl * KL;  // de / dm_c
    Eigen::Matrix<double, 3, 6> dpose;
    dpose << -skew(mc), -skew(dc);
    _jacobianOplusXj = A * dpose;

    const LineOrth& o = static_cast<const VertexLineOrth*>(_vertices[0])->estimate();
    const g2o::SE3Quat& T = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1])->estimate();
    const Mat3 R = T.rotation().toRotationMatrix();
    const Mat3 tR = skew(R * c + T.translation()) * R;
    const double c = std::cos(o.phi), sn = std::sin(o.phi);
    Eigen::Matrix<double, 3, 4> dline;
    dline.leftCols<3>() = R * (-c * o.U * skew(Vec3::UnitX())) + tR * (-sn * o.U * skew(Vec3::UnitY()));
    dline.col(3) = R * (-sn * o.U.col(0)) + tR * (c * o.U.col(1));
    _jacobianOplusXi = A * dline;
  }

  bool read(std::istream&) override { return false; }
  bool write(std::ostream&) const override { return false; }
  Eigen::Matrix<double, 2, 4> J_line() const { return _jacobianOplusXi; }  // 테스트용
  Eigen::Matrix<double, 2, 6> J_pose() const { return _jacobianOplusXj; }

 private:
  void project(Vec3& l, Vec3& mc, Vec3& dc) const {
    const LineOrth& o = static_cast<const VertexLineOrth*>(_vertices[0])->estimate();
    const g2o::SE3Quat& T = static_cast<const g2o::VertexSE3Expmap*>(_vertices[1])->estimate();
    const Mat3 R = T.rotation().toRotationMatrix();
    const Vec3 m = std::cos(o.phi) * o.U.col(0), d = std::sin(o.phi) * o.U.col(1);
    dc = R * d;
    mc = R * m + (R * c + T.translation()).cross(dc);
    l = KL * mc;
  }
};

}  // namespace vp
