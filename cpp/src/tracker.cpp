#include "vo_plucker/tracker.h"

#include <algorithm>
#include <stdexcept>

namespace vp {

void FrameObs::index() {
  pi_.clear();
  li_.clear();
  for (size_t i = 0; i < pid.size(); ++i) pi_[pid[i]] = i;
  for (size_t i = 0; i < lid.size(); ++i) li_[lid[i]] = i;
}

const Vec2* FrameObs::point(int j) const {
  const auto it = pi_.find(j);
  return it == pi_.end() ? nullptr : &uv[it->second];
}

const Vec4* FrameObs::line(int j) const {
  const auto it = li_.find(j);
  return it == li_.end() ? nullptr : &seg[it->second];
}

namespace {
Pose const_velocity(const Pose& a, const Pose& b) {  // a = k-2, b = k-1 -> k
  const Mat3 dR = b.R * a.R.transpose();
  const Vec3 dt = b.t - dR * a.t;
  return Pose{orthonormalize(dR * b.R), dR * b.t + dt};
}
}  // namespace

std::optional<Pose> VOTracker::predict() const {
  const int k = obs.size();
  if (failed_at >= 0 || k < 2 || !has_pose[k - 1] || !has_pose[k - 2]) return std::nullopt;
  return const_velocity(poses[k - 2], poses[k - 1]);
}

std::vector<int> VOTracker::add(FrameObs o, const std::optional<Pose>& r, const std::optional<Pose>& init) {
  const int k = obs.size();
  o.index();
  for (int j : o.pid) first_p.emplace(j, k);  // emplace = setdefault (처음 본 프레임 유지)
  for (int j : o.lid) first_l.emplace(j, k);
  obs.push_back(std::move(o));
  poses.emplace_back();
  has_pose.push_back(0);
  if (r) rel[k] = *r;
  const int I = prm.init_frame;
  if (failed_at >= 0 || k < I) return {};
  if (k > I) return track(k) ? std::vector<int>{k} : std::vector<int>{};

  // frame 0 & init_frame: 초기화 후 1..init_frame-1을 몰아서 추적
  if (!init) throw std::runtime_error("VOTracker: init pose required at init_frame");
  poses[0] = Pose{};
  poses[I] = Pose{init->R, init->t / init->t.norm()};
  has_pose[0] = has_pose[I] = 1;
  update_map(I);
  std::vector<int> done{0, I};
  for (int f = 1; f < I; ++f) {
    if (!track(f)) break;
    done.push_back(f);
  }
  return done;
}

void VOTracker::update_map(int k) {
  // 보이는 랜드마크를 지금까지의 모든 관측(포즈 아는 프레임)으로 다시 삼각측량 (structure-only 정제)
  const FrameObs& o = obs[k];
  std::vector<int> fr;
  std::vector<Pose> P;
  for (int j : o.pid) {
    const int f0 = first_p.at(j);
    if (k - f0 < prm.tri_gap || mature_p.count(j)) continue;
    fr.clear();
    for (int f = f0; f <= k; ++f)
      if (has_pose[f] && obs[f].point(j)) fr.push_back(f);
    fr = pick_frames(fr);
    P.clear();
    std::vector<Vec2> uvs;
    for (int f : fr) P.push_back(poses[f]), uvs.push_back(*obs[f].point(j));
    const auto X = tri_point_multi(cam, P, uvs);
    if (point_ok(cam, X, P, uvs, prm.point_min_deg)) {
      map_p[j] = *X;
      if (parallax_deg(*X, P.front(), P.back()) > prm.mature_deg) mature_p.insert(j);
    }
  }
  if (!prm.use_lines) return;
  for (int j : o.lid) {
    const int f0 = first_l.at(j);
    if (k - f0 < prm.tri_gap || mature_l.count(j)) continue;
    fr.clear();
    for (int f = f0; f <= k; ++f)
      if (has_pose[f] && obs[f].line(j)) fr.push_back(f);
    fr = pick_frames(fr);
    P.clear();
    std::vector<Vec4> sgs;
    for (int f : fr) P.push_back(poses[f]), sgs.push_back(*obs[f].line(j));
    const auto L = tri_line_multi(cam, P, sgs);
    if (line_ok(cam, L, P, sgs, prm.line_min_deg)) {
      map_l[j] = *L;
      if (plane_angle_deg(cam, P.front(), sgs.front(), P.back(), sgs.back()) > prm.mature_deg) mature_l.insert(j);
    } else if (!map_l.count(j)) {
      ++line_rejected;
    }
  }
}

bool VOTracker::track(int k) {
  // constant-velocity prediction
  Pose p0;
  if (k >= 2) {
    p0 = const_velocity(poses[k - 2], poses[k - 1]);
  } else {  // k=1: 초기화 포즈를 선형 보간한 값에서 시작
    p0 = Pose{poses[k - 1].R, poses[prm.init_frame].t / prm.init_frame};
  }

  const FrameObs& o = obs[k];
  std::vector<int> pid, lid;
  MapObs m;
  for (size_t i = 0; i < o.pid.size(); ++i) {
    const auto it = map_p.find(o.pid[i]);
    if (it == map_p.end()) continue;
    pid.push_back(o.pid[i]), m.X.push_back(it->second), m.uv.push_back(o.uv[i]);
  }
  if (prm.use_lines) {
    for (size_t i = 0; i < o.lid.size(); ++i) {
      const auto it = map_l.find(o.lid[i]);
      if (it == map_l.end()) continue;
      lid.push_back(o.lid[i]), m.L.push_back(it->second), m.seg.push_back(o.seg[i]);
    }
  }
  if (2 * (pid.size() + lid.size()) < 12) return fail(k, "too few map correspondences");  // 대응점 부족 -> tracking lost

  const auto rit = rel.find(k);
  auto solve = [&](const MapObs& mm, const Pose& pi) {
    if (rit == rel.end()) return estimate_pose(cam, pi, mm);
    const Pose& rr = rit->second;
    const Mat3 Rk = rr.R * poses[k - 1].R;
    const Vec3 base = rr.R * poses[k - 1].t;
    const double sc = estimate_scale(cam, Rk, base, rr.t, std::max((pi.t - base).dot(rr.t), 1e-3), mm);
    return Pose{Rk, base + sc * rr.t};
  };

  Pose p1 = solve(m, p0);
  // outlier culling: 이 프레임에서 크게 안 맞는 랜드마크는 맵에서 빼고(다시 삼각측량 대상) 포즈 재추정
  VecX r = residuals(cam, p1, m);
  const int np_ = pid.size();
  MapObs kept;
  bool culled = false;
  for (int i = 0; i < np_; ++i) {
    if (r.segment<2>(2 * i).norm() < prm.outlier_px) {
      kept.X.push_back(m.X[i]), kept.uv.push_back(m.uv[i]);
    } else {
      map_p.erase(pid[i]), mature_p.erase(pid[i]), culled = true;
    }
  }
  for (size_t i = 0; i < lid.size(); ++i) {
    const double e = std::max(std::abs(r(2 * np_ + 2 * i)), std::abs(r(2 * np_ + 2 * i + 1)));
    if (e < prm.outlier_px) {
      kept.L.push_back(m.L[i]), kept.seg.push_back(m.seg[i]);
    } else {
      map_l.erase(lid[i]), mature_l.erase(lid[i]), culled = true;
    }
  }
  if (kept.X.size() + kept.L.size() < 6) return fail(k, "almost all map correspondences are outliers");  // 맵이 거의 다 안 맞음 -> 실패
  if (culled) p1 = solve(kept, p1);
  r = residuals(cam, p1, kept);
  VecX a = r.cwiseAbs();
  const int n = a.size();
  std::sort(a.data(), a.data() + n);
  const double med = n % 2 ? a(n / 2) : 0.5 * (a(n / 2 - 1) + a(n / 2));
  if (!r.allFinite() || med > 3.0 * prm.pix_noise) return fail(k, "median reprojection error too large");  // tracking failure -> 여기서 멈춤

  poses[k] = Pose{orthonormalize(p1.R), p1.t};
  has_pose[k] = 1;
  update_map(k);
  return true;
}

}  // namespace vp
