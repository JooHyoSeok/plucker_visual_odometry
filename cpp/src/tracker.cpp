#include "vo_plucker/tracker.h"

#include "vo_plucker/optimizer.h"

#include <algorithm>
#include <cmath>

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

std::vector<int> VOTracker::inconsistent_tracks(double px) const {
  const int k = int(obs.size()) - 1;
  std::vector<int> out;
  if (k < 1 || !has_pose[k] || !has_pose[k - 1]) return out;
  const Pose &a = poses[k - 1], &b = poses[k];
  const Mat3 Rr = b.R * a.R.transpose();
  const Vec3 tr = b.t - Rr * a.t;
  std::vector<double> depth;
  for (size_t i = 0; i < obs[k].pid.size(); ++i)
    if (const auto it = map_p.find(obs[k].pid[i]); it != map_p.end()) depth.push_back((b.R * it->second + b.t)(2));
  if (depth.size() < 10) return out;
  std::nth_element(depth.begin(), depth.begin() + depth.size() / 2, depth.end());
  const bool still = (center(b) - center(a)).norm() / depth[depth.size() / 2] < 0.002;
  const Mat3 H = cam.K * Rr * cam.Kinv;  // 순수 회전 homography
  std::vector<int> ids;
  std::vector<Vec2> x1, x2;
  for (size_t i = 0; i < obs[k].pid.size(); ++i) {
    const int j = obs[k].pid[i];
    const Vec2* u0 = obs[k - 1].point(j);
    if (!u0 || map_p.count(j)) continue;
    const Vec2& u1 = obs[k].uv[i];
    if (still) {
      const Vec3 q = H * Vec3((*u0)(0), (*u0)(1), 1.0);
      if ((q.head<2>() / q(2) - u1).norm() > px) out.push_back(j);
    } else {
      ids.push_back(j);
      x1.push_back((cam.Kinv * Vec3((*u0)(0), (*u0)(1), 1.0)).head<2>());
      x2.push_back((cam.Kinv * Vec3(u1(0), u1(1), 1.0)).head<2>());
    }
  }
  if (!still && !ids.empty()) {
    const VecX e = sampson(Rr, tr, x1, x2, cam.fx);
    for (size_t i = 0; i < ids.size(); ++i)
      if (std::abs(e(i)) > px) out.push_back(ids[i]);
  }
  return out;
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
  if (failed_at >= 0) return {};
  if (init_frame_ >= 0) return track(k) ? std::vector<int>{k} : std::vector<int>{};
  if (prm.use_depth) {  // 깊이 초기화: 깊이 있는 점이 충분한 첫 프레임 = world (그 전 프레임은 포즈 없음)
    int nd = 0;
    for (float d : obs[k].depth) nd += std::isfinite(d);
    if (nd < prm.init_min_points) return {};
    init_frame_ = k, poses[k] = Pose{}, has_pose[k] = 1, kfs = {k};
    add_depth_points(k);
    return {k};
  }
  if (k == 0 || !init) return {};  // 초기화 전: 버퍼링

  // frame 0 & 초기화 프레임 I: 초기화 후 1..I-1을 몰아서 추적
  const int I = init_frame_ = k;
  poses[0] = Pose{};
  poses[I] = Pose{init->R, init->t / init->t.norm()};
  has_pose[0] = has_pose[I] = 1;
  kfs = {0, I};
  update_map(I);
  bool ok = int(map_p.size()) >= prm.init_min_points;
  if (!ok) init_note = "only " + std::to_string(map_p.size()) + " map points";
  std::vector<int> done{0, I};
  for (int f = 1; f < I && ok; ++f) {
    ok = track(f);
    if (!ok) init_note = "frame " + std::to_string(f) + ": " + fail_reason;
    done.push_back(f);
  }
  if (!ok) {  // 맵이 빈약하거나 중간 프레임 추적 실패 -> 이번 초기화는 버리고 다음 프레임에서 다시
    const std::string note = init_note;
    reset_init();
    init_note = note;
    return {};
  }
  return done;
}

void VOTracker::add_depth_points(int k) {
  const FrameObs& o = obs[k];
  const Pose& p = poses[k];
  for (size_t i = 0; i < o.depth.size(); ++i) {
    const int j = o.pid[i];
    if (!std::isfinite(o.depth[i]) || mature_p.count(j)) continue;
    map_p[j] = p.R.transpose() * (double(o.depth[i]) * (cam.Kinv * Vec3(o.uv[i](0), o.uv[i](1), 1.0)) - p.t);
    mature_p.insert(j), ++n_depth_pts;
  }
}

void VOTracker::reset_init() {
  map_p.clear(), map_l.clear(), mature_p.clear(), mature_l.clear(), kfs.clear();
  std::fill(has_pose.begin(), has_pose.end(), 0);
  init_frame_ = failed_at = -1, fail_reason.clear(), line_rejected = n_still = n_map6dof = 0;
}

void VOTracker::update_map(int k) {
  // 보이는 랜드마크를 지금까지의 관측(포즈 아는 프레임, kf_mapping이면 키프레임만)으로 다시 삼각측량 (structure-only 정제)
  const FrameObs& o = obs[k];
  std::vector<int> fr;
  std::vector<Pose> P;
  // 랜드마크 j를 본 프레임들 (f0 = 처음 본 프레임). 삼각측량할 만큼 모이지 않았으면 false
  auto frames_for = [&](int f0, auto&& seen) {
    fr.clear();
    if (prm.kf_mapping) {
      for (auto it = std::lower_bound(kfs.begin(), kfs.end(), f0); it != kfs.end() && *it <= k; ++it)
        if (seen(*it)) fr.push_back(*it);
      if (fr.size() < 2) return false;
    } else {
      if (k - f0 < prm.tri_gap) return false;
      for (int f = f0; f <= k; ++f)
        if (has_pose[f] && seen(f)) fr.push_back(f);
    }
    fr = pick_frames(fr);
    return true;
  };
  for (int j : o.pid) {
    if (mature_p.count(j) || !frames_for(first_p.at(j), [&](int f) { return obs[f].point(j) != nullptr; })) continue;
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
    if (mature_l.count(j) || !frames_for(first_l.at(j), [&](int f) { return obs[f].line(j) != nullptr; })) continue;
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

bool VOTracker::is_keyframe(int k, int n_map_used) const {
  // 직전 키프레임 대비 회전 보정 parallax (같은 트랙의 픽셀 이동에서 회전만으로 설명되는 부분을 뺀 것) 중앙값
  const int last = kfs.back();
  const Mat3 H = cam.K * poses[k].R * poses[last].R.transpose() * cam.Kinv;
  std::vector<double> d;
  for (size_t i = 0; i < obs[k].pid.size(); ++i)
    if (const Vec2* u = obs[last].point(obs[k].pid[i])) {
      const Vec3 q = H * Vec3((*u)(0), (*u)(1), 1.0);
      d.push_back((q.head<2>() / q(2) - obs[k].uv[i]).norm());
    }
  if (d.size() < 10) return true;  // 직전 키프레임과 공통 트랙이 거의 없음 -> 새 키프레임 필요
  std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
  const double med = d[d.size() / 2];
  if (med >= prm.kf_parallax_px) return true;
  return med >= 2.0 && (n_map_used < prm.kf_min_map || k - last >= prm.kf_max_gap);
}

bool VOTracker::track(int k) {
  // constant-velocity prediction
  Pose p0;
  if (k >= 2 && has_pose[k - 2] && has_pose[k - 1]) {
    p0 = const_velocity(poses[k - 2], poses[k - 1]);
  } else if (k > init_frame_) {  // 깊이 초기화 직후: 직전 포즈 그대로
    p0 = poses[k - 1];
  } else {  // 2-view 초기화 사이의 k=1: 초기화 포즈를 선형 보간한 값에서 시작
    p0 = Pose{poses[k - 1].R, poses[init_frame_].t / init_frame_};
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

  // 정지 판정: 맵 점 트랙이 직전 프레임에서 거의 안 움직였으면 포즈 유지 (2D-2D 방향은 노이즈, 삼각측량도 baseline 0)
  if (prm.still_px > 0 && k > init_frame_) {
    std::vector<double> flow;
    for (size_t i = 0; i < pid.size(); ++i)
      if (const Vec2* u = obs[k - 1].point(pid[i])) flow.push_back((*u - m.uv[i]).norm());
    if (flow.size() >= 20) {
      std::nth_element(flow.begin(), flow.begin() + flow.size() / 2, flow.end());
      if (flow[flow.size() / 2] < prm.still_px) {
        poses[k] = poses[k - 1], has_pose[k] = 1, ++n_still;
        return true;
      }
    }
  }

  auto rit = rel.find(k);
  if (prm.use_depth && prm.depth_6dof) rit = rel.end();  // metric 맵이 있으니 맵 기준 6-DoF (RGB-D VO 방식)
  if (rit != rel.end()) {
    bool weak = prm.catchup_6dof && k < init_frame_;  // 초기화 사이 프레임: 맵(0, I baseline)이 2D-2D보다 확실함
    if (!weak && prm.rel_min_px > 0) {  // 2D-2D 회전으로 보정한 맵 점 트랙 flow가 작으면 이동 방향이 노이즈
      const Mat3 H = cam.K * rit->second.R * cam.Kinv;
      std::vector<double> d;
      for (size_t i = 0; i < pid.size(); ++i)
        if (const Vec2* u = obs[k - 1].point(pid[i])) {
          const Vec3 q = H * Vec3((*u)(0), (*u)(1), 1.0);
          d.push_back((q.head<2>() / q(2) - m.uv[i]).norm());
        }
      if (d.size() >= 20) {
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        weak = d[d.size() / 2] < prm.rel_min_px;
      }
    }
    if (weak) rit = rel.end(), ++n_map6dof;
  }
  auto solve = [&](const MapObs& mm, const Pose& pi) {
    if (rit == rel.end()) return prm.pose_g2o ? motion_only_ba(cam, pi, mm) : estimate_pose(cam, pi, mm);
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
  if (!prm.kf_mapping) {
    update_map(k);
  } else if (k > init_frame_ && is_keyframe(k, kept.X.size())) {
    kfs.push_back(k);
    update_map(k);
  }
  if (prm.use_depth) add_depth_points(k);  // 깊이 있는 점은 삼각측량 결과보다 우선
  return true;
}

}  // namespace vp
