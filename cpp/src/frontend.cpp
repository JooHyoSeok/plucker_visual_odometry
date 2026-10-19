#include "vo_plucker/frontend.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <unordered_set>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/video/tracking.hpp>

namespace vp {

namespace {
const cv::Size kWin(21, 21);
const int kMaxLevel = 3;
const cv::TermCriteria kCrit(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 30, 0.01);

// forward-backward KLT. min_eig: 선 위 점은 gradient가 한 방향뿐이라(rank-1) 기본값(1e-4)이면 버려짐 -> 낮춰서 사용
// normals가 있으면 (선 샘플) edge를 따라 미끄러지는 성분은 무시하고 법선 방향 FB 오차만 본다
void klt(const cv::Mat& prev, const cv::Mat& cur, const std::vector<cv::Point2f>& p0, double fb_thresh,
         double min_eig, const std::vector<Vec2>* normals, const std::vector<cv::Point2f>* guess,
         std::vector<cv::Point2f>& p1, std::vector<char>& ok) {
  const size_t n = p0.size();
  ok.assign(n, 0);
  p1.clear();
  if (!n) return;
  std::vector<uchar> s1, s2;
  std::vector<float> err;
  int flags = 0;
  if (guess) p1 = *guess, flags = cv::OPTFLOW_USE_INITIAL_FLOW;  // 초기 flow 추정값에서 시작 (큰 움직임 대응)
  cv::calcOpticalFlowPyrLK(prev, cur, p0, p1, s1, err, kWin, kMaxLevel, kCrit, flags, min_eig);
  std::vector<cv::Point2f> p0b = p0;
  cv::calcOpticalFlowPyrLK(cur, prev, p1, p0b, s2, err, kWin, kMaxLevel, kCrit, cv::OPTFLOW_USE_INITIAL_FLOW, min_eig);
  for (size_t i = 0; i < n; ++i) {
    const float dx = p0[i].x - p0b[i].x, dy = p0[i].y - p0b[i].y;
    const double fb = normals ? std::abs(dx * (*normals)[i](0) + dy * (*normals)[i](1)) : std::sqrt(dx * dx + dy * dy);
    ok[i] = s1[i] == 1 && s2[i] == 1 && fb < fb_thresh && p1[i].x >= 0 && p1[i].x < cur.cols && p1[i].y >= 0 &&
            p1[i].y < cur.rows;
  }
}

std::vector<Vec2> seg_samples(const Vec4& s, int n) {
  std::vector<Vec2> out;
  const double step = (0.95 - 0.05) / (n - 1);
  for (int i = 0; i < n; ++i) {
    const double a = i == n - 1 ? 0.95 : i * step + 0.05;  // np.linspace(0.05, 0.95, n) 과 비트 단위로 같게
    out.push_back(s.head<2>() * (1 - a) + s.tail<2>() * a);
  }
  return out;
}

// total least squares line fit (float32, Python과 같은 정밀도) -> segment from extreme projections, max residual
Eigen::Vector4f fit_segment(const std::vector<cv::Point2f>& q, float& res) {
  float mx = 0, my = 0;
  for (const auto& p : q) mx += p.x, my += p.y;
  mx /= q.size(), my /= q.size();
  Eigen::Matrix<float, Eigen::Dynamic, 2> X(q.size(), 2);
  for (size_t i = 0; i < q.size(); ++i) X(i, 0) = q[i].x - mx, X(i, 1) = q[i].y - my;
  Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 2>> svd(X, Eigen::ComputeFullV);
  const Eigen::Matrix2f V = svd.matrixV();  // col 0 = 직선 방향, col 1 = 법선
  float smin = 0, smax = 0;
  res = 0;
  for (size_t i = 0; i < q.size(); ++i) {
    const float s = X(i, 0) * V(0, 0) + X(i, 1) * V(1, 0);
    res = std::max(res, std::abs(X(i, 0) * V(0, 1) + X(i, 1) * V(1, 1)));
    smin = i ? std::min(smin, s) : s, smax = i ? std::max(smax, s) : s;
  }
  return Eigen::Vector4f(mx + smin * V(0, 0), my + smin * V(1, 0), mx + smax * V(0, 0), my + smax * V(1, 0));
}

// tracked segment -> nearest LSD detection (angle < 3 deg, perpendicular dist < 3 px, overlapping)
const Vec4* snap_to_lsd(const Eigen::Vector4f& seg, const std::vector<Vec4>& det, const std::vector<Vec2>& det_dir,
                        const std::vector<Vec2>& det_mid) {
  float dx = seg(2) - seg(0), dy = seg(3) - seg(1);
  const float L = std::sqrt(dx * dx + dy * dy);
  dx /= L, dy /= L;
  const Vec2 dd(dx, dy), n(-dy, dx), s0(seg(0), seg(1));
  const double cos3 = std::cos(3.0 * M_PI / 180.0);
  int best = -1;
  double best_perp = 0;
  for (size_t i = 0; i < det.size(); ++i) {
    const double cosang = std::abs(det_dir[i].dot(dd));
    const double perp = std::abs((det_mid[i] - s0).dot(n));
    const double along = (det_mid[i] - s0).dot(dd);
    if (cosang > cos3 && perp < 3.0 && along > -0.2 * L && along < 1.2 * L && (best < 0 || perp < best_perp))
      best = i, best_perp = perp;
  }
  return best < 0 ? nullptr : &det[best];
}

Pose cv_pose(const cv::Mat& R, const cv::Mat& t) {
  Pose p;
  cv::cv2eigen(R, p.R);
  cv::cv2eigen(t, p.t);
  return p;
}
}  // namespace

Frontend::Frontend(const FrontendParams& prm) : prm(prm), lsd_(cv::createLineSegmentDetector(cv::LSD_REFINE_STD)) {
  if (prm.fld) fld_ = cv::ximgproc::createFastLineDetector();
}

FrameObs Frontend::step(const cv::Mat& g) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<cv::Vec4f> raw;
  if (fld_) fld_->detect(g, raw);
  else lsd_->detect(g, raw);
  lsd_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::vector<Vec4> det;
  std::vector<Vec2> det_dir, det_mid;
  for (const auto& l : raw) {
    const Vec4 d(l[0], l[1], l[2], l[3]);
    const Vec2 v = d.tail<2>() - d.head<2>();
    if (v.norm() < prm.min_line_len) continue;
    det.push_back(d), det_dir.push_back(v / (v.norm() + 1e-12)), det_mid.push_back((d.head<2>() + d.tail<2>()) / 2);
  }

  if (!prev_.empty()) {
    // ---- track points and line samples ----
    std::vector<Vec2> samp, nrm;
    for (const auto& s : lines) {
      for (const auto& q : seg_samples(s, prm.n_samples)) samp.push_back(q);
      const Vec2 d = (s.tail<2>() - s.head<2>()).normalized();
      nrm.insert(nrm.end(), prm.n_samples, Vec2(-d(1), d(0)));
    }
    std::vector<cv::Point2f> p1p;
    std::vector<char> ok_p;
    klt(prev_, g, pts, 1.0, 1e-4, nullptr, nullptr, p1p, ok_p);

    // 선 샘플 초기 위치 = 주변 점 트랙 5개의 중앙값 flow 만큼 이동시킨 곳
    std::vector<cv::Point2f> samp32(samp.size()), guess(samp.size());
    for (size_t i = 0; i < samp.size(); ++i) samp32[i] = guess[i] = cv::Point2f(samp[i](0), samp[i](1));
    std::vector<cv::Point2f> good0, flow;
    for (size_t i = 0; i < pts.size(); ++i)
      if (ok_p[i]) good0.push_back(pts[i]), flow.push_back(p1p[i] - pts[i]);
    if (good0.size() >= 5 && !samp.empty()) {
      std::vector<double> d2(good0.size());
      std::vector<int> idx(good0.size());
      for (size_t s = 0; s < samp.size(); ++s) {
        for (size_t i = 0; i < good0.size(); ++i) {
          const double dx = samp[s](0) - good0[i].x, dy = samp[s](1) - good0[i].y;
          d2[i] = dx * dx + dy * dy;
        }
        std::iota(idx.begin(), idx.end(), 0);
        std::nth_element(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) { return d2[a] < d2[b]; });
        float fx[5], fy[5];
        for (int m = 0; m < 5; ++m) fx[m] = flow[idx[m]].x, fy[m] = flow[idx[m]].y;
        std::nth_element(fx, fx + 2, fx + 5), std::nth_element(fy, fy + 2, fy + 5);
        guess[s] = cv::Point2f(samp[s](0) + fx[2], samp[s](1) + fy[2]);
      }
    }
    std::vector<cv::Point2f> p1l;
    std::vector<char> ok_l;
    klt(prev_, g, samp32, 1.0, 1e-6, &nrm, &guess, p1l, ok_l);

    // RANSAC fundamental on point tracks
    std::vector<int> idx;
    std::vector<cv::Point2f> a, b;
    for (size_t i = 0; i < pts.size(); ++i)
      if (ok_p[i]) idx.push_back(i), a.push_back(pts[i]), b.push_back(p1p[i]);
    if (idx.size() >= 8) {
      cv::Mat inl;
      cv::findFundamentalMat(a, b, cv::FM_RANSAC, 1.0, 0.999, 1000, inl);
      if (!inl.empty())
        for (size_t m = 0; m < idx.size(); ++m)
          if (!inl.at<uchar>(m)) ok_p[idx[m]] = 0;
    }
    std::vector<cv::Point2f> pts_new;
    std::vector<int> pid_new;
    for (size_t i = 0; i < pts.size(); ++i)
      if (ok_p[i]) pts_new.push_back(p1p[i]), pid_new.push_back(pid[i]);
    pts.swap(pts_new), pid.swap(pid_new);

    std::vector<Vec4> new_lines;
    std::vector<int> new_lid;
    for (size_t i = 0; i < lines.size(); ++i) {
      std::vector<cv::Point2f> q;
      for (int m = 0; m < prm.n_samples; ++m)
        if (ok_l[i * prm.n_samples + m]) q.push_back(p1l[i * prm.n_samples + m]);
      if (q.size() < 4) continue;
      float res;
      const Eigen::Vector4f seg = fit_segment(q, res);
      if (res > 1.5f) continue;
      const Vec4* snapped = snap_to_lsd(seg, det, det_dir, det_mid);
      if (!snapped || (snapped->tail<2>() - snapped->head<2>()).norm() < prm.min_line_len * 0.8) continue;
      new_lines.push_back(*snapped), new_lid.push_back(lid[i]);
    }
    lines.swap(new_lines), lid.swap(new_lid);
  }

  // ---- replenish points ----
  if (int(pts.size()) < prm.min_pts) {
    cv::Mat mask(g.size(), CV_8U, cv::Scalar(255));
    for (const auto& p : pts) cv::circle(mask, cv::Point(int(p.x), int(p.y)), 8, 0, -1);
    std::vector<cv::Point2f> newp;
    cv::goodFeaturesToTrack(g, newp, prm.max_pts - pts.size(), 0.005, 8, mask);
    for (const auto& p : newp) pts.push_back(p), pid.push_back(next_p_++);
  }

  // ---- replenish lines (긴 것부터, 기존 선과 겹치지 않게) ----
  if (int(lines.size()) < prm.min_lines && !det.empty()) {
    cv::Mat mask(g.size(), CV_8U, cv::Scalar(255));
    auto draw = [&](const Vec4& s) {
      cv::line(mask, cv::Point(int(s(0)), int(s(1))), cv::Point(int(s(2)), int(s(3))), 0, 10);
    };
    for (const auto& s : lines) draw(s);
    std::vector<int> order(det.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<double> len(det.size());
    for (size_t i = 0; i < det.size(); ++i) len[i] = (det[i].tail<2>() - det[i].head<2>()).norm();
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return len[x] > len[y]; });
    const size_t n_before = lines.size();
    for (int i : order) {
      if (int(lines.size()) >= prm.max_lines) break;
      int covered = 0;
      for (const auto& q : seg_samples(det[i], prm.n_samples)) {
        const int x = std::clamp(int(q(0)), 0, g.cols - 1), y = std::clamp(int(q(1)), 0, g.rows - 1);
        covered += mask.at<uchar>(y, x) == 0;
      }
      if (covered >= 3) continue;
      lines.push_back(det[i]);
      draw(det[i]);
    }
    for (size_t i = n_before; i < lines.size(); ++i) lid.push_back(next_l_++);
  }

  prev_ = g.clone();
  ++k_;
  hist_.emplace_back();
  cv::buildOpticalFlowPyramid(prev_, hist_.back(), kWin, kMaxLevel, true);  // reattach가 프레임마다 다시 안 만들게 캐싱
  if (int(hist_.size()) > prm.reattach_frames + 1) hist_.erase(hist_.begin());
  FrameObs o;
  o.pid = pid;
  for (const auto& p : pts) o.uv.emplace_back(p.x, p.y);
  o.lid = lid;
  o.seg = lines;
  o.index();
  return o;
}

int Frontend::reattach(const Camera& cam, const VOTracker& vo, const Pose& pred, FrameObs& o) {
  const int k = k_;
  if (int(vo.obs.size()) != k) return 0;  // o가 vo.obs 바로 다음 프레임이어야 함
  // 1) 후보: 최근 프레임에서 본(가장 최근 관측 기준) 맵 점 중 지금 트랙이 없는 것 -> 예측 포즈로 투영
  struct Cand { int id; cv::Point2f from, guess; };
  std::unordered_map<int, std::vector<Cand>> by_frame;  // 마지막으로 본 프레임 -> 후보
  std::unordered_set<int> seen;
  for (int f = k - 1; f >= std::max(0, k - prm.reattach_frames); --f) {
    const FrameObs& of = vo.obs[f];
    for (size_t i = 0; i < of.pid.size(); ++i) {
      const int j = of.pid[i];
      if (!seen.insert(j).second || o.point(j)) continue;
      const auto it = vo.map_p.find(j);
      if (it == vo.map_p.end()) continue;
      const Vec3 q = cam.K * (pred.R * it->second + pred.t);
      if (q(2) < 0.5) continue;  // 카메라 앞 0.5 이상 (KITTI VO 단위)
      const double u = q(0) / q(2), v = q(1) / q(2);
      if (u < 10 || u >= cam.W - 10 || v < 10 || v >= cam.H - 10) continue;
      by_frame[f].push_back({j, cv::Point2f(of.uv[i](0), of.uv[i](1)), cv::Point2f(u, v)});
    }
  }
  // 현재 트랙 위치 격자 (중복/합치기 판단용, 4 px)
  const float R = 4;
  auto cell = [&](const cv::Point2f& p) { return int(p.y / R) * 100000 + int(p.x / R); };
  std::unordered_multimap<int, int> grid;  // cell -> o 안의 인덱스
  for (size_t i = 0; i < o.uv.size(); ++i) grid.emplace(cell(cv::Point2f(o.uv[i](0), o.uv[i](1))), i);
  auto near = [&](const cv::Point2f& p) -> int {
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        auto r = grid.equal_range(cell(p) + dy * 100000 + dx);
        for (auto it = r.first; it != r.second; ++it)
          if (std::hypot(o.uv[it->second](0) - p.x, o.uv[it->second](1) - p.y) < R) return it->second;
      }
    return -1;
  };

  // 2) 마지막으로 본 프레임 영상 -> 현재 영상 KLT
  const std::vector<cv::Mat>& cur = hist_.back();
  int added = 0;
  for (auto& [f, cands] : by_frame) {
    const int h = int(hist_.size()) - 1 - (k - f);
    if (h < 0) continue;
    std::vector<cv::Point2f> p0, p1;
    for (const auto& c : cands) p0.push_back(c.from), p1.push_back(c.guess);
    std::vector<uchar> s1, s2;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(hist_[h], cur, p0, p1, s1, err, kWin, kMaxLevel, kCrit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
    std::vector<cv::Point2f> p0b = p0;
    cv::calcOpticalFlowPyrLK(cur, hist_[h], p1, p0b, s2, err, kWin, kMaxLevel, kCrit, cv::OPTFLOW_USE_INITIAL_FLOW, 1e-4);
    for (size_t i = 0; i < cands.size(); ++i) {
      const cv::Point2f& p = p1[i];
      if (!s1[i] || !s2[i] || cv::norm(p0[i] - p0b[i]) > 1.0 || cv::norm(p - cands[i].guess) > prm.reattach_px ||
          p.x < 0 || p.x >= cam.W || p.y < 0 || p.y >= cam.H)
        continue;
      const int ni = near(p);
      if (ni >= 0) {
        if (vo.map_p.count(o.pid[ni])) continue;  // 이미 맵 점이 추적 중인 자리
        o.pid[ni] = pid[ni] = cands[i].id;        // 맵에 없는 새 트랙 -> 원래 맵 점 id로 합침
      } else {
        o.pid.push_back(cands[i].id), o.uv.emplace_back(p.x, p.y);
        pts.push_back(p), pid.push_back(cands[i].id);
        grid.emplace(cell(p), o.uv.size() - 1);
      }
      ++added;
    }
  }
  if (added) o.index();
  return added;
}

Pose robust_init(const Camera& c, const FrameObs& o0, const FrameObs& oI, double dist_thresh) {
  std::vector<int> ids;
  for (int j : o0.pid)
    if (oI.point(j)) ids.push_back(j);
  std::sort(ids.begin(), ids.end());
  std::vector<cv::Point2d> p1, p2;
  for (int j : ids) {
    const Vec2 &a = *o0.point(j), &b = *oI.point(j);
    p1.emplace_back(a(0), a(1)), p2.emplace_back(b(0), b(1));
  }
  cv::Mat K, inl, R, t;
  cv::eigen2cv(c.K, K);
  const cv::Mat E = cv::findEssentialMat(p1, p2, K, cv::RANSAC, 0.999, 1.0, 1000, inl);
  cv::recoverPose(E.rowRange(0, 3), p1, p2, K, R, t, dist_thresh, inl);  // 해가 여러 개면 첫 번째
  std::vector<Vec2> x1, x2;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (!inl.at<uchar>(i)) continue;
    x1.push_back((c.Kinv * Vec3(p1[i].x, p1[i].y, 1.0)).head<2>());
    x2.push_back((c.Kinv * Vec3(p2[i].x, p2[i].y, 1.0)).head<2>());
  }
  Pose p = cv_pose(R, t);
  p.t.normalize();
  std::printf("  init: %zu/%zu inliers\n", x1.size(), ids.size());
  return refine_relative_pose(p, x1, x2, c.fx);
}

void Frontend::drop(const std::vector<int>& ids) {
  if (ids.empty()) return;
  const std::unordered_set<int> bad(ids.begin(), ids.end());
  size_t n = 0;
  for (size_t i = 0; i < pid.size(); ++i)
    if (!bad.count(pid[i])) pts[n] = pts[i], pid[n] = pid[i], ++n;
  pts.resize(n), pid.resize(n);
}

std::optional<Pose> relative_pose(const Camera& c, const FrameObs& a, const FrameObs& b,
                                  const std::unordered_map<int, Vec3>* static_ids, double dist_thresh) {
  if (static_ids) {
    int n = 0;
    for (int j : b.pid) n += a.point(j) && static_ids->count(j);
    if (n < 30) static_ids = nullptr;
  }
  std::vector<cv::Point2d> x1, x2;
  for (size_t i = 0; i < b.pid.size(); ++i) {
    const Vec2* pa = a.point(b.pid[i]);
    if (!pa || (static_ids && !static_ids->count(b.pid[i]))) continue;
    const Vec2 u = (c.Kinv * Vec3((*pa)(0), (*pa)(1), 1.0)).head<2>();
    const Vec2 w = (c.Kinv * Vec3(b.uv[i](0), b.uv[i](1), 1.0)).head<2>();
    x1.emplace_back(u(0), u(1)), x2.emplace_back(w(0), w(1));
  }
  if (x1.size() < 15) return std::nullopt;
  const cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat inl, R, t;
  const cv::Mat E = cv::findEssentialMat(x1, x2, I, cv::RANSAC, 0.999, 1.0 / c.fx, 1000, inl);
  if (E.rows != 3 || E.cols != 3) return std::nullopt;
  cv::recoverPose(E, x1, x2, I, R, t, dist_thresh, inl);
  std::vector<Vec2> v1, v2;
  for (size_t i = 0; i < x1.size(); ++i)
    if (inl.at<uchar>(i)) v1.emplace_back(x1[i].x, x1[i].y), v2.emplace_back(x2[i].x, x2[i].y);
  if (v1.size() < 15) return std::nullopt;
  return refine_relative_pose(cv_pose(R, t), v1, v2, c.fx, 15);
}

}  // namespace vp
