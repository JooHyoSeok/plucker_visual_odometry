#include "vo_plucker/viewer.h"

#include <chrono>
#include <cstdio>
#include <opencv2/imgproc.hpp>
#include <pangolin/pangolin.h>

namespace vp {

namespace {
// 관측 끝점 픽셀의 광선과 3D 직선의 최근접점 -> 직선 위 점 (선분 끝점 복원용)
std::optional<Vec3> backproject_on_line(const Camera& c, const Pose& p, const Line& L, double u, double v) {
  const Vec3 d = L.head<3>(), p0 = d.cross(L.tail<3>()), C = center(p);
  const Vec3 r = p.R.transpose() * (c.Kinv * Vec3(u, v, 1.0));
  const Vec3 w0 = C - p0;
  const double a = r.dot(r), b = r.dot(d), cc = d.dot(d), dd = r.dot(w0), e = d.dot(w0);
  const double den = a * cc - b * b;
  if (den < 1e-12) return std::nullopt;
  return Vec3(p0 + (a * e - b * dd) / den * d);
}

// image line l (ax + by + c = 0) clipped to the image -> two points
bool clip_line(const Vec3& l, int W, int H, cv::Point& p1, cv::Point& p2) {
  std::vector<cv::Point> pts;
  if (std::abs(l(1)) > 1e-9)
    for (double x : {0.0, double(W)}) {
      const double y = -(l(0) * x + l(2)) / l(1);
      if (y >= 0 && y <= H) pts.emplace_back(x, y);
    }
  if (std::abs(l(0)) > 1e-9)
    for (double y : {0.0, double(H)}) {
      const double x = -(l(1) * y + l(2)) / l(0);
      if (x >= 0 && x <= W) pts.emplace_back(x, y);
    }
  if (pts.size() < 2) return false;
  p1 = pts[0], p2 = pts[1];
  return true;
}

pangolin::OpenGlMatrix gl_twc(const Pose& p) {
  pangolin::OpenGlMatrix M;
  M.SetIdentity();
  const Mat3 Rwc = p.R.transpose();
  const Vec3 C = center(p);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) M.m[c * 4 + r] = Rwc(r, c);
    M.m[12 + r] = C(r);
  }
  return M;
}

void draw_camera(const Pose& p, double w) {
  const double h = w * 0.75, z = w * 0.6;
  glPushMatrix();
  glMultMatrixd(gl_twc(p).m);
  glLineWidth(2);
  glColor3f(1.0f, 0.6f, 0.0f);
  glBegin(GL_LINES);
  for (double sx : {-1.0, 1.0})
    for (double sy : {-1.0, 1.0}) glVertex3d(0, 0, 0), glVertex3d(sx * w, sy * h, z);
  for (double s : {-1.0, 1.0}) {
    glVertex3d(-w, s * h, z), glVertex3d(w, s * h, z);
    glVertex3d(s * w, -h, z), glVertex3d(s * w, h, z);
  }
  glEnd();
  glPopMatrix();
}
}  // namespace

Viewer::Viewer(const Camera& cam, std::vector<Vec3> gt_centers, const Pose& gt_cw0) : cam_(cam) {
  for (const Vec3& C : gt_centers) gt_.push_back(gt_cw0.R * C + gt_cw0.t);  // GT를 frame-0 카메라 좌표로
  th_ = std::thread(&Viewer::run, this);
}

Viewer::~Viewer() {
  quit_ = true;
  if (th_.joinable()) th_.join();
}

void Viewer::finish(const std::string& snapshot) {
  if (snapshot.empty()) {
    std::printf("done - close the viewer window to exit\n");
  } else {
    std::lock_guard<std::mutex> lk(mu_);
    snapshot_ = snapshot;
  }
  if (th_.joinable()) th_.join();
}

void Viewer::update(const cv::Mat& bgr, const VOTracker& vo) {
  const int k = int(vo.obs.size()) - 1;
  const FrameObs& o = vo.obs[k];
  const bool tracked = vo.has_pose[k];
  const Pose& p = vo.poses[k];

  // ---- 2D: 추적 중인 점(노랑) / 선(하늘색), 맵 투영 점(초록) / 선(보라) ----
  cv::Mat img = bgr.clone();
  for (const Vec2& u : o.uv) cv::circle(img, cv::Point(u(0), u(1)), 2, {0, 220, 255}, -1);
  for (const Vec4& s : o.seg) cv::line(img, {int(s(0)), int(s(1))}, {int(s(2)), int(s(3))}, {255, 255, 0}, 2, cv::LINE_AA);
  int n_mp = 0, n_ml = 0;
  if (tracked) {
    for (size_t i = 0; i < o.pid.size(); ++i) {
      const auto it = vo.map_p.find(o.pid[i]);
      if (it == vo.map_p.end()) continue;
      const Vec3 q = cam_.K * (p.R * it->second + p.t);
      if (q(2) > 0) cv::circle(img, cv::Point(q(0) / q(2), q(1) / q(2)), 5, {0, 255, 0}, 1, cv::LINE_AA), ++n_mp;
    }
    for (int j : o.lid) {
      const auto it = vo.map_l.find(j);
      cv::Point a, b;
      if (it != vo.map_l.end() && clip_line(project_line(cam_, it->second, p), cam_.W, cam_.H, a, b))
        cv::line(img, a, b, {255, 0, 255}, 1, cv::LINE_AA), ++n_ml;
    }
  }
  const char* state = vo.failed_at >= 0 ? "LOST" : tracked ? "tracking" : "init";
  cv::rectangle(img, {0, 0}, {img.cols, 26}, {0, 0, 0}, -1);
  cv::putText(img,
              cv::format("frame %d  %s  tracked pts %zu  lines %zu  |  map in view: pts %d  lines %d", k, state,
                         o.pid.size(), o.lid.size(), n_mp, n_ml),
              {8, 19}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {255, 255, 255}, 1, cv::LINE_AA);

  // ---- 3D 스냅샷 ----
  std::vector<Vec3> traj, all, vis;
  for (size_t f = 0; f < vo.poses.size(); ++f)
    if (vo.has_pose[f]) traj.push_back(center(vo.poses[f]));
  all.reserve(vo.map_p.size());
  for (const auto& kv : vo.map_p) all.push_back(kv.second);
  for (int j : o.pid)
    if (const auto it = vo.map_p.find(j); tracked && it != vo.map_p.end()) vis.push_back(it->second);

  std::lock_guard<std::mutex> lk(mu_);
  img_ = img, img_new_ = true;
  traj_.swap(traj), pts_all_.swap(all), pts_vis_.swap(vis);
  for (auto it = segs_.begin(); it != segs_.end();) it = vo.map_l.count(it->first) ? std::next(it) : segs_.erase(it);
  segs_vis_.clear();
  if (tracked) {
    for (size_t i = 0; i < o.lid.size(); ++i) {
      const auto it = vo.map_l.find(o.lid[i]);
      if (it == vo.map_l.end()) continue;
      const Vec4& s = o.seg[i];
      const auto A = backproject_on_line(cam_, p, it->second, s(0), s(1));
      const auto B = backproject_on_line(cam_, p, it->second, s(2), s(3));
      // 광선이 직선과 거의 평행하면 끝점이 멀리 튐 -> line_ok와 같은 깊이 범위만 그림
      auto depth_ok = [&](const Vec3& X) { const double z = (p.R * X + p.t)(2); return z > 0.5 && z < 60; };
      if (A && B && depth_ok(*A) && depth_ok(*B)) segs_[o.lid[i]] = {*A, *B}, segs_vis_.push_back(o.lid[i]);
    }
    cur_ = p, has_cur_ = true;
  }
}

void Viewer::draw_map(bool show_pts, bool show_lines, bool show_gt) {
  if (show_pts) {
    glPointSize(1.5);
    glColor3f(0.55f, 0.55f, 0.55f);
    glBegin(GL_POINTS);
    for (const Vec3& X : pts_all_) glVertex3d(X(0), X(1), X(2));
    glEnd();
    glPointSize(3);
    glColor3f(1.0f, 0.2f, 0.2f);
    glBegin(GL_POINTS);
    for (const Vec3& X : pts_vis_) glVertex3d(X(0), X(1), X(2));
    glEnd();
  }
  if (show_lines) {
    glLineWidth(1.5);
    glColor3f(0.3f, 0.5f, 1.0f);
    glBegin(GL_LINES);
    for (const auto& kv : segs_) glVertex3dv(kv.second.first.data()), glVertex3dv(kv.second.second.data());
    glEnd();
    glLineWidth(3);
    glColor3f(0.0f, 0.9f, 1.0f);
    glBegin(GL_LINES);
    for (int j : segs_vis_) glVertex3dv(segs_[j].first.data()), glVertex3dv(segs_[j].second.data());
    glEnd();
  }
  if (show_gt && !gt_.empty() && traj_.size() > 1) {
    // 원점 고정 정렬: 스케일 하나만 처음 50프레임으로 (VO 단위로 GT를 맞춤)
    double num = 0, den = 0;
    for (size_t f = 0; f < std::min<size_t>({50, traj_.size(), gt_.size()}); ++f)
      num += traj_[f].dot(gt_[f]), den += traj_[f].squaredNorm();
    const double s = num / std::max(den, 1e-12);
    glLineWidth(2);
    glColor3f(0.85f, 0.85f, 0.85f);
    glBegin(GL_LINE_STRIP);
    for (const Vec3& Y : gt_) glVertex3d(Y(0) / s, Y(1) / s, Y(2) / s);
    glEnd();
  }
  glLineWidth(2.5);
  glColor3f(0.2f, 0.9f, 0.2f);
  glBegin(GL_LINE_STRIP);
  for (const Vec3& C : traj_) glVertex3d(C(0), C(1), C(2));
  glEnd();
  if (has_cur_) draw_camera(cur_, 1.0);
}

void Viewer::run() {
  const int W = 1280, H = 960, PANEL = 170;
  const double img_frac = 0.32;  // 아래쪽 영상 영역 비율
  pangolin::CreateWindowAndBind("vo_plucker", W, H);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(img_frac, 1.0, 0.0, pangolin::Attach::Pix(PANEL));
  pangolin::Var<bool> follow("menu.Follow camera", true, true), show_pts("menu.Points", true, true),
      show_lines("menu.Lines", true, true), show_gt("menu.Ground truth", true, true), pause("menu.Pause", false, true),
      top("menu.Top view", false, false);
  pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(W, H, 600, 600, W / 2.0, H / 2.0, 0.1, 1e5),
                                    pangolin::ModelViewLookAt(0, -12, -25, 0, 0, 10, 0, -1, 0));
  const double map_aspect = -double(W - PANEL) / (H * (1 - img_frac));
  pangolin::View& d_cam = pangolin::CreateDisplay()
                              .SetBounds(img_frac, 1.0, pangolin::Attach::Pix(PANEL), 1.0, map_aspect)
                              .SetHandler(new pangolin::Handler3D(s_cam));
  pangolin::View& d_img = pangolin::CreateDisplay().SetBounds(0.0, img_frac, 0.0, 1.0, double(cam_.W) / cam_.H);
  pangolin::GlTexture tex(cam_.W, cam_.H, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

  pangolin::OpenGlMatrix last_twc;
  last_twc.SetIdentity();
  int snap_frames = -1;
  while (!pangolin::ShouldQuit() && !quit_) {
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (has_cur_) last_twc = gl_twc(cur_);
      if (pangolin::Pushed(top)) {  // 위에서 내려다보기 (KITTI: y가 아래 방향)
        follow = false;
        const Vec3 C = has_cur_ ? center(cur_) : Vec3::Zero();
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(C(0), C(1) - 150, C(2), C(0), C(1), C(2), 0, 0, 1));
      }
      if (follow) s_cam.Follow(last_twc);
      d_cam.Activate(s_cam);
      draw_map(show_pts, show_lines, show_gt);
      if (img_new_ && !img_.empty()) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // 행 바이트 수(W*3)가 4의 배수가 아니면 기본 정렬(4)로는 영상이 기울어짐
        tex.Upload(img_.data, GL_BGR, GL_UNSIGNED_BYTE), img_new_ = false;
      }
      if (!snapshot_.empty() && snap_frames < 0) snap_frames = 2;
    }
    d_img.Activate();
    glColor3f(1, 1, 1);
    tex.RenderToViewportFlipY();
    if (snap_frames == 1) pangolin::SaveWindowOnRender(snapshot_);
    pangolin::FinishFrame();
    pause_ = pause;
    if (snap_frames > 0 && --snap_frames == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  closed_ = true;
  pangolin::DestroyWindow("vo_plucker");
  
}

}  // namespace vp
