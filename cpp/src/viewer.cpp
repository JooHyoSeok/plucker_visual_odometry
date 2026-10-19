#include "vo_plucker/viewer.h"

#include <algorithm>
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

Viewer::Viewer(const Camera& cam, std::vector<Pose> gt_cw, bool sim3_align, std::optional<RunOptions> start)
    : cam_(cam), gt_cw_(std::move(gt_cw)), align_init_(sim3_align), start_(std::move(start)) {
  th_ = std::thread(&Viewer::run, this);
}

std::optional<RunOptions> Viewer::wait_start() {
  std::unique_lock<std::mutex> lk(mu_);
  start_cv_.wait(lk, [&] { return started_ || closed_; });
  return started_ ? start_ : std::nullopt;
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
  if (f0_ < 0 && tracked) f0_ = std::find(vo.has_pose.begin(), vo.has_pose.end(), 1) - vo.has_pose.begin();
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

bool Viewer::sim3_to_gt(Eigen::Matrix4d& S, double& ate) const {
  if (f0_ < 0) return false;
  const int n = std::min<int>(traj_.size(), int(gt_cw_.size()) - f0_);
  if (n < 3) return false;
  Eigen::Matrix<double, 3, Eigen::Dynamic> X(3, n), Y(3, n);
  for (int i = 0; i < n; ++i) X.col(i) = traj_[i], Y.col(i) = center(gt_cw_[f0_ + i]);
  S = Eigen::umeyama(X, Y, true);
  ate = std::sqrt(((S.topLeftCorner<3, 3>() * X).colwise() + S.topRightCorner<3, 1>() - Y).colwise().squaredNorm().mean());
  return true;
}

void Viewer::draw_map(bool show_pts, bool show_lines, bool show_gt, const Eigen::Matrix4d* S) {
  if (S) {  // 추정한 것들(점, 선, 궤적, 카메라)을 GT 좌표계로: Eigen은 column-major라 그대로 GL 행렬
    glPushMatrix();
    glMultMatrixd(S->data());
  }
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
  glLineWidth(2.5);
  glColor3f(0.2f, 0.9f, 0.2f);
  glBegin(GL_LINE_STRIP);
  for (const Vec3& C : traj_) glVertex3d(C(0), C(1), C(2));
  glEnd();
  if (has_cur_) draw_camera(cur_, 1.0);
  if (S) glPopMatrix();

  if (S && show_gt) {  // GT는 자기 world 좌표 그대로 + 프레임별 위치 오차 (10프레임마다)
    const int n = std::min<int>(traj_.size(), int(gt_cw_.size()) - f0_);
    glLineWidth(2);
    glColor3f(0.85f, 0.85f, 0.85f);
    glBegin(GL_LINE_STRIP);
    for (size_t f = f0_; f < gt_cw_.size(); ++f) glVertex3dv(center(gt_cw_[f]).data());
    glEnd();
    glLineWidth(1);
    glColor3f(1.0f, 0.85f, 0.2f);  // 노랑: 위치 오차 (빨강은 보이는 맵 점)
    glBegin(GL_LINES);
    for (int i = 0; i < n; i += 10) {
      const Vec3 a = S->topLeftCorner<3, 3>() * traj_[i] + S->topRightCorner<3, 1>();
      glVertex3dv(a.data()), glVertex3dv(center(gt_cw_[f0_ + i]).data());
    }
    glEnd();
  } else if (show_gt && f0_ >= 0 && f0_ < int(gt_cw_.size()) && traj_.size() > 1) {
    // 원점 고정 정렬: 시작 프레임 GT 카메라 좌표로 옮기고, 스케일 하나만 처음 50프레임으로 (VO 단위로 GT를 맞춤)
    const Pose& g0 = gt_cw_[f0_];
    std::vector<Vec3> Y;
    for (size_t f = f0_; f < gt_cw_.size(); ++f) Y.push_back(g0.R * center(gt_cw_[f]) + g0.t);
    double num = 0, den = 0;
    for (size_t f = 0; f < std::min<size_t>({50, traj_.size(), Y.size()}); ++f)
      num += traj_[f].dot(Y[f]), den += traj_[f].squaredNorm();
    const double s = num / std::max(den, 1e-12);
    glLineWidth(2);
    glColor3f(0.85f, 0.85f, 0.85f);
    glBegin(GL_LINE_STRIP);
    for (const Vec3& y : Y) glVertex3d(y(0) / s, y(1) / s, y(2) / s);
    glEnd();
  }
}

void Viewer::run() {
  const int W = 1280, H = 960, PANEL = 200;
  const double img_frac = 0.32;  // 아래쪽 영상 영역 비율
  pangolin::CreateWindowAndBind("vo_plucker", W, H);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::CreatePanel("menu").SetBounds(img_frac, 1.0, 0.0, pangolin::Attach::Pix(PANEL));
  // 시작 화면: 체크박스 묶음을 서로 배타적으로 (Pangolin 에 콤보박스가 없음). Start 뒤에는 잠금
  using Box = std::unique_ptr<pangolin::Var<bool>>;
  const std::vector<std::string> PR{"thumb", "netvlad", "salad", "mast3r"}, VER{"orb", "mast3r", "orb+mast3r"};
  Box start_btn, ui_loop, ui_lines;
  std::vector<Box> ui_pr, ui_ver;
  std::unique_ptr<pangolin::Var<std::string>> ui_state;
  int sel_pr = 0, sel_ver = 0;
  if (start_) {
    start_btn = std::make_unique<pangolin::Var<bool>>("menu.Start", false, false);
    ui_state = std::make_unique<pangolin::Var<std::string>>("menu.State", "press Start");
    ui_loop = std::make_unique<pangolin::Var<bool>>("menu.Loop closure", start_->loop, true);
    for (size_t i = 0; i < PR.size(); ++i) {
      ui_pr.push_back(std::make_unique<pangolin::Var<bool>>("menu.Detect " + PR[i], PR[i] == start_->pr, true));
      if (PR[i] == start_->pr) sel_pr = i;
    }
    for (size_t i = 0; i < VER.size(); ++i) {
      ui_ver.push_back(std::make_unique<pangolin::Var<bool>>("menu.Verify " + VER[i], VER[i] == start_->ver, true));
      if (VER[i] == start_->ver) sel_ver = i;
    }
    ui_lines = std::make_unique<pangolin::Var<bool>>("menu.Track lines", start_->lines, true);
  }
  auto exclusive = [&](std::vector<Box>& box, int& sel) {  // 클릭한 것 하나만 켜짐 (Start 뒤에는 클릭 무시)
    for (int i = 0; i < int(box.size()); ++i)
      if (box[i]->GuiChanged() && !started_) sel = i;
    for (int i = 0; i < int(box.size()); ++i) *box[i] = i == sel;
  };
  pangolin::Var<bool> follow("menu.Follow camera", true, true), show_pts("menu.Points", true, true),
      show_lines("menu.Lines", true, true), show_gt("menu.Ground truth", true, true), pause("menu.Pause", false, true),
      top("menu.Top view", false, false), align("menu.Sim3 align", align_init_, true);
  pangolin::Var<std::string> ate_txt("menu.ATE", "-");
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
    if (start_) {
      exclusive(ui_pr, sel_pr), exclusive(ui_ver, sel_ver);
      if (started_) *ui_loop = start_->loop, *ui_lines = start_->lines;
      if (pangolin::Pushed(*start_btn) && !started_) {
        std::lock_guard<std::mutex> lk(mu_);
        *start_ = RunOptions{bool(*ui_loop), PR[sel_pr], VER[sel_ver], bool(*ui_lines)};
        started_ = true;
        *ui_state = start_->loop ? "running: " + start_->pr + " + " + start_->ver : "running: no loop";
        start_cv_.notify_all();
      }
    }
    glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    {
      std::lock_guard<std::mutex> lk(mu_);
      Eigen::Matrix4d S;
      double ate = 0;
      const bool aligned = align && sim3_to_gt(S, ate);
      ate_txt = aligned ? cv::format("%.2f m s=%.2f", ate, std::cbrt(S.topLeftCorner<3, 3>().determinant())) : "-";
      // 정렬 중이면 카메라도 GT 좌표계로 (Follow / Top view용, 회전과 위치만)
      Pose view = has_cur_ ? cur_ : Pose{};
      Mat3 R0 = Mat3::Identity();  // 시작 카메라 -> world (위쪽 = -카메라 y, 앞 = 카메라 z)
      if (aligned) {
        const double s = std::cbrt(S.topLeftCorner<3, 3>().determinant());
        const Mat3 Rs = S.topLeftCorner<3, 3>() / s;
        const Vec3 C = S.topLeftCorner<3, 3>() * center(view) + S.topRightCorner<3, 1>();
        view.R = view.R * Rs.transpose(), view.t = -view.R * C;
        R0 = gt_cw_[f0_].R.transpose();
      }
      if (has_cur_) last_twc = gl_twc(view);
      if (pangolin::Pushed(top)) {  // 위에서 내려다보기 (카메라 y가 아래 방향인 차량/핸드헬드 기준)
        follow = false;
        const Vec3 C = center(view), eye = C - 150 * R0.col(1), up = R0.col(2);
        s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(eye(0), eye(1), eye(2), C(0), C(1), C(2), up(0), up(1), up(2)));
      }
      if (follow) s_cam.Follow(last_twc);
      d_cam.Activate(s_cam);
      draw_map(show_pts, show_lines, show_gt, aligned ? &S : nullptr);
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
  {
    std::lock_guard<std::mutex> lk(mu_);
    closed_ = true;
  }
  start_cv_.notify_all();
  pangolin::DestroyWindow("vo_plucker");
  
}

}  // namespace vp
