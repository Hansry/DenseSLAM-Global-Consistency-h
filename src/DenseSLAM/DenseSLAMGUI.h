#ifndef DENSESLAM_DENSESLAMGUI_H
#define DENSESLAM_DENSESLAMGUI_H

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <string>

#include <gflags/gflags.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <pangolin/pangolin.h>
#include <stdlib.h>

#include "DenseSlam.h"
#include "PrecomputedDepthProvider.h"
#include "InstRecLib/VisoSparseSFProvider.h"
#include "DSHandler3D.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace SparsetoDense{
namespace gui{

using namespace std;
using namespace instreclib;

using namespace SparsetoDense;
using namespace SparsetoDense::eval;
using namespace SparsetoDense::utils;

static const int kUiWidth = 300;

/// \brief What reconstruction error to visualize (used for inspecting the evaluation).
enum VisualizeError {
  kNone = 0,
  kInputVsLidar,
  kFusionVsLidar,
  kInputVsFusion,
  kEnd
};

struct ParamSLAMGUI{
  int frame_limit = 0;
  int evaluation_delay = 0;
  bool close_on_complete = false;
  bool chase_cam = false;
  bool record = false;
  bool autoplay = false;
  bool viewRaycastDepth = false;
};

/// \brief The main GUI and entry point for DenseSLAM
class PangolinGui{
public:
    PangolinGui(DenseSlam *dense_slam, Input *input, ParamSLAMGUI& paramSLAM)
      : dense_slam_(dense_slam),
        dense_slam_input_(input),
        paramGUI(paramSLAM),
        width_(dense_slam->GetInputWidth()),
        height_(dense_slam->GetInputHeight()),
        depth_preview_buffer_(dense_slam->GetInputHeight(), dense_slam->GetInputWidth()),
        OrbSlamMapDrawer_(dense_slam->GetOrbSlamMapDrawerGlobal()),
        OrbSlamFrameDrawer_(dense_slam->GetOrbSlamFrameDrawerGlobal()),
        OrbSlamTracker_(dense_slam->GetOrbSlamTrackerGlobal())
  {
     lidar_vis_colors_ = new unsigned char[2500000];
     lidar_vis_vertices_ = new float[2500000];
     OrbSlamTwc_.SetIdentity();
     CreatePangolinDisplays();
  }
  
  PangolinGui(const PangolinGui&) = delete;//表示删除默认构造函数，&表示左值引用
  PangolinGui(PangolinGui&&) = delete;//表示删除右值拷贝构造函数，&&表示右值引用
  PangolinGui& operator=(const PangolinGui&) = delete;//表示删除默认拷贝构造函数
  PangolinGui& operator=(PangolinGui&&) = delete;//表示删除默认移动拷贝构造函数
  
  
  virtual ~PangolinGui() {
    // No need to delete any view pointers; Pangolin deletes those itself on shutdown.
    delete pane_texture_;
    delete pane_texture_mono_uchar_;

    delete lidar_vis_colors_;
    delete lidar_vis_vertices_;
  }
  
  /// \brief 绘制InfiniTAM光线投影后的视图对应的相机位姿，用锥体形状表示
  /// \param current_time_ms 对最新的位姿颜色进行加深，例如加深为绿色
  void DrawPose(long current_time_ms);
  
  //用锥体形状表示位姿
  void DrawPoseFrustum(const Eigen::Matrix4f &pose, const Eigen::Vector3f &color,
                       float frustum_scale, float frustum_root_cube_scale) const;
  
  /// \brief Executes the main Pangolin input and rendering loop.
  void Run();
  
  /// \brief Renders informative labels for the currently active track.
  /// Meant to be rendered over the segmentation preview window pane.
  /// 显示当前跟踪检测出来的物体，并进行标记
  void DrawInstanceLables();
  
  /// \brief Renders a simple preview of the scene flow information onto the currently active pane.
  /// 显示检测出的光流
  //void PreviewSparseSF(const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow, const pangolin::View &view);
    void PreviewSparseSF(const vector<RawFlow, Eigen::aligned_allocator<RawFlow>> &flow, const pangolin::View &view);
  
  /// \brief Produces a visual pixelwise diff image of the supplied depth maps, into out_image.
  /// 生成所提供深度图和预测的深度图的pixelwise误差
  void DiffDepthmaps(
      const cv::Mat1s &input_depthmap,
      const float* rendered_depth,
      int width,
      int height,
      int delta_max,
      uchar * out_image,
      float baseline_m,
      float focal_length_px
  );
  
  /// \brief Renders the velodyne points for visual inspection.
  /// \param lidar_points
  /// \param P Left camera matrix.
  /// \param Tr Transforms velodyne points into the left camera's frame.
  /// \note For fused visualization we need to use the depth render as a zbuffer when rendering
  /// LIDAR points, either in OpenGL, or manually by projecting LIDAR points and manually checking
  /// their resulting depth. But we don't need this visualization yet; So far, it's enough to render
  /// the LIDAR results for sanity, and then for every point in the cam frame look up the model
  /// depth and compare the two.
  void PreviewLidar(
      const Eigen::MatrixX4f &lidar_points,
      const Eigen::Matrix34f &P,
      const Eigen::Matrix4f &Tr,
      const pangolin::View &view
  );
  
protected:
  
  /// \brief Creates the GUI layout and widgets.
  /// \note The layout is biased towards very wide images (~2:1 aspect ratio or more), which is very
  /// common in autonomous driving datasets.
  void CreatePangolinDisplays();

  /// \brief Advances to the next input frame, and integrates it into the map.
  void ProcessFrame();

  static void DrawOutlinedText(cv::Mat &target, const string &text, int x, int y, float scale = 1.5f) {
   int thickness = static_cast<int>(round(1.1 * scale));
   int outline_factor = 3;
   cv::putText(target, text, cv::Point_<int>(x, y),
               cv::FONT_HERSHEY_DUPLEX, scale, cv::Scalar(0, 0, 0), outline_factor * thickness, CV_AA);
   cv::putText(target, text, cv::Point_<int>(x, y),
               cv::FONT_HERSHEY_DUPLEX, scale, cv::Scalar(230, 230, 230), thickness, CV_AA);
 }
 
 private:
  DenseSlam *dense_slam_;
  Input *dense_slam_input_;
    
  ORB_SLAM2::MapDrawer* OrbSlamMapDrawer_;
  ORB_SLAM2::FrameDrawer* OrbSlamFrameDrawer_;
  ORB_SLAM2::Tracking* OrbSlamTracker_;  
  pangolin::OpenGlMatrix OrbSlamTwc_;
  SparsetoDense::gui::ParamSLAMGUI paramGUI;
  
  /// Input frame dimensions. They dictate the overall window size.
  /// 通过输入的图像大小来定义界面的大小
  int width_, height_;

  pangolin::View *main_view_;
  pangolin::View *orbslam_view_;
  pangolin::View *detail_views_;
  pangolin::View rgb_view_;
  pangolin::View depth_view_;
  pangolin::View raycast_depth_view_;
  /**
  pangolin::View *orb_trajectory_view_;
  pangolin::View *dense_map_fpv_view_; //first person view
  **/
  
  pangolin::OpenGlMatrix proj_;
  pangolin::OpenGlRenderState *pane_cam_;
  pangolin::OpenGlRenderState *orb_pane_cam_;
  
  /**
  pangolin::OpenGlRenderState *orb_Trajectory_pane_cam_;
  pangolin::OpenGlRenderState *dense_map_pane_cam_;
  **/
  
  // Graph plotter and its data logger object
//   pangolin::Plotter *plotter_track;
  pangolin::Plotter *plotter_memory;
  pangolin::DataLog data_log_track;
  pangolin::DataLog data_log_memory;

  //图片纹理定义
  pangolin::GlTexture *pane_texture_;
  pangolin::GlTexture *pane_texture_mono_uchar_;
  pangolin::GlTexture *pane_texture_dense_;

  pangolin::Var<string> *NumLocalMap;
//   pangolin::Var<string> *NumActiveLocalMap;
  pangolin::Var<string> *NumFrame;
  pangolin::Var<string> *NumKeyFrame;
  pangolin::Var<string> *CurrentLocalMapStartKeyframeNo;
  pangolin::Var<string> *CurrentLocalMapEndKeyframeNo;

  // Atomic because it gets set from a UI callback. Technically, Pangolin shouldn't invoke callbacks
  // from a different thread, but using atomics for this is generally a good practice anyway.
  atomic<int> active_object_count_;

  /// \brief When this is on, the input gets processed as fast as possible, without requiring any
  /// user input.
  pangolin::Var<bool> *autoplay_;
  /// \brief Whether to display the RGB and depth previews directly from the input, or from the
  /// static scene, i.e., with the dynamic objects removed.
  pangolin::Var<bool> *display_raw_previews_;    // These objects remain under Pangolin's management, so they don't need to be deleted by the

  /// \brief Whether to preview the sparse scene flow on the input and current instance RGP panes.
  pangolin::Var<bool> *preview_sf_;
  
  pangolin::Var<bool> *sparseMap_show_points_;
  pangolin::Var<bool> *sparseMap_show_graph_;
  pangolin::Var<bool> *sparseMap_show_keyFrame_;

  // TODO(andrei): Reset button.

  // Indicates which object is currently being visualized in the GUI.
  int visualized_object_idx_ = 0;

  int current_preview_type_ = PreviewType::kColor;
  int current_preview_depth_type = PreviewType::kRaycastDepth;
  int current_preview_rgb_type = PreviewType::kRaycastImage;

  int current_lidar_vis_ = VisualizeError::kNone;

  cv::Mat1s depth_preview_buffer_;

  unsigned char *lidar_vis_colors_;
  float *lidar_vis_vertices_;

  /// \brief Prepares the contents of an OpenCV Mat object for rendering with Pangolin (OpenGL).
  /// Does not actually render the texture.
  /// 利用Opencv Mat对象中内容来渲染Pangolin
  static void UploadCvTexture(
      const cv::Mat &mat,
      pangolin::GlTexture &texture,
      bool color,
      GLenum data_type
  ) {
    int old_alignment, old_row_length;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_alignment);
    glGetIntegerv(GL_UNPACK_ROW_LENGTH, &old_row_length);

    int new_alignment = (mat.step & 3) ? 1 : 4;
    int new_row_length = static_cast<int>(mat.step / mat.elemSize());
    glPixelStorei(GL_UNPACK_ALIGNMENT, new_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, new_row_length);

    // [RIP] If left unspecified, Pangolin assumes your texture type is single-channel luminance,
    // so you get dark, uncolored images.
    GLenum data_format = (color) ? GL_BGR : GL_LUMINANCE;
    texture.Upload(mat.data, data_format, data_type);

    glPixelStorei(GL_UNPACK_ALIGNMENT, old_alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, old_row_length);
  }
};    
}//gui
}//SparsetoDense


#endif