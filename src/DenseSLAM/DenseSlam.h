#ifndef DENSESLAM_DENSESLAM_H
#define DENSESLAM_DENSESLAM_H

#include <Eigen/StdVector>

#include <pangolin/display/opengl_render_state.h>

#include "InfiniTamDriver.h"
#include "InstRecLib/SparseSFProvider.h"
#include "Input.h"
#include "OrbSLAMDriver.h"
DECLARE_bool(dynamic_weights);

namespace SparsetoDense {
namespace eval {
class Evaluation;
}
}

namespace SparsetoDense {

using namespace instreclib;
// using namespace instreclib::reconstruction;
// using namespace instreclib::segmentation;
using namespace SparsetoDense::drivers;

// TODO(andrei): Get rid of ITM-specific image objects for visualization.
/// \brief The central class of the DynSLAM system.
/// It processes input stereo frames and generates separate maps for all encountered object
/// instances, as well one for the static background.
/// 处理双目输入帧，对于所有遇到的物体使其地图从静态背景地图从分离
class DenseSlam {
 public:
  DenseSlam(InfiniTamDriver *itm_static_scene_engine,
	  ORB_SLAM2::drivers::OrbSLAMDriver *orb_static_engine,
          SparseSFProvider *sparse_sf_provider,
          const Vector2i &input_shape,
          const Eigen::Matrix34f& proj_left_rgb,
          const Eigen::Matrix34f& proj_right_rgb,
          float stereo_baseline_m,
          bool enable_direct_refinement,
          bool dynamic_mode,
          int fusion_every)
    : static_scene_(itm_static_scene_engine),
      orbslam_static_scene_(orb_static_engine),
      sparse_sf_provider_(sparse_sf_provider),
      // Allocate the ITM buffers on the CPU and on the GPU (true, true).
      out_image_(new ITMUChar4Image(input_shape, true, true)),
      out_image_float_(new ITMFloatImage(input_shape, true, true)),
      input_rgb_image_(new cv::Mat3b(input_shape.x, input_shape.y)),
      input_raw_depth_image_(new cv::Mat1s(input_shape.x, input_shape.y)),
      current_frame_no_(0),
      input_width_(input_shape.x),
      input_height_(input_shape.y),
      dynamic_mode_(dynamic_mode),
      pose_history_({ Eigen::Matrix4f::Identity() }),
      projection_left_rgb_(proj_left_rgb),
      projection_right_rgb_(proj_right_rgb),
      stereo_baseline_m_(stereo_baseline_m),
      experimental_fusion_every_(fusion_every)
  {}

  /// \brief Reads in and processes the next frame from the data source.
  /// This is where most of the interesting stuff happens.
  void ProcessFrame(Input *input);

  /// \brief Returns an RGB preview of the latest color frame.
  /// \brief 返回最新帧的RGB预览
  const cv::Mat3b* GetRgbPreview() {
    return input_rgb_image_;
  }

  /// \brief Returns a preview of the latest depth frame.
  /// \brief 返回最新深度图的预览
  const cv::Mat1s* GetDepthPreview() {
    return input_raw_depth_image_;
  }

  /// \brief Returns a preview of the static parts from the latest depth frame.
  /// \brief 返回最新光线投影得到的深度图静态部分的预览
  const float* GetRaycastDepthPreview(
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview,
      bool enable_compositing) {
        
      static_scene_->GetFloatImage(out_image_float_, preview, model_view);
      return out_image_float_->GetData(MEMORYDEVICE_CPU);
  }

  /// \brief Returns an RGBA preview of the reconstructed static map.
  /// \brief 返回重构的静态地图的RGBA预览
  const unsigned char* GetMapRaycastPreview(
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview,
      bool enable_compositing) {
       
      static_scene_->GetImage(out_image_, preview, model_view);
      return out_image_->GetData(MEMORYDEVICE_CPU)->getValues();
    }

  int GetInputWidth() const {
    return input_width_;
  }

  int GetInputHeight() const {
    return input_height_;
  }

  int GetCurrentFrameNo() const {
    return current_frame_no_;
  }

  void SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const;

  void SaveDynamicObject(const std::string &dataset_name, const std::string &depth_name, int object_id) const;

  // Variants would solve this nicely, but they are C++17-only... TODO(andrei): Use Option<>.
  // Will error out if no flow information is available.
  const SparseSceneFlow& GetLatestFlow() {
    return sparse_sf_provider_->GetFlow();
  }

  /// \brief Returns the most recent egomotion computed by the primary tracker.
  /// Composing these transforms from the first frame is equivalent to the `GetPose()` method, which 
  /// returns the absolute current pose.
  /// \brief 得到的是前一帧到当前帧的变换矩阵
  Eigen::Matrix4f GetLastEgomotion() const {
    return static_scene_->GetLastEgomotion();
  }

  /// \brief Returns the current pose of the camera in the coordinate frame used by the tracker.
  /// For the KITTI dataset (and the KITTI-odometry one) this represents the center of the left camera.
  Eigen::Matrix4f GetPose() const {
    /// XXX: inconsistency between the reference frames of this and the pose history?
    return static_scene_->GetPose();
  }

  //const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  const std::vector<Eigen::Matrix4f>&
  GetPoseHistory() const {
    return pose_history_;
  }

//   size_t GetStaticMapMemoryBytes() const {
//     return static_scene_->GetUsedMemoryBytes();
//   }
// 
//   size_t GetStaticMapSavedDecayMemoryBytes() const {
//     return static_scene_->GetSavedDecayMemoryBytes();
//   }
// 
//   const VoxelDecayParams& GetStaticMapDecayParams() const {
//     return static_scene_->GetVoxelDecayParams();
//   }
// 
//   void WaitForJobs() {
//     // TODO(andrei): fix this; it does not actually work and never blocks...
//     static_scene_->WaitForMeshDump();
//   }

  const Eigen::Matrix34f GetLeftRgbProjectionMatrix() const {
    return projection_left_rgb_;
  };

  const Eigen::Matrix34f GetRightRgbProjectionMatrix() const {
    return projection_right_rgb_;
  };

  float GetStereoBaseline() const {
    return stereo_baseline_m_;
  }

  bool IsDynamicMode() const {
    return dynamic_mode_;
  }

  /// \brief Run voxel decay all the way up to the latest frame.
  /// Useful for cleaning up at the end of the sequence. Should not be used mid-sequence.
//   void StaticMapDecayCatchup() {
//     static_scene_->DecayCatchup();
//   }
  
  void orbslam_static_scene_trackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double& timestamp){
    orbslam_static_scene_->orbTrackRGBDSLAM(im,depthmap,timestamp);
  }
  
  void orbslam_static_scene_trackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, int timestamp){
    orbslam_static_scene_->orbTrackStereo(imLeft, imRight, timestamp);
  }
  
  ORB_SLAM2::MapDrawer* GetOrbSlamMapDrawerGlobal() const{
     return orbslam_static_scene_->GetOrbSlamMapDrawer();
  }
  
  ORB_SLAM2::FrameDrawer* GetOrbSlamFrameDrawerGlobal() const{
     return orbslam_static_scene_->GetOrbSlamFrameDrawer();
  }
  
  vector<ORB_SLAM2::KeyFrame*> GetOrbSlamKeyFrameDatabate() const {
     return orbslam_static_scene_->GetOrbSlamMapDrawer()->mpMap->GetAllKeyFrames();
  }
  
  ORB_SLAM2::Tracking* GetOrbSlamTrackerGlobal() const{
     return orbslam_static_scene_->GetTracker();
  }
  
  int GetNumLocalMap() const {
    return static_scene_->GetMapNumber();
  }
  
  int GetNumActiveLocalMap() const {
    return static_scene_->GetLocalActiveMapNumber();
  }
  
  int GetKeyFrameNum() const {
    return orbslam_static_scene_->GetOrbSlamMapDrawer()->mpMap->GetAllKeyFrames().size();
  }
  SUPPORT_EIGEN_FIELDS;

private:
  InfiniTamDriver *static_scene_;
  ORB_SLAM2::drivers::OrbSLAMDriver *orbslam_static_scene_;
//   SegmentationProvider *segmentation_provider_;
//   InstanceReconstructor *instance_reconstructor_;
  SparseSFProvider *sparse_sf_provider_;
//   dynslam::eval::Evaluation *evaluation_;

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;
  
  int current_frame_no_;
  int input_width_;
  int input_height_;
  
  cv::Mat orbSLAM2_Pose; 
  int orbSLAMTrackingState = 0;
  double lastKeyFrameTimeStamp = 0;

  /// \brief Enables object-awareness, object reconstruction, etc. Without this, the system is
  ///        basically just outdoor InfiniTAM.
  /// \brief 重要的参数，启动物体检测、物体重建等功能，如果没有该功能，那么就只是基于InfiniTAM了
  bool dynamic_mode_;

  /// \brief If dynamic mode is on, whether to force instance reconstruction even for non-dynamic
  /// objects, like parked cars. All experiments in the thesis are performed with this 'true'.
  /// \brief 如果启动动态模式，是否对物体实例进行重构，即使是对停着的车
  bool always_reconstruct_objects_ = true;

 // std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> pose_history_;
  std::vector<Eigen::Matrix4f> pose_history_;
  /// \brief Matrix for projecting 3D homogeneous coordinates in the left gray camera's coordinate
  ///        frame to 2D homogeneous coordinates in the left color camera's coordinate frame
  ///        (expressed in pixels).
  /// \brief 归一化平面到像素平面？
  const Eigen::Matrix34f projection_left_rgb_;
  const Eigen::Matrix34f projection_right_rgb_;
  const float stereo_baseline_m_;

  /// \brief Stores the result of the most recent segmentation.
//   std::shared_ptr<instreclib::segmentation::InstanceSegmentationResult> latest_seg_result_;

  /// \brief Perform dense depth computation and dense fusion every K frames.
  /// A value of '1' is the default, and means regular operation fusing every frame.
  /// This is used to evaluate the impact of doing semantic segmentation less often. Note that we
  /// still *do* perform it, as we still need to evaluate the system every frame.
  /// TODO(andrei): Support instance tracking in this framework: we would need SSF between t and t-k,
  ///               so we DEFINITELY need separate VO to run in, say, 50ms at every frame, and then
  ///               heavier, denser feature matching to run in ~200ms in parallel to the semantic
  ///               segmentation, matching between this frames and, say, 3 frames ago. Luckily,
  /// libviso should keep track of images internally, so if we have two instances we can just push
  /// data and get SSF at whatever intervals we would like.
  const int experimental_fusion_every_;

  /// \brief Returns a path to the folder where the dataset's meshes should be dumped, creating it
  ///        using a native system call if it does not exist.
  std::string EnsureDumpFolderExists(const string& dataset_name) const;
};

}

#endif //DYNSLAM_DYNSLAM_H
