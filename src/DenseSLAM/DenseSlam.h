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


struct TodoListEntry {
      TodoListEntry(int _MapDataID, int startTimeStamp, int endTimeStamp)
		: dataId(_MapDataID), 
		startKeyframeTimeStamp(startTimeStamp), 
		endKeyframeTimeStamp(endTimeStamp){}
      TodoListEntry(void) {}
      //dataId为在localmap中子地图的index
      int dataId;
      int startKeyframeTimeStamp;
      int endKeyframeTimeStamp;
};

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
      pose_history_({ Eigen::Matrix4f::Identity() }),
      projection_left_rgb_(proj_left_rgb),
      projection_right_rgb_(proj_right_rgb),
      stereo_baseline_m_(stereo_baseline_m),
      experimental_fusion_every_(fusion_every) 
  {
      mFeatures_ = ExtractKeyPointNum();
      mGoalFeatures_ = 0.5 * mFeatures_;
  }
public:
  // PD控制器参数
  int mTrackIntensity = 0;
  float pdThreshold_ = 0.0;

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
        
      static_scene_->GetFloatImage(out_image_float_, preview, model_view, currentLocalMap);
      return out_image_float_->GetData(MEMORYDEVICE_CPU);
  }

  /// \brief Returns an RGBA preview of the reconstructed static map.
  /// \brief 返回重构的静态地图的RGBA预览
  const unsigned char* GetMapRaycastPreview(
      const pangolin::OpenGlMatrix &model_view,
      PreviewType preview,
      bool enable_compositing) {
       
      static_scene_->GetImage(out_image_, preview, model_view, currentLocalMap);
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

  /// @brief 注意这里的GetPose()返回的是当前子地图到当前帧的相对变换
  Eigen::Matrix4f GetPose() const {
    /// XXX: inconsistency between the reference frames of this and the pose history?
    if(currentLocalMap != NULL){
       return static_scene_->GetLocalMapPose(currentLocalMap);
    }
    else{
       return Eigen::Matrix4f::Identity(4,4);
    }
  }

  //const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>&
  const std::vector<Eigen::Matrix4f>&
  GetPoseHistory() const {
    return pose_history_;
  }

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

  void saveLocalMapToHostMemory(int LocalMapId){
    ITMLib::Engine::ITMLocalMap *LocalMapswapOut= static_scene_->GetMapManager()->getLocalMap(LocalMapId);
    static_scene_->GetSwappingEngine()->SaveToGlobalMemory(LocalMapswapOut->scene);
  }

  const Eigen::Matrix34f GetLeftRgbProjectionMatrix() const {
    return projection_left_rgb_;
  };

  const Eigen::Matrix34f GetRightRgbProjectionMatrix() const {
    return projection_right_rgb_;
  };

  float GetStereoBaseline() const {
    return stereo_baseline_m_;
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
  
  ITMLib::Engine::ITMLocalMap* GetCurrentLocalMap() const {
    return currentLocalMap;
  }
  
  int GetNumLocalMap() const {
    return static_scene_->GetMapNumber();
  }
  
  int GetNumActiveLocalMap() const {
    runtime_error("connot use activeLocalMap temperal !");
    return static_scene_->GetLocalActiveMapNumber();
  }
  
  int GetKeyFrameNum() const {
    return orbslam_static_scene_->GetOrbSlamMapDrawer()->mpMap->GetAllKeyFrames().size();
  }
  
  std::vector<TodoListEntry> GettodoList () const{
    return todoList;
  }
  
  size_t GetStaticMapMemoryBytes() const{
    if (currentLocalMap != NULL){
       return static_scene_->GetLocalMapUsedMemoryBytes(currentLocalMap);
    }
    return 0;
  }
  
  int ExtractKeyPointNum(){
    string fPath = orbslam_static_scene_->getOrbParamFile();
    cv::FileStorage fSetting(fPath, cv::FileStorage::READ);
    return fSetting["ORBextractor.nFeatures"];
  }
  
  float PDController(int currFeatures, int lastFeatures){
    float output = mkp*(float)(mGoalFeatures_-currFeatures) + mkd * (float)(lastFeatures-currFeatures)/mDeltaTime_;
    if (output < 0){
      output = 0.0;
    }
    pdThreshold_ = output;
    return output;
  }
  
  bool shouldStartNewLocalMap(int CurrentLocalMapIdx) const; 
  
  int createNewLocalMap(ITMLib::Objects::ITMPose& GlobalPose);
  
  SUPPORT_EIGEN_FIELDS;

private:
  InfiniTamDriver *static_scene_;
  ORB_SLAM2::drivers::OrbSLAMDriver *orbslam_static_scene_;
  SparseSFProvider *sparse_sf_provider_;
//   dynslam::eval::Evaluation *evaluation_;
  
  std::vector<TodoListEntry> todoList;

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;
  
  int current_frame_no_;
  int input_width_;
  int input_height_;
  
  // NOTE PD控制器的参数
  int mFeatures_;
  int mGoalFeatures_;
  int mPreTrackIntensity = 0;
  float mkp = 0.001;
  float mkd = 0.0001;
  ///由于每一帧的时间（包括深度图计算、VO计算、地图融合）接近100ms左右，故delta_t设为0.1ms
  float mDeltaTime_ = 0.1;
  
  // NOTE orbslam的参数
  cv::Mat orbSLAM2_Pose; 
  int orbSLAMTrackingState = 0;
  double lastKeyFrameTimeStamp = 0;
  
  /// NOTE 判断是否开启新地图的阈值
  const int N_originalblocks = 1000;
  ///const float F_originalBlocksThreshold = 0.15f; //0.4f
  
  /// 将F_originalBlocksThreadhold设为-1.0,意味这暂时不开启新的地图
  const float F_originalBlocksThreshold = -1.0f;
  bool shouldCreateNewLocalMap = false;
  bool shouldClearPoseHistory = false;
  
  ITMLib::Engine::ITMLocalMap* currentLocalMap = NULL;

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
