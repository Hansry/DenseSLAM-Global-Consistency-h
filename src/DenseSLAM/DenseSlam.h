#ifndef DENSESLAM_DENSESLAM_H
#define DENSESLAM_DENSESLAM_H

#include <Eigen/StdVector>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>

#include <pangolin/display/opengl_render_state.h>

#include "InfiniTamDriver.h"
#include "InstRecLib/SparseSFProvider.h"
#include "Input.h"
#include "OrbSLAMDriver.h"
#include <map>
#include <string>
#include <cmath>

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

struct fusionFrameInfo{
     fusionFrameInfo(cv::Mat pose, cv::Mat3b rgb, cv::Mat1s depth, short flag, cv::Mat depthweight)
                  :poseinfo(pose), 
                  rgbinfo(rgb), 
                  depthinfo(depth),
                  flaginfo(flag),
		  depthweightinfo(depthweight){}
                  
     fusionFrameInfo(void) {}
     cv::Mat poseinfo;
     cv::Mat3b rgbinfo;
     cv::Mat1s depthinfo;
     short flaginfo;
     cv::Mat depthweightinfo;
};

struct mapKeyframeInfo{
     mapKeyframeInfo(double timestamp, cv::Mat pose): 
                 timestampinfo(timestamp),
                 poseinfok(pose){}
                 
     mapKeyframeInfo(void) {}
     double timestampinfo;
     cv::Mat poseinfok;
};

struct currFrameInfo{
     currFrameInfo(cv::Mat3b rgb, cv::Mat1s depth, cv::Mat depthweight, double timestamp)
                  :rgbinfoc(rgb), 
                  depthinfoc(depth),
                  depthweightinfoc(depthweight),
                  timestampinfoc(timestamp){}
                  
     currFrameInfo(void) {}
     cv::Mat3b rgbinfoc;
     cv::Mat1s depthinfoc;
     cv::Mat depthweightinfoc;
     double timestampinfoc;
};

class DenseSlam {
 public:
  DenseSlam(InfiniTamDriver *itm_static_scene_engine,
	  ORB_SLAM2::drivers::OrbSLAMDriver *orb_static_engine,
          const Vector2i &input_shape,
          const Eigen::Matrix34f& proj_left_rgb,
          float stereo_baseline_mm,
          bool enable_direct_refinement,
          PostPocessParams post_processing,
	  OnlineCorrectionParams online_correction,
	  bool use_orbslam_vo)
    : static_scene_(itm_static_scene_engine),
      orbslam_static_scene_(orb_static_engine),
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
      stereo_baseline_mm_(stereo_baseline_mm),
      online_correction(online_correction),
      post_processing(post_processing),
      use_orbslam_vo(use_orbslam_vo)
  {
      mFeatures_ = ExtractKeyPointNum();
      mGoalFeatures_ = 0.5 * mFeatures_;
  }
public:
  // PD控制器参数
  int mTrackIntensity = 0;

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

  void SaveStaticMap(const std::string &dataset_name, const ITMLib::Engine::ITMLocalMap* currentLocalMap, int localMap_no) const;

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

  void Decay(){
     if(currentLocalMap != NULL){
       static_scene_->Decay(currentLocalMap);
    }
  }
  
  void DecayCatchup(){
    if(currentLocalMap != NULL){
      static_scene_->DecayCatchup(currentLocalMap);
    }
  }
  
  size_t GetStaticMapSavedDecayMemoryBytes() const {
    return static_scene_->GetSavedDecayMemoryBytes();
  }

  const VoxelDecayParams& GetStaticMapDecayParams() const {
    return static_scene_->GetVoxelDecayParams();
  }

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


  float GetStereoBaseline() const {
    return stereo_baseline_mm_;
  }
  
  bool OnlineCorrection();
  
  bool depthPostProcessing(double currBAKFTime);
  
  Eigen::Matrix4f MatrixDoubleToFloat(Eigen::Matrix4d Eig_matrix){
     
     Eigen::Matrix4f tempMatrix;
     tempMatrix << (float)Eig_matrix(0,0), (float)Eig_matrix(0,1), (float)Eig_matrix(0,2), (float)Eig_matrix(0,3),
                   (float)Eig_matrix(1,0), (float)Eig_matrix(1,1), (float)Eig_matrix(1,2), (float)Eig_matrix(1,3),
                   (float)Eig_matrix(2,0), (float)Eig_matrix(2,1), (float)Eig_matrix(2,2), (float)Eig_matrix(2,3),
                   (float)Eig_matrix(3,0), (float)Eig_matrix(3,1), (float)Eig_matrix(3,2), (float)Eig_matrix(3,3);
     return tempMatrix;
  }
  
  Eigen::Matrix4d MatrixFloatToDouble(Eigen::Matrix4f Eig_matrix){
     Eigen::Matrix4d tempMatrix;
     tempMatrix << (double)Eig_matrix(0,0), (double)Eig_matrix(0,1), (double)Eig_matrix(0,2), (double)Eig_matrix(0,3),
                   (double)Eig_matrix(1,0), (double)Eig_matrix(1,1), (double)Eig_matrix(1,2), (double)Eig_matrix(1,3),
                   (double)Eig_matrix(2,0), (double)Eig_matrix(2,1), (double)Eig_matrix(2,2), (double)Eig_matrix(2,3),
                   (double)Eig_matrix(3,0), (double)Eig_matrix(3,1), (double)Eig_matrix(3,2), (double)Eig_matrix(3,3);
     return tempMatrix;
  }

  /// \brief Run voxel decay all the way up to the latest frame.
  /// Useful for cleaning up at the end of the sequence. Should not be used mid-sequence.
//   void StaticMapDecayCatchup() {
//     static_scene_->DecayCatchup();
//   }
  
  void orbslam_static_scene_trackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double& timestamp){
//     cout << "DenseSLAM.h 235" << endl;
    orbslam_static_scene_->orbTrackRGBDSLAM(im,depthmap,timestamp);
  }
  
  void orbslam_static_scene_trackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, int timestamp){
    orbslam_static_scene_->orbTrackStereo(imLeft, imRight, timestamp);
  }
  
  void orbslam_static_scene_trackMonular(const cv::Mat &im,  int timestamp){
    orbslam_static_scene_->orbTrackMonocular(im, timestamp);
  }
  
  list<ORB_SLAM2::KeyFrame*>* orbslam_static_scene_localBAKF(void){
    return orbslam_static_scene_->GetOrbSlamLocalBAKeyframe();
  }
  
  std::condition_variable* orbslam_tracking_cond(void) {
    return orbslam_static_scene_->GetTrackingCondVar();
  }
  
  std::condition_variable* orbslam_tracking_cond_n(void){
    return orbslam_static_scene_->GetTrackingCondVar_n();
  }
  
  bool* orbslam_tracking_gl(void){
    return orbslam_static_scene_->GetTrackingGL();
  }
  
  bool* orbslam_tracking_gl_n(void){
    return orbslam_static_scene_->GetTrackingGL_n();
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
  
  size_t GetCurrMapAllocatedMapMemoryBytes() const {
    if(currentLocalMap != NULL){
       return static_scene_->GetCurrMapAllocateMemoryBytes(currentLocalMap);
    }
    return 0;
  }
  
  int ExtractKeyPointNum(){
    string fPath = orbslam_static_scene_->getOrbParamFile();
    cv::FileStorage fSetting(fPath, cv::FileStorage::READ);
    return fSetting["ORBextractor.nFeatures"];
  }
  
  float PDController(int currFeatures, int lastFeatures){
    int currDiff = lastFeatures - currFeatures;
    /*
    if(abs(lastFeatures-currFeatures)> 2 * mPreDiff ){
      if(lastFeatures > currFeatures){
         currDiff = 2 * mPreDiff;
      }
      else{
	 currDiff = -2 * mPreDiff;
      }
    }
    else{
      currDiff = lastFeatures - currFeatures;
    }
    */
    
    /*
    if(currFeatures > 1.5 * lastFeatures){
      currFeatures = 1.5 * lastFeatures;
    }
    */
    
    float output = mkp*(float)(mGoalFeatures_-currFeatures) + mkd * (float)(currDiff)/mDeltaTime_;
    if (output < 0){
      output = 0.0;
    }
    return output;
  }
  
  float GetPDThreadhold() const{
    return PDThreshold_;
  }
  
  void SaveTUMTrajectory(const string& filename){
    orbslam_static_scene_->SaveTUMTrajectory(filename);
  }
  
  bool is_identity_matrix(cv::Mat matrix);
  
  bool shouldStartNewLocalMap(int CurrentLocalMapIdx) const; 
  
  int createNewLocalMap(ITMLib::Objects::ITMPose& GlobalPose);
//   map<double, std::pair<cv::Mat3b, cv::Mat1s>> mframeDataBase;
  map<double, currFrameInfo> mframeDataBase;
  
  //融合帧的timestamp,位姿，RGB信息和深度图
  map<double, fusionFrameInfo> mfusionFrameDataBase;
  
  std::vector<TodoListEntry> todoList;
  
  InfiniTamDriver* GetStaticScene() {
    return static_scene_;
  }
  
  int64_t GetTotalFusionTime() const{
    return fusionTotalTime;
  }
  
  SUPPORT_EIGEN_FIELDS;

private:
  InfiniTamDriver *static_scene_;
  ORB_SLAM2::drivers::OrbSLAMDriver *orbslam_static_scene_;
  
  PostPocessParams post_processing;
  OnlineCorrectionParams online_correction;
  bool use_orbslam_vo;
  
  double currFrameTimeStamp;
  int64_t fusionTotalTime = 0; 

  ITMUChar4Image *out_image_;
  ITMFloatImage *out_image_float_;
  cv::Mat3b *input_rgb_image_;
  cv::Mat3b *input_right_rgb_image_;
  cv::Mat1s *input_raw_depth_image_;
  
  cv::Mat3b input_rgb_image_n;
  cv::Mat1s input_raw_depth_image_n;
  cv::Mat3b input_right_rgb_image_n;
  
  cv::Mat3b input_rgb_image_copy_;
  cv::Mat1s input_raw_depth_image_copy_;
  cv::Mat input_weight_copy_;
  
  int current_frame_no_ = 0;
  int current_keyframe_no_ = 1;
  int input_width_;
  int input_height_;
  
  // NOTE PD控制器的参数
  int mFeatures_;
  int mGoalFeatures_;
  int mPreTrackIntensity = 0;
  int mPreDiff = 0;
  float mkp = 0.8;
  float mkd = 0.08;
  ///由于每一帧的时间（包括深度图计算、VO计算、地图融合）接近100ms左右，故delta_t设为0.1ms
  float mDeltaTime_ = 0.1;
  float PDThreshold_ = 0.0;
  
//   float mkp = 0.25;
//   float mkd = 0.05;
//   ///由于每一帧的时间（包括深度图计算、VO计算、地图融合）接近100ms左右，故delta_t设为0.1ms
//   float mDeltaTime_ = 0.15;
  
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
  std::queue<pair<ORB_SLAM2::KeyFrame, cv::Mat>> mkeyframeForNewLocalMap;
  std::mutex mMutexKeyframeQueue;
 
  std::vector<Eigen::Matrix4f> pose_history_;
  /// \brief 归一化平面到像素平面？
  const Eigen::Matrix34f projection_left_rgb_;
  const float stereo_baseline_mm_;

  /// \brief Returns a path to the folder where the dataset's meshes should be dumped, creating it
  ///        using a native system call if it does not exist.
  std::string EnsureDumpFolderExists(const string& dataset_name) const;
  
  
  ORB_SLAM2::KeyFrame* mcurrBAKeyframe;
  
  std::mutex mMutexFrameDataBase;
  std::mutex mMutexBAKF;
  std::mutex mMutexCond;
  std::mutex mMutexCond1;
};

}

#endif //DYNSLAM_DYNSLAM_H
