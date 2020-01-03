#include <chrono>
#include <thread>
#include <future>

#include "DenseSlam.h"

DEFINE_bool(dynamic_weights, false, "Whether to use depth-based weighting when performing fusion.");
DECLARE_bool(semantic_evaluation);
DECLARE_int32(evaluation_delay);
DEFINE_bool(external_odo, true, "Whether to use external VO");
DEFINE_bool(useOrbSLAMVO, true, "Whether to use OrbSLAM VO");
DEFINE_bool(useOrbSLMKeyFrame, true, "Whether to use Keyframe strategy in ORB_SLAM2");

namespace SparsetoDense {

void DenseSlam::ProcessFrame(Input *input) {
  // Read the images from the first part of the pipeline
  if (! input->HasMoreImages()) {
    cout << "No more frames left in image source." << endl;
    return;
  }
  
  bool first_frame = (current_frame_no_ == 0);
      
  /// @brief 更新当前buf存储的color image和 depth 
  utils::Tic("Read input and compute depth");
  if(!input->ReadNextFrame()) {
    throw runtime_error("Could not read input from the data source.");
  }
  utils::Toc();
  
  utils::Tic("Input preprocessing");
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  utils::Toc();

  /// @brief 对orbslam进行跟踪，同时进行线程的分离
  future<void> orbslamVO = async(launch::async, [this, &input]{
  cv::Mat orbSLAMInputDepth(input_raw_depth_image_->rows, input_raw_depth_image_->cols, CV_32FC1);
  for(int row =0; row<input_rgb_image_->rows; row++){
    for(int col =0; col<input_rgb_image_->cols; col++){
      orbSLAMInputDepth.at<float>(row,col) = (float)input_raw_depth_image_->at<ushort>(row,col)/1000.0;
    }
  }
    orbslam_static_scene_trackRGBD(*input_rgb_image_, 
				 orbSLAMInputDepth, 
				 (double)current_frame_no_);
  });  
  
  /// NOTE 如果当前帧不是关键帧，那么则不进行融合或者更新，直接就return
  orbslamVO.get();
  lastKeyFrameTimeStamp = GetOrbSlamTrackerGlobal()->mpLastKeyFrameTimeStamp();
  if((int)lastKeyFrameTimeStamp != current_frame_no_){
        current_frame_no_++;
        return;
  }
  
  /// NOTE 若当前帧为关键帧，则得到当前帧在世界坐标系下的位姿以及跟踪状态
  orbSLAM2_Pose = orbslam_static_scene_->GetPose();
  orbSLAMTrackingState = orbslam_static_scene_->GetOrbSlamTrackingState();
  
  /// NOTE 如果是第一帧，则创建子地图，设置创建的子地图基于世界坐标系的位姿
  if(first_frame || shouldCreateNewLocalMap){
    int currentLocalMapIdx = static_scene_->GetMapManager()->createNewLocalMap();
    ITMLib::Objects::ITMPose tempPose;
    tempPose.SetInvM(drivers::EigenToItm(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)));
    static_scene_->GetMapManager()->setEstimatedGlobalPose(currentLocalMapIdx, tempPose);
    todoList.push_back(TodoListEntry(currentLocalMapIdx,
				     lastKeyFrameTimeStamp,
				     lastKeyFrameTimeStamp));
    
    /// NOTE 这里由于显卡内存不足，只能在创建新子地图的时候将前一个子地图给remove掉
//    if(todoList.size()>1){
//       static_scene_->GetMapManager()->removeLocalMap(todoList.front().dataId);
//       int swapOutLocalMapID = todoList[todoList.size()-2].dataId;
//       future<void> swap_out = async(launch::async, [this, &swapOutLocalMapID]{
// 	   saveLocalMapToHostMemory(swapOutLocalMapID);
//       });
//    }
//    shouldCreateNewLocalMap = false;
  }
  
  currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
  todoList.back().endKeyframeTimeStamp = lastKeyFrameTimeStamp;
  
  /// @brief 利用左右图像计算稀疏场景光流
  if(FLAGS_external_odo){
    /// 使用ORBSLAM的里程计
    if(FLAGS_useOrbSLAMVO){
      /// NOTE "2"意味着 OrbSLAM 跟踪成功
      if(!orbSLAM2_Pose.empty() && orbSLAMTrackingState == 2){
	    static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose));
	    Eigen::Matrix4f currLocalMapPose = drivers::ItmToEigen(currentLocalMap->estimatedGlobalPose.GetM());
	    if(shouldClearPoseHistory){
	      pose_history_.clear();
	    }
	    pose_history_.push_back((currLocalMapPose*ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)).inverse());
	    shouldClearPoseHistory = false;
	 }
      }
    else{
      ///使用双目光流计算出来的里程计
      future<void> ssf_and_vo = async(launch::async, [this, &input, &first_frame] {
      utils::Tic("Sparse Scene Flow");

      // Whether to use input from the original cameras. Unavailable with the tracking dataset.
      // When original gray images are not used, the color ones are converted to grayscale and passed
      // to the visual odometry instead.
      bool original_gray = false;

      cv::Mat1b *left_gray;
      cv::Mat1b *right_gray;
      // 如果是灰度图直接计算光流
      if (original_gray) {
         input->GetCvStereoGray(&left_gray, &right_gray);
      }
      // 如果输入的是RGB，先转成灰度图再计算光流
      else {
        cv::Mat3b *left_col, *right_col;
        input->GetCvStereoColor(&left_col, &right_col);

        left_gray = new cv::Mat1b(left_col->rows, left_col->cols);
        right_gray = new cv::Mat1b(right_col->rows, right_col->cols);

        cv::cvtColor(*left_col, *left_gray, cv::COLOR_RGB2GRAY);
        cv::cvtColor(*right_col, *right_gray, cv::COLOR_RGB2GRAY);
      }
    
      sparse_sf_provider_->ComputeSparseSF(
         make_pair((cv::Mat1b *) nullptr, (cv::Mat1b *) nullptr),
         make_pair(left_gray, right_gray)
      );
      if (!sparse_sf_provider_->FlowAvailable() && !first_frame) {
         cerr << "Warning: could not compute scene flow." << endl;
      }
      utils::Toc("Sparse Scene Flow", false);
  
      utils::Tic("Visual Odometry");
      /// 得到最新的位姿，相对于上一帧的位姿
      /// T_{current, previous} 
      Eigen::Matrix4f delta = sparse_sf_provider_->GetLatestMotion();

      //使用光流进行跟踪
      //new_pose为当前帧到世界坐标系(第一帧)下的位姿变换
      //Tcurrent_w = Tcurrent_previous * previous_w
      Eigen::Matrix4f new_pose = delta * pose_history_[pose_history_.size() - 1];
      static_scene_->SetPoseLocalMap(currentLocalMap, new_pose.inverse());
      pose_history_.push_back(new_pose);//将当前帧的位姿存储到vector中，方便下一次计算使用

      if (! original_gray) {
        delete left_gray;
        delete right_gray;
      }
      utils::Toc("Visual Odometry", false);
      });
      ssf_and_vo.get();
    }
  }
  /// 使用内部使用的里程计，目前还没有写好
  else{
     orbslamVO.get();
     orbSLAM2_Pose = orbslam_static_scene_->GetPose();
     orbSLAMTrackingState = orbslam_static_scene_->GetOrbSlamTrackingState();
     /// NOTE "2"意味着 OrbSLAM 跟踪成功
     if(!orbSLAM2_Pose.empty() && orbSLAMTrackingState == 2){
	   static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose).inverse());
	   pose_history_.push_back(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose));
    }
  }
  if(FLAGS_useOrbSLAMVO){
      /// NOTE orbSLAMTrackingState == 2 意味着orbslam跟踪成功
      if (current_frame_no_ % experimental_fusion_every_ == 0 && !orbSLAM2_Pose.empty() && orbSLAMTrackingState == 2) {
         utils::Tic("Static map fusion");
	 static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
// 	 static_scene_->TrackLocalMap(currentLocalMap);/////delete after
         static_scene_->IntegrateLocalMap(currentLocalMap);
         static_scene_->PrepareNextStepLocalMap(currentLocalMap);
         utils::TocMicro();

//    Decay old, possibly noisy, voxels to improve map quality and reduce its memory footprint.
//    utils::Tic("Map decay");
//    static_scene_->Decay();
//    utils::TocMicro();
       }
    }
   else{
      if (current_frame_no_ % experimental_fusion_every_ == 0) {
         utils::Tic("Static map fusion");
	 static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
         static_scene_->IntegrateLocalMap(currentLocalMap);
         static_scene_->PrepareNextStepLocalMap(currentLocalMap);
         utils::TocMicro();
      }
   }
   
   if(shouldStartNewLocalMap(todoList.back().dataId) && !first_frame ){
       shouldCreateNewLocalMap = true;
       shouldClearPoseHistory = true;
  }
   current_frame_no_++;
}

bool DenseSlam::shouldStartNewLocalMap(int CurrentLocalMapIdx) const {
    int allocated =  static_scene_->GetMapManager()->getLocalMapSize(CurrentLocalMapIdx);
    int counted = static_scene_->GetMapManager()->countVisibleBlocks(CurrentLocalMapIdx, 0, N_originalblocks, true);
    int tmp = N_originalblocks;
    if(allocated < tmp) {
      tmp = allocated;
    }
    std::cout << "counted: " << (float)counted << std::endl;
    std::cout << "tmp: " << (float)tmp << std::endl;

    std::cout << "((float)counted/(float)tmp): " << ((float)counted/(float)tmp) << std::endl;
    return ((float)counted/(float)tmp) < F_originalBlocksThreshold;
}

int DenseSlam::createNewLocalMap(ITMLib::Objects::ITMPose& GlobalPose){
    int newIdx = static_scene_->GetMapManager()->createNewLocalMap();
    static_scene_->GetMapManager()->setEstimatedGlobalPose(newIdx, GlobalPose);
    return newIdx;
}

void DenseSlam::SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const {
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string map_fpath = utils::Format("%s/static-%s-mesh-%06d-frames.obj",
                                   target_folder.c_str(),
                                   depth_name.c_str(),
                                   current_frame_no_);
  cout << "Saving full static map to: " << map_fpath << endl;
  static_scene_->SaveSceneToMesh(map_fpath.c_str());
}

std::string DenseSlam::EnsureDumpFolderExists(const string &dataset_name) const {
  // TODO-LOW(andrei): Make this more cross-platform and more secure.
  string today_folder = utils::GetDate();
  string target_folder = "mesh_out/" + dataset_name + "/" + today_folder;
  if(system(utils::Format("mkdir -p '%s'", target_folder.c_str()).c_str())) {
    throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
  }

  return target_folder;
}

}
