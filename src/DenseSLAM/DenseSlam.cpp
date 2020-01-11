#include <chrono>
#include <thread>
#include <future>

#include "DenseSlam.h"

DEFINE_bool(dynamic_weights, false, "Whether to use depth-based weighting when performing fusion.");
DECLARE_bool(semantic_evaluation);
DECLARE_int32(evaluation_delay);
DEFINE_bool(external_odo, true, "Whether to use external VO");
DEFINE_bool(useOrbSLAMVO, true, "Whether to use OrbSLAM VO");
DEFINE_bool(useSparseFlowVO, true, "Whether to use SparseFlow VO");
DEFINE_bool(useOrbSLMKeyFrame, true, "Whether to use Keyframe strategy in ORB_SLAM2");
DEFINE_bool(useFusion, true, "Whether to use Fusion Strategy");


namespace SparsetoDense {

void DenseSlam::ProcessFrame(Input *input) {
  // Read the images from the first part of the pipeline
  if (! input->HasMoreImages()) {
     cout << "No more frames left in image source." << endl;
     return;
  }
  utils::Tic("Read input and compute depth");
  if(!input->ReadNextFrame()) {
     throw runtime_error("Could not read input from the data source.");
  }
  
  bool first_frame = (current_keyframe_no_ == 1);
  /// @brief 更新当前buf存储的color image和 depth 
  
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  utils::Toc();

  /// @brief 对orbslam进行跟踪，同时进行线程的分离
  utils::Tic("Compute VO");
  future<void> orbslamVO = async(launch::async, [this, &input]{
  cv::Mat orbSLAMInputDepth(input_raw_depth_image_->rows, input_raw_depth_image_->cols, CV_32FC1);

  for(int row =0; row<input_rgb_image_->rows; row++){
    for(int col =0; col<input_rgb_image_->cols; col++){
      orbSLAMInputDepth.at<float>(row,col) = ((float)input_raw_depth_image_->at<int16_t>(row,col))/1000.0;
    }
  }
  
//   imshow("orbSLAMInputDepth: ", orbSLAMInputDepth);
//   cv::waitKey(0);
  
  if(input->GetSensorType() == Input::RGBD){
     orbslam_static_scene_trackRGBD(*input_rgb_image_, 
				    orbSLAMInputDepth, 
				    (double)current_frame_no_);
  }
  else if(input->GetSensorType() == Input::STEREO){
     cv::Mat3b* right_color_image = new cv::Mat3b(input_rgb_image_->rows, input_rgb_image_->cols);
     input->GetRightColor(*right_color_image);
     orbslam_static_scene_trackStereo(*input_rgb_image_, *right_color_image, (double)current_frame_no_);
     delete right_color_image;
  }
  else if(input->GetSensorType() == Input::MONOCULAR){
     orbslam_static_scene_trackMonular(*input_rgb_image_, (double)current_frame_no_);
  }
  });  
  
  /// NOTE 如果当前帧不是关键帧，那么则不进行融合或者更新，直接就return
  orbslamVO.get();
  orbSLAMTrackingState = orbslam_static_scene_->GetOrbSlamTrackingState();
  //orbSLAMTrackingState == -1意味着系统未准备好，==1意味着没有初始化成功,对于单目来说需要判断是否初始化
  if(orbSLAMTrackingState==-1 ||  orbSLAMTrackingState == 1){
    current_frame_no_++;
    utils::Toc();
    return;
  }
  
  lastKeyFrameTimeStamp = GetOrbSlamTrackerGlobal()->mpLastKeyFrameTimeStamp();
  mTrackIntensity = GetOrbSlamTrackerGlobal()->getMatchInlier();
  
  PDThreshold_ = PDController(mTrackIntensity, mPreTrackIntensity);
  //在这里实现PD控制器的实现，以及特征点的设置
  mPreTrackIntensity = mTrackIntensity;
  
  ///当threadhold大于mTrackIntensity的时候，就说明此时需要进行位姿的融合了
  if(!first_frame && PDThreshold_ > mTrackIntensity && FLAGS_useFusion){
      utils::Tic("Fusion Pose of VO");
      currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
      //主要为跟踪做准备
      static_scene_->PrepareNextStepLocalMap(currentLocalMap);
      //更新当前的RGB及深度图
      static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
      //做跟踪
      static_scene_->TrackLocalMap(currentLocalMap);
      //由raycast得到的深度图与当前深度图做ICP跟踪得到的位姿Tw->c
      ITMLib::Objects::ITMPose tempDensePose;
      tempDensePose.SetInvM(drivers::EigenToItm(static_scene_->GetLocalMapPose(currentLocalMap)));
      //由orbSLAM2计算出来的位姿Tw->c
      orbSLAM2_Pose = orbslam_static_scene_->GetPose();
      ITMLib::Objects::ITMPose tempSlamPose;
      tempSlamPose.SetInvM(drivers::EigenToItm(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)));
      ITMLib::Objects::ITMPose tempfusionPose;
      int diff = PDThreshold_ - mTrackIntensity;
      float ratio = (float)diff/(float)PDThreshold_;
      /// Pose的存储顺序应该为：
      /// tx,ty,tz,rx,ry,rz
      float pose[6]= {0.0};
      for(int i=0; i<6 ; i++){
	  pose[i] = ratio * tempDensePose.GetParams()[i] + (1-ratio) * tempSlamPose.GetParams()[i];
      }
      tempfusionPose.SetFrom(pose);
      tempfusionPose.Coerce();
      //其中tempfusionPose.GetM()为Tc->w
      cout << "SetCurrFrameToWorldPose: " << tempSlamPose.GetInvM() << endl;
      cout << "Fusion Pose: "<< tempfusionPose.GetInvM() << endl;
      orbslam_static_scene_->SetCurrFrameToWorldPose(ORB_SLAM2::drivers::EigenToMat(drivers::ItmToEigen(tempfusionPose.GetM())));
      utils::Toc();
  }
  
  /// 需要是关键帧的时候才进行地图的融合
  if((int)lastKeyFrameTimeStamp != current_frame_no_){
        current_frame_no_++;
	utils::Toc();
        return;
  }
  
  /// NOTE 若当前帧为关键帧，则得到当前帧在世界坐标系下的位姿以及跟踪状态
  orbSLAM2_Pose = orbslam_static_scene_->GetPose();
  cout << "orbSLAM2_Pose: " << orbSLAM2_Pose << endl;
  
  /// NOTE 如果是第一帧，则创建子地图，设置创建的子地图基于世界坐标系的位姿，这部分代码主要用于多子图的构建，目前暂时不是很重要
  if(first_frame || shouldCreateNewLocalMap){
    int currentLocalMapIdx = static_scene_->GetMapManager()->createNewLocalMap();
    ITMLib::Objects::ITMPose tempPose;
    //InvM指世界坐标系到相机的变换
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
    shouldCreateNewLocalMap = false;
  }
  
  //这个判断条件主要是避免重复调用多的currentLocalMap
  if(currentLocalMap == NULL){
      currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
  }
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
    else if(input->GetSensorType() == Input::STEREO && FLAGS_useSparseFlowVO){
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
     runtime_error("Currently unsupported VO mode!");
     orbslamVO.get();
     orbSLAM2_Pose = orbslam_static_scene_->GetPose();
     orbSLAMTrackingState = orbslam_static_scene_->GetOrbSlamTrackingState();
     /// NOTE "2"意味着 OrbSLAM 跟踪成功
     if(!orbSLAM2_Pose.empty() && orbSLAMTrackingState == 2){
	   static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose).inverse());
	   pose_history_.push_back(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose));
    }
  }
  utils::Toc();
  
  utils::Tic("Static map fusion");
 /// NOTE orbSLAMTrackingState == 2 意味着orbslam跟踪成功
  if (current_frame_no_ % experimental_fusion_every_ == 0 && !orbSLAM2_Pose.empty() && orbSLAMTrackingState == 2) {
       cout << "experimental_fusion_every_: " << endl;
       if(first_frame || PDThreshold_ < mTrackIntensity || !FLAGS_useFusion){
          static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
       }
       static_scene_->IntegrateLocalMap(currentLocalMap);
       //由于PrepareNextStepLocalMap为raycast准备下一帧进行ICP求位姿用的，因此如果使用orbslam作为VO的话，可以不使用这个函数
//     static_scene_->PrepareNextStepLocalMap(currentLocalMap);
//     Decay old, possibly noisy, voxels to improve map quality and reduce its memory footprint.
//     utils::Tic("Map decay");
//     static_scene_->Decay();
//     utils::Toc();
   }
   else{
      if (current_frame_no_ % experimental_fusion_every_ == 0) {
	 if(first_frame || PDThreshold_ < mTrackIntensity || !FLAGS_useFusion){
            static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
         }
         static_scene_->IntegrateLocalMap(currentLocalMap);
      }
   }
   utils::Toc();
   /*
   if(shouldStartNewLocalMap(todoList.back().dataId) && !first_frame ){
       shouldCreateNewLocalMap = true;
       shouldClearPoseHistory = true;
  }
  */
   current_frame_no_++;
   current_keyframe_no_ ++;
}

bool DenseSlam::shouldStartNewLocalMap(int CurrentLocalMapIdx) const {
    int allocated =  static_scene_->GetMapManager()->getLocalMapSize(CurrentLocalMapIdx);
    int counted = static_scene_->GetMapManager()->countVisibleBlocks(CurrentLocalMapIdx, 0, N_originalblocks, true);
    int tmp = N_originalblocks;
    if(allocated < tmp) {
      tmp = allocated;
    }
//     std::cout << "counted: " << (float)counted << std::endl;
//     std::cout << "tmp: " << (float)tmp << std::endl;
//     std::cout << "((float)counted/(float)tmp): " << ((float)counted/(float)tmp) << std::endl;
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
