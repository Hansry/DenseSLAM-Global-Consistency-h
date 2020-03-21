#include <chrono>
#include <thread>
#include <future>

#include "DenseSlam.h"

DEFINE_bool(dynamic_weights, false, "Whether to use depth-based weighting when performing fusion.");
DECLARE_bool(semantic_evaluation);
DECLARE_int32(evaluation_delay);
DEFINE_bool(useOrbSLAMVO, true, "Whether to use OrbSLAM VO");
DEFINE_bool(useSparseFlowVO, true, "Whether to use SparseFlow VO");
DEFINE_bool(useOrbSLMKeyFrame, true, "Whether to use Keyframe strategy in ORB_SLAM2");
DEFINE_bool(external_odo, true, "Whether to use external VO");
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
  
  currFrameTimeStamp = input->GetCurrentFrame_double();
//   printf("%s%f", "DenseSlam26, currentFrame TimeStamp: \n", currFrameTimeStamp);
  
  bool first_frame = (current_keyframe_no_ == 1);
  /// @brief 更新当前buf存储的color image和 depth 
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  utils::Toc();
  
  /// @brief 对orbslam进行跟踪，同时进行线程的分离
  utils::Tic("Compute VO");
  
  input_rgb_image_n = (*input_rgb_image_).clone();
  input_raw_depth_image_n = (*input_raw_depth_image_).clone();
  
  cv::Mat orbSLAMInputDepth(input_raw_depth_image_n.rows, input_raw_depth_image_n.cols, CV_32FC1);
//   std::cout << "denseslam 47: " << std::endl;

  for(int row =0; row<input_rgb_image_n.rows; row++){
    for(int col =0; col<input_rgb_image_n.cols; col++){
      orbSLAMInputDepth.at<float>(row,col) = ((float)input_raw_depth_image_n.at<int16_t>(row,col))/1000.0;
    }
  }
  
  if(!first_frame && out_image_float_->GetData(MEMORYDEVICE_CPU)!=nullptr){
    cv::Mat raycastDepth(input_raw_depth_image_n.rows, input_raw_depth_image_n.cols, CV_32FC1);
    for(int row=0; row<input_rgb_image_n.rows; row++){
      for(int col=0; col<input_rgb_image_n.cols; col++){
	raycastDepth.at<float>(row,col) = static_cast<float>(out_image_float_->GetData(MEMORYDEVICE_CPU)[row * input_rgb_image_n.cols + col]);
      }
    }
    
    unique_lock<mutex> lock(mMutexCond);
    *(orbslam_static_scene_->GetTrackinRaycastDepth()) = raycastDepth.clone();
    *(orbslam_static_scene_->GetTrackingDepth()) = orbSLAMInputDepth.clone();
  }
  
  future<void> orbslamVO = async(launch::async, [this, &input, &orbSLAMInputDepth]{
    
//   imshow("orbSLAMInputDepth: ", orbSLAMInputDepth);
//   cv::waitKey(0);
  
  if(input->GetSensorType() == Input::RGBD){
     orbslam_static_scene_trackRGBD(input_rgb_image_n, 
				    orbSLAMInputDepth, 
				    currFrameTimeStamp);
  }
  else if(input->GetSensorType() == Input::STEREO){
     cv::Mat3b* right_color_image = new cv::Mat3b(input_rgb_image_n.rows, input_rgb_image_n.cols);
     input->GetRightColor(*right_color_image);
     orbslam_static_scene_trackStereo(input_rgb_image_n, *right_color_image, currFrameTimeStamp);
     delete right_color_image;
  }
  else if(input->GetSensorType() == Input::MONOCULAR){
     orbslam_static_scene_trackMonular(input_rgb_image_n, currFrameTimeStamp);
  }
  
  { 
     unique_lock<mutex> lock(mMutexFrameDataBase);
     mframeDataBase[currFrameTimeStamp]= make_pair((input_rgb_image_n), (input_raw_depth_image_n));
  }
     current_frame_no_++;
  });  
    
  ///从OrbSlam中的localMapping线程提取经过localBA后的当前帧
  if(orbslam_static_scene_localBAKF()->empty()){
       return;
  }
  mcurrBAKeyframe = orbslam_static_scene_localBAKF()->front();
  orbslam_static_scene_localBAKF()->pop_front();
  if(mcurrBAKeyframe->isBad()){
     return;
  }
  double currBAKFTime = mcurrBAKeyframe->mTimeStamp;
//   cout << "DenseSlam 104: currBAKFTime: " << currBAKFTime << endl;
//   cout << "DenseSlam.cpp 175: currBAKFTime: " << currBAKFTime << endl;
  {
    unique_lock<mutex> lock(mMutexBAKF);
    std::map<double, std::pair<cv::Mat3b, cv::Mat1s>>::iterator iter;
    iter = mframeDataBase.find(currBAKFTime);
    if(iter != mframeDataBase.end()){
      input_rgb_image_copy_ = (iter->second.first).clone();
      input_raw_depth_image_copy_ = (iter->second.second).clone();
      //删除mframeDataBase和iter中间所有的元素
      mframeDataBase.erase(mframeDataBase.begin(), iter);
    }
    else{
	 return;
    }
  }
  orbSLAM2_Pose = mcurrBAKeyframe->GetPoseInverse();
  
    /// NOTE 如果是第一帧，则创建子地图，设置创建的子地图基于世界坐标系的位姿，这部分代码主要用于多子图的构建
  if(first_frame || shouldCreateNewLocalMap){
    int currentLocalMapIdx = static_scene_->GetMapManager()->createNewLocalMap();
    ITMLib::Objects::ITMPose tempPose;
    //InvM指世界坐标系到相机的变换
    tempPose.SetInvM(drivers::EigenToItm(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)));
    static_scene_->GetMapManager()->setEstimatedGlobalPose(currentLocalMapIdx, tempPose);
    todoList.push_back(TodoListEntry(currentLocalMapIdx,
				     lastKeyFrameTimeStamp,
				     lastKeyFrameTimeStamp));
    
    *orbslam_tracking_gl()=true;
    orbslam_tracking_cond()->notify_one();
//     orbslamVO.get();
//     *orbslam_tracking_gl() = false;
    
//     shouldCreateNewLocalMap = false;
    first_frame = false;
  }
  
  //这个判断条件主要是避免重复调用多的currentLocalMap
  if(currentLocalMap == NULL){
      currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
  }
  todoList.back().endKeyframeTimeStamp = lastKeyFrameTimeStamp;
  
  
  fusionFrameInfo currFusionFrame(orbSLAM2_Pose, input_rgb_image_copy_, input_raw_depth_image_copy_, 0);
  mfusionFrameDataBase[currBAKFTime] = currFusionFrame;
  
  vector<ORB_SLAM2::KeyFrame*> currAllKeyFrame = orbslam_static_scene_->GetMap()->GetAllKeyFrames();
//   sort(currAllKeyFrame.begin(),currAllKeyFrame.end(),ORB_SLAM2::KeyFrame::lts);
  
  //error, timestamp
  map<float, mapKeyframeInfo, greater<float>> mapPoseError;
  
  utils::Tic("Online Correction !");
  for(size_t i=0; i<currAllKeyFrame.size();i++){
      ORB_SLAM2::KeyFrame* currMapKeyframe = currAllKeyFrame[i];
      if(currMapKeyframe->isBad()){
	continue;
      }
      
      double currMapKeyframeTS  = currMapKeyframe->mTimeStamp;
      
      std::map<double, fusionFrameInfo>::iterator fusioniter;
      fusioniter = mfusionFrameDataBase.find(currMapKeyframeTS);
      
      if(fusioniter != mfusionFrameDataBase.end()){
	//计算误差倆位姿之间的误差
	Eigen::Matrix4f currPose = ORB_SLAM2::drivers::MatToEigen(currMapKeyframe->GetPoseInverse());
	Eigen::Matrix4f prePose = ORB_SLAM2::drivers::MatToEigen(fusioniter->second.poseinfo);
	fusioniter->second.flaginfo = 1;
	
	//李群,右差
	Eigen::Matrix4f poseDiff = prePose.inverse() * currPose;
	
	if(is_identity_matrix(ORB_SLAM2::drivers::EigenToMat(poseDiff))){
	  continue;
	}
		
	//转成李代数
	ITMLib::Objects::ITMPose tempDiff;
	tempDiff.SetInvM(drivers::EigenToItm(poseDiff));
	//se3的存储顺序为:
	//tx, ty, tz, rx, ry, rz
	float se3[6] = {0.0};
	for(int i=0; i<6; i++){
	  se3[i] = tempDiff.GetParams()[i];
	}
	
	//反对称符号
	Eigen::Matrix4f poseError;
	poseError <<  0.0,   -se3[5],  se3[4], se3[0],
	             se3[5],  0.0,   -se3[3],  se3[1],
		    -se3[4], se3[3],   0.0,    se3[2],
		      0.0,    0.0,     0.0,    0.0;
		      
	Eigen::Matrix4f poseErrorWeight;
	poseErrorWeight << 0.5, 0.0, 0.0, 0.0,
	                   0.0, 0.5, 0.0, 0.0,
			   0.0, 0.0, 0.5, 0.0,
			   0.0, 0.0, 0.0, 1.0;
	
	Eigen::Matrix4f innerProduct = poseError * poseErrorWeight * poseError.transpose();
	float traceValue =  innerProduct.trace();
	
	float rightError = sqrt(traceValue);
		
	mapKeyframeInfo currKeyframe(currMapKeyframeTS, currMapKeyframe->GetPoseInverse());
	
        mapPoseError[rightError] = currKeyframe;
      }
      else{
	continue;
      }
  }
  
  //当找到的位姿误差大于10帧了
  //在线调整的帧数
  int correctNum = 5;
  if(mapPoseError.size()>2){
    
     int countNum = 0;
     map<float,mapKeyframeInfo,greater<float>>::iterator errorIter;
     for(errorIter=mapPoseError.begin(); errorIter!=mapPoseError.end(); errorIter++){
       
       double timestamp = errorIter->second.timestampinfo;       
       std::map<double, fusionFrameInfo>::iterator defusioniter;
       defusioniter = mfusionFrameDataBase.find(timestamp);
       
       //其实这个判断条件应该可以不加的
       if(defusioniter!=mfusionFrameDataBase.end()){
	 
	 cv::Mat currPose = defusioniter->second.poseinfo;
// 	 cout << mfusionFrameDataBase[timestamp].poseinfo << endl;
	 cv::Mat3b currRGBInfo = defusioniter->second.rgbinfo;
	 cv::Mat1s currDepthInfo = defusioniter->second.depthinfo;
	 
// 	 std::cout << "DenseSlam.cpp 242: M_d: " << currPose.inv() << std::endl;

	 //Deintegrate
	 static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(currPose));
// 	 cout << "DenseSlam245: timestamp: " << timestamp << endl;
	 static_scene_->UpdateView(currRGBInfo, currDepthInfo, timestamp);
	 static_scene_->DeIntegrateLocalMap(currentLocalMap);
	 
 	 //Reintegrate
	 defusioniter->second.poseinfo = errorIter->second.poseinfok.clone();
	 currPose = defusioniter->second.poseinfo.clone();
// 	 cout << mfusionFrameDataBase[timestamp].poseinfo << endl;
// 	 std::cout << "DenseSlam.cpp 252: M_d: " << currPose.inv() << std::endl;

	 static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(currPose));
	 static_scene_->IntegrateLocalMap(currentLocalMap);
	 
	 countNum ++;
       }
       
       if(countNum > correctNum){
	 break;
       }
    } 
  }
  utils::Toc();
  
  int beforeCull = mfusionFrameDataBase.size();
  utils::Tic("Deintegrate unoptimizer fusion keyframe and map!");
  //清除掉那些没有在mapKeyframe中找到的已经融合到地图中的关键帧，因为这些帧并不参与orbslam中的优化，可能是经过cullkeyframe后已经被剔除掉的
  for(map<double, fusionFrameInfo>::iterator iter=mfusionFrameDataBase.begin(); iter!=mfusionFrameDataBase.end(); iter++){
    if(iter->second.flaginfo == 0){
      cv::Mat currPose = iter->second.poseinfo;
      cv::Mat3b currRGBInfo = iter->second.rgbinfo;
      cv::Mat1s currDepthInfo = iter->second.depthinfo;
      
      static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(currPose));
      static_scene_->UpdateView(currRGBInfo, currDepthInfo, 0.0);
      static_scene_->DeIntegrateLocalMap(currentLocalMap);
      
      mfusionFrameDataBase.erase(iter);
    }   
  }
  int afterCull = mfusionFrameDataBase.size();
  utils::Toc();
  cout << "Cull keyframe Num: "<< beforeCull - afterCull << endl;
  
  
  /// @brief 利用左右图像计算稀疏场景光流
  if(FLAGS_external_odo){
    /// 使用ORBSLAM的里程计
    if(FLAGS_useOrbSLAMVO){
      /// NOTE "2"意味着 OrbSLAM 跟踪成功
      /// 由于调用的是LocalBA后的位姿，因此可以不需要是否跟踪成功
      if(!orbSLAM2_Pose.empty()){
	    static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose));
	    {
	      unique_lock<mutex> lock(mMutexCond1);
	      (*orbslam_static_scene_->GetPreKeyframePose()) = orbSLAM2_Pose.clone();
	    }
// 	    cout << "DenseSlam 280: " << orbSLAM2_Pose << endl;
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
  /// 使用内部使用的里程计
  else{
      currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
      //主要为跟踪做准备
      static_scene_->PrepareNextStepLocalMap(currentLocalMap);
      //更新当前的RGB及深度图
      static_scene_->UpdateView(input_rgb_image_copy_, input_raw_depth_image_copy_, 0.0);
      //做跟踪
      static_scene_->TrackLocalMap(currentLocalMap);
      //由raycast得到的深度图与当前深度图做ICP跟踪得到的位姿Tw->c
  }
  utils::Toc();
  orbslamVO.get();
  
  utils::Tic("Static map fusion");
  if (FLAGS_external_odo && current_frame_no_ % experimental_fusion_every_ == 0 && !orbSLAM2_Pose.empty()) {
       static_scene_->UpdateView(input_rgb_image_copy_, input_raw_depth_image_copy_, currBAKFTime);
       static_scene_->IntegrateLocalMap(currentLocalMap);
       utils::Tic("Map decay");
       Decay();
       utils::Toc();
   }
   else{
      if (current_frame_no_ % experimental_fusion_every_ == 0) {
//          static_scene_->UpdateView(input_rgb_image_copy_, input_raw_depth_image_copy_);
         static_scene_->IntegrateLocalMap(currentLocalMap);
	 utils::Tic("Map decay");
         Decay();
         utils::Toc();
      }
   }
   int64_t fusion_time = utils::Toc();
   fusionTotalTime += fusion_time;
   /*
   if(shouldStartNewLocalMap(todoList.back().dataId) && !first_frame ){
       shouldCreateNewLocalMap = true;
       shouldClearPoseHistory = true;
  }
  */
   current_keyframe_no_ ++;
}

bool DenseSlam::is_identity_matrix(cv::Mat matrix){
  int flags = 1;
  for(int row=0; row<matrix.rows; row++){
    for(int col=0; col<matrix.cols; col++){
      if(matrix.at<float>(row,row) != 1.0){
	flags = 0;
      }
      if(row != col && matrix.at<float>(row, col) != 0.0){
	flags = 0;
      }
    }
  }
  if(flags == 0) {return false;}
  else {return true;}
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

void DenseSlam::SaveStaticMap(const std::string &dataset_name, const std::string &depth_name, const ITMLib::Engine::ITMLocalMap* currentLocalMap, int localMap_no) const {
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string map_fpath = target_folder + "/" + "mesh-" + depth_name + "-" + std::to_string(localMap_no) + "-frames.obj";
  static_scene_->SaveCurrSceneToMesh(map_fpath.c_str(), currentLocalMap->scene);
  cout << "Saving full static map to: " << map_fpath << endl;
}

std::string DenseSlam::EnsureDumpFolderExists(const string &dataset_name) const {
  string today_folder = utils::GetDate();
  string target_folder = "/home/hansry/DenseSLAM-Global-Consistency-h/mesh_out/" + dataset_name + "/" + today_folder;
  string mkdir_folder = "mkdir -p " + target_folder;
  bool unsucess = std::system(mkdir_folder.c_str());
//   if(unsucess) {
//     throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
//   }
  return target_folder;
}
}
