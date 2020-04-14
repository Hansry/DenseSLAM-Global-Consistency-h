#include <chrono>
#include <thread>
#include <future>

#include "DenseSlam.h"

namespace SparsetoDense {

void DenseSlam::ProcessFrame(Input *input) {
  // Read the images from the first part of the pipeline
//   if (! input->HasMoreImages()) {
//      cout << "No more frames left in image source." << endl;
//      return;
//   }
  
  utils::Tic("Read input and compute depth");
  if(!input->ReadNextFrame()) {
     throw runtime_error("Could not read input from the data source.");
  }
  
  currFrameTimeStamp = input->GetCurrentFrame_double();
//   printf("%s%f", "DenseSlam26, currentFrame TimeStamp: \n", currFrameTimeStamp);
  
  bool first_frame = (current_keyframe_no_ == 1);
  /// @brief 更新当前buf存储的color image和 depth 
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  if(input->GetSensorType() == Input::STEREO){
    input->GetRightColor(&input_right_rgb_image_);
  }
  utils::Toc();
  
  /// @brief 对orbslam进行跟踪，同时进行线程的分离
  utils::Tic("Compute VO");
  
  input_rgb_image_n = (*input_rgb_image_).clone();
  input_raw_depth_image_n = (*input_raw_depth_image_).clone();
  
  cv::Mat orbSLAMInputDepth(input_raw_depth_image_n.rows, input_raw_depth_image_n.cols, CV_32FC1);

  for(int row =0; row<input_rgb_image_n.rows; row++){
    for(int col =0; col<input_rgb_image_n.cols; col++){
      orbSLAMInputDepth.at<float>(row,col) = ((float)input_raw_depth_image_n.at<int16_t>(row,col))/1000.0;
    }
  }
  
     ///@brief 主要是为了在orbslam中加上icp,但是貌似没有太大的效果，待测
//   if(!first_frame && out_image_float_->GetData(MEMORYDEVICE_CPU)!=nullptr){
//     cv::Mat raycastDepth(input_raw_depth_image_n.rows, input_raw_depth_image_n.cols, CV_32FC1);
//     for(int row=0; row<input_rgb_image_n.rows; row++){
//       for(int col=0; col<input_rgb_image_n.cols; col++){
// 	raycastDepth.at<float>(row,col) = static_cast<float>(out_image_float_->GetData(MEMORYDEVICE_CPU)[row * input_rgb_image_n.cols + col]);
//       }
//     }
//     
//     unique_lock<mutex> lock(mMutexCond);
//     *(orbslam_static_scene_->GetTrackinRaycastDepth()) = raycastDepth.clone();
//     *(orbslam_static_scene_->GetTrackingDepth()) = orbSLAMInputDepth.clone();
//   }
  
  future<void> orbslamVO = async(launch::async, [this, &input, &orbSLAMInputDepth]{
    
//   imshow("orbSLAMInputDepth: ", orbSLAMInputDepth);
//   cv::waitKey(0);
  
  if(input->GetSensorType() == Input::RGBD){
     orbslam_static_scene_trackRGBD(input_rgb_image_n, 
				    orbSLAMInputDepth, 
				    currFrameTimeStamp);
  }
  else if(input->GetSensorType() == Input::STEREO){
     input_right_rgb_image_n = (*input_right_rgb_image_).clone();
     orbslam_static_scene_trackStereo(input_rgb_image_n, input_right_rgb_image_n, currFrameTimeStamp);
  }
  else if(input->GetSensorType() == Input::MONOCULAR){
     orbslam_static_scene_trackMonular(input_rgb_image_n, currFrameTimeStamp);
  }
  
  { 
     unique_lock<mutex> lock(mMutexFrameDataBase);
     cv::Mat depthweight = cv::Mat::ones(input_raw_depth_image_->rows, input_raw_depth_image_->cols, CV_8UC1);
     currFrameInfo currNormalFrame(input_rgb_image_n, input_raw_depth_image_n, depthweight, currFrameTimeStamp);
     mframeDataBase[currFrameTimeStamp] = currNormalFrame;
//      mframeDataBase[currFrameTimeStamp]= make_pair((input_rgb_image_n), (input_raw_depth_image_n));
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
  //世界坐标系到当前帧坐标系的变换矩阵
  orbSLAM2_Pose = mcurrBAKeyframe->GetPoseInverse();
  
  utils::Tic("Depth of fusion frame update and post processing !");
  if(post_processing_.enabled){
     //这是为了第一帧关键帧在融合的时候没有相邻帧进行depth postproceessing,因此保证mframeDataBase大于
     if(mframeDataBase.size() < 2){
        return;
     }
     bool depthPocessSucess = depthPostProcessing(currBAKFTime);
     if(!depthPocessSucess){
         return;
     }
  }
  else{
    unique_lock<mutex> lock(mMutexBAKF);
    std::map<double, currFrameInfo>::iterator iter;
    iter = mframeDataBase.find(currBAKFTime); 
    if(iter != mframeDataBase.end()){
       ///当前关键帧的深度以及颜色信息
      input_rgb_image_copy_ = (iter->second.rgbinfoc).clone();  
      input_raw_depth_image_copy_ = (iter->second.depthinfoc).clone();
      input_weight_copy_ = (iter->second.depthweightinfoc).clone();
      
      //删除mframeDataBase和iter中间所有的元素
      mframeDataBase.erase(mframeDataBase.begin(), iter);
    }
    else{
      return;
    }
  }
  utils::Toc();

    /// NOTE 如果是第一帧，则创建子地图，设置创建的子地图基于世界坐标系的位姿，这部分代码主要用于多子图的构建
  if(first_frame || shouldCreateNewLocalMap){
    
    int currentLocalMapIdx = static_scene_->GetMapManager()->createNewLocalMap();
    ITMLib::Objects::ITMPose tempPose;
    //InvM指世界坐标系到相机的变换
    //将orbslam地图的第一帧看做是世界坐标系，因此在构建稠密地图的时候需要设置世界坐标系到稠密地图第一帧的变换矩阵
    tempPose.SetInvM(drivers::EigenToItm(ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)));
    static_scene_->GetMapManager()->setEstimatedGlobalPose(currentLocalMapIdx, tempPose);
    todoList.push_back(TodoListEntry(currentLocalMapIdx,
				     lastKeyFrameTimeStamp,
				     lastKeyFrameTimeStamp));
    
    *orbslam_tracking_gl()=true;
    orbslam_tracking_cond()->notify_one();
    first_frame = false;
  }
  
  //这个判断条件主要是避免重复调用多的currentLocalMap
  if(currentLocalMap == NULL){
      currentLocalMap = static_scene_->GetMapManager()->getLocalMap(todoList.back().dataId);
  }
  todoList.back().endKeyframeTimeStamp = lastKeyFrameTimeStamp;
  
  fusionFrameInfo currFusionFrame(orbSLAM2_Pose, input_rgb_image_copy_, input_raw_depth_image_copy_, 0, input_weight_copy_);
  mfusionFrameDataBase[currBAKFTime] = currFusionFrame;
  mfusionFrameDataBaseForRaycast[currBAKFTime] = currFusionFrame;
  

  if(save_raycastdepth_.enabled){
     const std::string dataset_name = input->GetSequenceName();
     const std::string fname_format = input->GetConfig().fname_format;
    
     SaveRaycastDepth(dataset_name, fname_format);
  }
  
  if(save_raycastrgb_.enabled){
    const std::string dataset_name = input->GetSequenceName();
    const std::string fname_format = input->GetConfig().fname_format;
    
    SaveRaycastRGB(dataset_name, fname_format);
  }
  
  utils::Tic("Online Correction !");
  if(online_correction_.enabled){
     OnlineCorrection();
  }
  utils::Toc();
  
  if(use_orbslam_vo_ && !orbSLAM2_Pose.empty()){
    /// 使用ORBSLAM的里程计
      /// NOTE "2"意味着 OrbSLAM 跟踪成功
      /// 由于调用的是LocalBA后的位姿，因此可以不需要是否跟踪成功
    static_scene_->SetPoseLocalMap(currentLocalMap, ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose));
    {
      unique_lock<mutex> lock(mMutexCond1);
      (*orbslam_static_scene_->GetPreKeyframePose()) = orbSLAM2_Pose.clone();
    }
    Eigen::Matrix4f currLocalMapPose = drivers::ItmToEigen(currentLocalMap->estimatedGlobalPose.GetM());
    if(shouldClearPoseHistory){
	  pose_history_.clear();
    }
    pose_history_.push_back((currLocalMapPose * ORB_SLAM2::drivers::MatToEigen(orbSLAM2_Pose)).inverse());
    shouldClearPoseHistory = false;
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
  orbslamVO.get();
  
  utils::Tic("Static map fusion");
  if (use_orbslam_vo_ &&  !orbSLAM2_Pose.empty()) {
       static_scene_->UpdateView(input_rgb_image_copy_, input_raw_depth_image_copy_, currBAKFTime);
       static_scene_->IntegrateLocalMap(currentLocalMap);
       
       if(mfusionFrameDataBase.size() > slide_window_.max_age && slide_window_.enabled){
         utils::Tic("Slide Window");
         SlideWindowMap();
	 SlideWindowPose();
         utils::Toc();
       }
       
       utils::Tic("Map decay");
       Decay();
       utils::Toc();
   }
   else{
//       static_scene_->UpdateView(input_rgb_image_copy_, input_raw_depth_image_copy_);
         static_scene_->IntegrateLocalMap(currentLocalMap);
	 
	 if(mfusionFrameDataBase.size() > slide_window_.max_age && slide_window_.enabled){
	    utils::Tic("Slide Window");
            SlideWindowMap();
	    SlideWindowPose();
            utils::Toc();
	 }
	 
	 utils::Tic("Map decay");
         Decay();
         utils::Toc();
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

void DenseSlam::SlideWindowPose(){
     int DataBaseSize = mfusionFrameDataBase.size();
     int cullSize = DataBaseSize - slide_window_.max_age;
     map<double, fusionFrameInfo>::iterator iter= mfusionFrameDataBase.begin();
     int count = 0;
     for(iter; iter!=mfusionFrameDataBase.end(); iter++){
       mfusionFrameDataBase.erase(iter);
       count ++;
       if(count > (cullSize-1)){
	 break;
      }
    }
}

bool DenseSlam::OnlineCorrection(){
    vector<ORB_SLAM2::KeyFrame*> currAllKeyFrame = orbslam_static_scene_->GetMap()->GetAllKeyFrames();
//   sort(currAllKeyFrame.begin(),currAllKeyFrame.end(),ORB_SLAM2::KeyFrame::lts);
  
  //error, timestamp
  map<float, mapKeyframeInfo, greater<float>> mapPoseError;
  
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
  int correctNum = online_correction_.CorrectionNum;
  if(mapPoseError.size()>10){
    
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
}

bool DenseSlam::depthPostProcessing(double currBAKFTime){
  
    float fx = projection_left_rgb_(0,0);
    float fy = projection_left_rgb_(1,1);
    float cx = projection_left_rgb_(0,2);
    float cy = projection_left_rgb_(1,2);
    float inv_fx = 1.0/fx;
    float inv_fy = 1.0/fy;
    
    unique_lock<mutex> lock(mMutexBAKF);

    std::map<double, currFrameInfo>::iterator iter_curr;
    iter_curr = mframeDataBase.find(currBAKFTime);
    
    std::map<double, currFrameInfo>::iterator iter_prev_rgb;
    iter_prev_rgb = mframeDataBase.find(currBAKFTime);
    
    int count = 0;
    if(iter_curr != mframeDataBase.end()){
      
      cv::Mat1s curr_depth = iter_curr->second.depthinfoc.clone();
      cv::Mat curr_gray;
      cv::cvtColor(iter_curr->second.rgbinfoc, curr_gray, cv::COLOR_BGR2GRAY);
      
      if(mframeDataBase.size() > 1 && orbslam_static_scene_->GetgetFrameDataInfo()->size() > 1){
	iter_prev_rgb--;
	
	std::map<double, cv::Mat>::iterator iter_prev_pose;
        iter_prev_pose = orbslam_static_scene_->GetgetFrameDataInfo()->find(iter_prev_rgb->second.timestampinfoc);
	
	if(iter_prev_pose != orbslam_static_scene_->GetgetFrameDataInfo()->end()){
	
	   cv::Mat prev_gray;
	   cv::cvtColor(iter_prev_rgb->second.rgbinfoc, prev_gray, cv::COLOR_BGR2GRAY);
	
	   cv::Mat1s prev_depth = iter_prev_rgb->second.depthinfoc.clone();
	
	   cv::Mat prev_pose = (iter_prev_pose->second).clone();
	
//            cv::Mat combineImg(curr_gray.rows, 2*curr_gray.cols, CV_8UC1);
//            for(int row=0; row < curr_gray.rows; row++){
// 	     for(int col=0; col < curr_gray.cols; col++){
// 	       combineImg.at<uint8_t>(row,col) = curr_gray.at<uint8_t>(row,col);
// 	     }
// 	   }
// 	
// 	   for(int row=0; row < curr_gray.rows; row++){
// 	     for(int col=curr_gray.cols; col < 2*curr_gray.cols; col++){
// 	        combineImg.at<uint8_t>(row,col) = prev_gray.at<uint8_t>(row,col);
// 	     }
// 	   }
	   
	   if(post_processing_.show_post_processing){
 	      imshow("previous depth", curr_depth);
	   }
	   for(int row=0; row < curr_depth.rows; row++){
 	      for(int col=0; col < curr_depth.cols; col++){
	        cv::Mat depth_curr_coord = cv::Mat::ones(3,1,CV_32FC1);
	        cv::Mat depth_prev_coord = cv::Mat::ones(3,1,CV_32FC1);
	        //将空间点从图像坐标投影到基于当前帧的空间点
	        depth_curr_coord.at<float>(2,0) = ((float)curr_depth.at<int16_t>(row,col))/1000.0;
	        if(depth_curr_coord.at<float>(2,0)<0.005) {
	          continue;
	        }
	     
	        depth_curr_coord.at<float>(0,0) = depth_curr_coord.at<float>(2,0) * (row - cx) * inv_fx; //X = Z*(u-cx)/fx
	        depth_curr_coord.at<float>(1,0) = depth_curr_coord.at<float>(2,0) * (col - cy) * inv_fy; //Y = Z*(v-cy)/fy
	     
	        //当前帧相对于上一帧的变换矩阵Tpc = Tpw * Twc
	        cv::Mat Tpc =  (prev_pose.inv()) * orbSLAM2_Pose;
	        //世界坐标系下的空间点
	        depth_prev_coord = Tpc.rowRange(0,3).colRange(0,3) * depth_curr_coord + Tpc.rowRange(0,3).col(3);
	     
	        //将前一帧的空间点投影到前一帧的像平面上
	        //+0.5主要是为了四舍五入
	        int row_u =  fx * depth_prev_coord.at<float>(0,0) * (1.0/depth_prev_coord.at<float>(2,0)) + cx + 0.5; //u = fx*X/Z + cx
	        int col_v =  fy * depth_prev_coord.at<float>(1,0) * (1.0/depth_prev_coord.at<float>(2,0)) + cy + 0.5; //v = fy*Y/Z + cy
	     
	        if(row_u < 0.1 || col_v < 0.1 || (row_u+1) > prev_gray.rows || (col_v+1) > prev_gray.cols){
	          continue;
	        } 
	     
 	        float prev_gray_intensity =  (float)prev_depth.at<uint16_t>(row_u, col_v) / 1000.0;
	           if(prev_gray_intensity<0.005){
 	              continue;
 	        }
	     
	        float curr_gray_intensity =  depth_prev_coord.at<float>(2,0);
	        float diff_depth = abs(prev_gray_intensity - curr_gray_intensity);
	     
	        if((diff_depth/curr_gray_intensity) > post_processing_.filterThreshold && row > post_processing_.filterArea * curr_depth.rows){
	           curr_depth.at<int16_t>(row,col) = 0;
// 	           cv::line(combineImg, cv::Point(col,row), cv::Point(prev_gray.cols+col_v, row_u), cv::Scalar(0,0,0), 1, 8);
	        }
	        count ++;
	     }
	   }
	   
	   orbslam_static_scene_->GetgetFrameDataInfo()->erase(orbslam_static_scene_->GetgetFrameDataInfo()->begin(), iter_prev_pose);
	 }      
      }
      
      ///当前关键帧的深度以及颜色信息
      input_rgb_image_copy_ = (iter_curr->second.rgbinfoc).clone();  
      input_raw_depth_image_copy_ = curr_depth.clone();
      if(post_processing_.show_post_processing){
        cv::imshow("filted depth: ", input_raw_depth_image_copy_);
        cv::waitKey(0);
      }
      input_weight_copy_ = (iter_curr->second.depthweightinfoc).clone();
      
      //删除mframeDataBase和iter中间所有的元素
      mframeDataBase.erase(mframeDataBase.begin(), iter_prev_rgb);
      return true;
    }
    else{
	 return false;
    }
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

void DenseSlam::SaveRaycastDepth(const std::string &dataset_name, const string &fname_format) {
  
    if(mfusionFrameDataBaseForRaycast.size() > save_raycastdepth_.delayNum){
        map<double, fusionFrameInfo>::iterator raycast_iter = mfusionFrameDataBaseForRaycast.begin();
	// Tsd = (Tdw * Tws).inv() 
        Eigen::Matrix4f currLocalMapPose = drivers::ItmToEigen(currentLocalMap->estimatedGlobalPose.GetM()); //Tdw
	cv::Mat currDenseMapPose = ORB_SLAM2::drivers::EigenToMat(currLocalMapPose); //Tdw
	
        cv::Mat raycast_pose = (currDenseMapPose * raycast_iter->second.poseinfo).inv();
	
        pangolin::OpenGlMatrix pm_raycast(ORB_SLAM2::drivers::MatToEigen(raycast_pose));
        int current_preview_depth_type = PreviewType::kRaycastDepth;
        /// NOTE 光线投影回来的深度图
     
        cv::Mat tempRaycastShort(raycast_iter->second.depthinfo.rows, raycast_iter->second.depthinfo.cols, CV_16UC1);
        const float* tempRaycastDepth = GetRaycastDepthPreview(pm_raycast, static_cast<PreviewType>(current_preview_depth_type), save_raycastdepth_.compositing_dense); 
        //将raycast depth转成16位
        SparsetoDense::FloatDepthmapToInt16(tempRaycastDepth, tempRaycastShort);
     
        string target_folder = "../raycastdepth/" + dataset_name;
        string mkdir_folder = "mkdir -p " + target_folder;
        std::system(mkdir_folder.c_str());
        string index = utils::Format(fname_format ,(int)raycast_iter->first);
        string imgSavePath = target_folder + "/" + index;
        cv::imwrite(imgSavePath, tempRaycastShort);
        raycast_iter++;
	if(!save_raycastrgb_.enabled){
           mfusionFrameDataBaseForRaycast.erase(mfusionFrameDataBaseForRaycast.begin(), raycast_iter);
	}
   }
}

void DenseSlam::SaveRaycastRGB(const std::string &dataset_name, const string &fname_format) {
     if(mfusionFrameDataBaseForRaycast.size() > save_raycastrgb_.delayNum){
        map<double, fusionFrameInfo>::iterator raycast_iter = mfusionFrameDataBaseForRaycast.begin();
	
	// Tsd = (Tdw * Tws).inv() 
        Eigen::Matrix4f currLocalMapPose = drivers::ItmToEigen(currentLocalMap->estimatedGlobalPose.GetM()); //Tdw
	cv::Mat currDenseMapPose = ORB_SLAM2::drivers::EigenToMat(currLocalMapPose); //Tdw
        cv::Mat raycast_pose = (currDenseMapPose * raycast_iter->second.poseinfo).inv();
	
        pangolin::OpenGlMatrix pm_raycast(ORB_SLAM2::drivers::MatToEigen(raycast_pose));
        int current_preview_depth_type = PreviewType::kColor;
        /// NOTE 光线投影回来的深度图
     
        cv::Mat tempRaycastShort(raycast_iter->second.depthinfo.rows, raycast_iter->second.depthinfo.cols, CV_8UC3);

	ITMUChar4Image* out_save_image_ = new ITMUChar4Image(input_shape_, true, true);
	static_scene_->GetImage(out_save_image_, static_cast<PreviewType>(current_preview_depth_type), pm_raycast, currentLocalMap);
	
        //将raycast depth转成16位
        SparsetoDense::ItmToCvMat(out_save_image_, tempRaycastShort);
     
        string target_folder = "../raycastRGB/" + dataset_name;
        string mkdir_folder = "mkdir -p " + target_folder;
        std::system(mkdir_folder.c_str());
        string index = utils::Format(fname_format ,(int)raycast_iter->first);
        string imgSavePath = target_folder + "/" + index; 
        cv::imwrite(imgSavePath, tempRaycastShort);
	delete out_save_image_;
        raycast_iter++;
        mfusionFrameDataBaseForRaycast.erase(mfusionFrameDataBaseForRaycast.begin(), raycast_iter);
   }
}

void DenseSlam::SaveStaticMap(const std::string &dataset_name, const ITMLib::Engine::ITMLocalMap* currentLocalMap, int localMap_no) const {
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string map_fpath = target_folder + "/" + "mesh-" + std::to_string(localMap_no) + "-frames.obj";
  static_scene_->SaveCurrSceneToMesh(map_fpath.c_str(), currentLocalMap->scene);
  cout << "Saving full static map to: " << map_fpath << endl;
}

std::string DenseSlam::EnsureDumpFolderExists(const string &dataset_name) const {
  string today_folder = utils::GetDate();
  string target_folder = "../mesh_out/" + dataset_name + "/" + today_folder;
  string mkdir_folder = "mkdir -p " + target_folder;
  bool unsucess = std::system(mkdir_folder.c_str());
//   if(unsucess) {
//      throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
//   }
  return target_folder;
}
}
