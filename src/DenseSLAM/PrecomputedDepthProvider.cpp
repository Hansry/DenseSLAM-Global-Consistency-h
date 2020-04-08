
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthProvider.h"
#include "Utils.h"
#include "../pfmLib/ImageIOpfm.h"

namespace SparsetoDense {

using namespace std;

/// @brief 若input_is_depth_为true，则直接out是深度图，反之out是视差图
template<typename T>
void PrecomputedDepthProvider::ReadPrecomputed(T frame_idx, cv::Mat &out) const {
  string depth_fpath = this->folder_ + "/" + utils::Format(this->fname_format_, frame_idx);
  // 读取由dispnet计算而来的depth
  cout << "PrecomputedDepthProvider: 39: " << depth_fpath << endl;
  out = cv::imread(depth_fpath, CV_LOAD_IMAGE_UNCHANGED); //Remeber to add "-1"
  
  if (out.cols == 0 || out.rows == 0) {
    throw runtime_error(utils::Format(
        "Could not read precomputed depth map from file [%s]. Please check that the file exists, and is a readable, valid, image.",
        depth_fpath.c_str()));
  }
  
//    cv::imshow("out_before:",out);
//    cv::waitKey(0);

  if (this->input_is_depth_) {
    // 由于直接读取深度，因此需要保证深度的最大值，保证是以mm为单位的
    // 由于TUM是以5000的比例进行存储的，所以将kMeterToMillimeters这个值设为5000
    float max_depth_mm_f = GetMaxDepthMeters() * kMetersToMillimeters;
    int16_t max_depth_mm_s = static_cast<int16_t>(round(max_depth_mm_f));
    float kitti_factor = 1000.0/256.0;
    for(int i = 0; i < out.rows; ++i) {
      for(int j = 0; j < out.cols; ++j) {
        if(out.type() == CV_32FC1) {
	  if(this->input_->GetDatasetType() == Input::KITTI){
	   out.at<float>(i,j) = out.at<float>(i,j); 
	  }
	  if(this->input_->GetDatasetType() == Input::TUM || this->input_->GetDatasetType() == Input::ICLNUIM){
            out.at<float>(i, j) = out.at<float>(i, j) / 5.0;
	  }
          float depth = out.at<float>(i, j);
          if (depth > max_depth_mm_f) {
            out.at<float>(i, j) = 0.0f;
          }
        }
        else {
	  ///由于对于TUM-RGBD数据集，其存储在图片上是以 realValue(m)*5000(5*mm)存储的，这里我们的out是16位存储为mm，因此需要除以5
	  if(this->input_->GetDatasetType() == Input::KITTI){
	    if(out.at<int16_t>(i,j) > GetMaxDepthMeters()*256){
	      out.at<int16_t>(i,j)=0;
	    }
	    out.at<int16_t>(i, j) = static_cast<int16_t>((float)out.at<int16_t>(i, j)*kitti_factor);
	  }
	  else if(this->input_->GetDatasetType() == Input::TUM || this->input_->GetDatasetType() == Input::ICLNUIM){
            out.at<int16_t>(i, j) = static_cast<int16_t>(((float)out.at<int16_t>(i, j))/5.0);
	    
	    if (out.at<int16_t>(i, j) > max_depth_mm_s) {
               out.at<int16_t>(i, j) = 0;
            }
          }
        }
      }
    }
  }
}

void PrecomputedDepthProvider::GetDepth(int frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale){
    if (input_is_depth_) {
       std::cout << "Will read precomputed depth..." << std::endl;
       ReadPrecomputed(frame_idx, out_depth);
       std::cout << "Done reading precomputed depth for specific frame [" << frame_idx << "]." << std::endl;
       return;
     }
}

void PrecomputedDepthProvider::GetDepth(double frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale){
    if (input_is_depth_) {
       std::cout << "Will read precomputed depth..." << std::endl;
       ReadPrecomputed(frame_idx, out_depth);
       std::cout << "Done reading precomputed depth for specific frame [" << frame_idx << "]." << std::endl;
       return;
     }
}

} // namespace SparsetoDense
