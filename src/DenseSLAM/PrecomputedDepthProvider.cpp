
#include <iostream>

#include <highgui.h>
#include "PrecomputedDepthProvider.h"
#include "Utils.h"
#include "../pfmLib/ImageIOpfm.h"

namespace SparsetoDense {

using namespace std;

const string kDispNetName = "precomputed-dispnet";
const string kPrecomputedElas = "precomputed-elas";

void PrecomputedDepthProvider::DisparityOrDepthMapFromStereo(const cv::Mat&,
                                                      const cv::Mat&,
                                                      cv::Mat &out_disparity) {
     ReadPrecomputed(this->input_->GetCurrentFrame(), out_disparity);
}

/// @brief 若input_is_depth_为true，则直接out是深度图，反之out是视差图
void PrecomputedDepthProvider::ReadPrecomputed(int frame_idx, cv::Mat &out) const {
  // TODO(andrei): Read correct precomputed depth when doing evaluation of arbitrary frames.
  // For testing, in the beginning we directly read depth (not disparity) maps from the disk.
  string depth_fpath = this->folder_ + "/" + utils::Format(this->fname_format_, frame_idx);

  // 读取由dispnet计算而来的depth
  if (utils::EndsWith(depth_fpath, ".pfm")) {
    // DispNet outputs depth maps as 32-bit float single-channel HDR images. Not a lot of programs
    // can load them natively for manual inspection. Photoshop can, but cannot natively display the
    // 32-bit image unless it is first converted to 16-bit.
    ReadFilePFM(out, depth_fpath);
  //读取OpenCV的XML
  } else {
    // Otherwise load an OpenCV XML dump (since we need 16-bit signed depth maps, which OpenCV
    // cannot save as regular images, even though the PNG spec has nothing against them).
    cv::FileStorage fs(depth_fpath, cv::FileStorage::READ);
    if(!fs.isOpened()) {
      throw std::runtime_error("Could not read precomputed depth map.");
    }
    fs["depth-frame"] >> out;
    if (out.type() != CV_16SC1) {
      throw std::runtime_error("Precomputed depth map had the wrong format.");
    }
  }

  if (out.cols == 0 || out.rows == 0) {
    throw runtime_error(utils::Format(
        "Could not read precomputed depth map from file [%s]. Please check that the file exists, and is a readable, valid, image.",
        depth_fpath.c_str()));
  }

  if (this->input_is_depth_) {
    // 由于直接读取深度，因此需要保证深度的最大值，保证是以mm为单位的
    float max_depth_mm_f = GetMaxDepthMeters() * kMetersToMillimeters;
    int16_t max_depth_mm_s = static_cast<int16_t>(round(max_depth_mm_f));

    for(int i = 0; i < out.rows; ++i) {
      for(int j = 0; j < out.cols; ++j) {
        if(out.type() == CV_32FC1) {
          float depth = out.at<float>(i, j);
          if (depth > max_depth_mm_f) {
            out.at<float>(i, j) = 0.0f;
          }
        }
        else {
          int16_t depth = out.at<int16_t>(i, j);
          if (depth > max_depth_mm_s) {
            out.at<int16_t>(i, j) = 0;
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

     ReadPrecomputed(frame_idx, out_disparity_);

     // TODO(andrei): Remove code duplication between this and 'DepthProvider'.
     if (out_disparity_.type() == CV_32FC1) {
         DepthFromDisparityMap<float>(out_disparity_, calibration, out_depth, scale);
     } 
     else if (out_disparity_.type() == CV_16SC1) {
        throw std::runtime_error("Unsupported.");
//      DepthFromDisparityMap<uint16_t>(out_disparity_, calibration, out_depth);
     } 
     else {
          throw std::runtime_error(utils::Format(
               "Unknown data type for disparity matrix [%s]. Supported are CV_32FC1 and CV_16SC1.", utils::Type2Str(out_disparity_.type()).c_str()
      ));
    }
}


const string &PrecomputedDepthProvider::GetName() const {
  if (utils::EndsWith(fname_format_, "pfm")) {
    return kDispNetName;
  }
  else {
    return kPrecomputedElas;
  }
}

} // namespace SparsetoDense
