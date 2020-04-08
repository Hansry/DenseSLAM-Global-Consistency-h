
#include "Input.h"
#include "PrecomputedDepthProvider.h"

namespace SparsetoDense {

using namespace std;

template<typename T>
void Input::GetFrameCvImages(T frame_idx, std::shared_ptr<cv::Mat3b> &rgb, std::shared_ptr<cv::Mat1s> &raw_depth) {
  
  cv::Mat3b rgb_right_temp(GetRgbSize());
  rgb.reset(new cv::Mat3b(GetRgbSize()));
  raw_depth.reset(new cv::Mat1s(GetDepthSize()));

  ReadLeftColor(frame_idx, *rgb);
  if(mSensor == STEREO){
      ReadRightColor(frame_idx, rgb_right_temp);
  }
   // If we're using precomputed depth, make sure you tell it exactly which frame we are evaluating.
  GetDepthProvider()->GetDepth(frame_idx, this->stereo_calibration_, depth_buf_small_, input_scale_);
  cv::resize(depth_buf_small_, *raw_depth, cv::Size(), 1.0/input_scale_, 1.0/input_scale_, cv::INTER_NEAREST);
}

bool Input::HasMoreImages() {
  string next_fpath = "";
  if(mDatasetType == KITTI){
    next_fpath = GetFrameName(dataset_folder_, config_.left_color_folder, config_.fname_format, frame_idx_int);
    cout << "next_fpath: 29" << next_fpath << endl;
  }
  else if(mDatasetType == TUM){
    double frame_idx = atof(frame_idx_pair.first.c_str());
    next_fpath = GetFrameName(dataset_folder_, config_.left_color_folder, config_.fname_format, frame_idx);
  }
  else if(mDatasetType == ICLNUIM){
    next_fpath = GetFrameName(dataset_folder_, config_.left_color_folder, config_.fname_format, frame_idx_int);
  }
  else{
    throw runtime_error("Currently Unspported dataset type !");
  }
  bool fileExists = utils::FileExists(next_fpath);
  if(!fileExists){
    frame_idx_index++;
    frame_idx_int = frame_idx_index;
  }
  return fileExists;
}

bool Input::ReadNextFrame() {
  /// @brief 读取id为frame_idx的color image,并将其保存在left or right_frame_color_buf_中
  if(mDatasetType == KITTI){
    ReadLeftColor(frame_idx_int, left_frame_color_buf_);
    currentFrameTimeStamp = (double)frame_idx_int;
  }
  else if(mDatasetType == TUM){
    double frame_idx = atof(frame_idx_pair.first.c_str());
    ReadLeftColor(frame_idx, left_frame_color_buf_);
    string tempString = frame_idx_pair.first;
    std::istringstream ss(tempString);
    ss >> currentFrameTimeStamp;
    printf("%s%f\n","Input 50: currentFrame:",currentFrameTimeStamp);
  }
  else if(mDatasetType == ICLNUIM){
    ReadLeftColor(frame_idx_int, left_frame_color_buf_);
    currentFrameTimeStamp = (double)frame_idx_int;
  }
  else{
    throw runtime_error("Currently Unspported dataset type !");
  }
  const auto &rgb_size = GetRgbSize();
  
  if(left_frame_color_buf_.rows != rgb_size.height || left_frame_color_buf_.cols != rgb_size.width){
     int u_coner = (left_frame_color_buf_.cols - rgb_size.width)/2.0;
     int v_coner = left_frame_color_buf_.rows - rgb_size.height;
     cv::Rect rect(u_coner, v_coner, rgb_size.width, rgb_size.height);
     left_frame_color_buf_ = left_frame_color_buf_(rect);
  }
  
  
  if (left_frame_color_buf_.rows != rgb_size.height || left_frame_color_buf_.cols != rgb_size.width) {
    cerr << "Unexpected left RGB frame size. Got " << left_frame_color_buf_.size() << ", but the "
         << "calibration file specified " << rgb_size << "." << endl;
    cerr << "Was using format [" << config_.fname_format << "] in dir ["
         << config_.left_color_folder << "]." << endl;
    return false;
  }
  
  if(mSensor==Input::STEREO){
    if(mDatasetType == KITTI){
       ReadRightColor(frame_idx_int, right_frame_color_buf_);
    }
    else if(mDatasetType == TUM){
       double frame_idx = atof(frame_idx_pair.first.c_str());
       ReadRightColor(frame_idx, right_frame_color_buf_);
    }
    else if(mDatasetType == ICLNUIM){
       ReadRightColor(frame_idx_int, right_frame_color_buf_);
    }
    else{
       throw runtime_error("Unspporte frame_idx type !");
    }
    
    if(right_frame_color_buf_.rows != rgb_size.height || right_frame_color_buf_.cols != rgb_size.width){
       int u_coner = (right_frame_color_buf_.cols - rgb_size.width)/2.0;
       int v_coner = right_frame_color_buf_.rows - rgb_size.height;
       cv::Rect rect(u_coner, v_coner, rgb_size.width, rgb_size.height);
       right_frame_color_buf_ = right_frame_color_buf_(rect);
     }
     
     // Sanity checks to ensure the dimensions from the calibration file and the actual image dimensions correspond.
     if (right_frame_color_buf_.rows != rgb_size.height || right_frame_color_buf_.cols != rgb_size.width) {
        cerr << "Unexpected right RGB frame size. Got " << right_frame_color_buf_.size() << ", but the calibration file specified " << rgb_size << "." << endl;
        cerr << "Was using format [" << config_.fname_format << "] in dir [" << config_.right_color_folder << "]." << endl;
        return false;
    }
  }
  
  cv::Mat1s &depth_out = (input_scale_ != 1.0f) ? depth_buf_small_ : depth_buf_;
  if(mDatasetType == KITTI){
       depth_provider_->GetDepth(frame_idx_int, stereo_calibration_, depth_out, input_scale_);
  }
  else if(mDatasetType == TUM){
       double frame_idx = atof(frame_idx_pair.second.c_str());
       depth_provider_->GetDepth(frame_idx, stereo_calibration_, depth_out, input_scale_);
  }
  else if(mDatasetType == ICLNUIM){
      depth_provider_->GetDepth(frame_idx_int, stereo_calibration_, depth_out, input_scale_);
  }
  else{
      throw runtime_error("Unspporte frame_idx type !");
  }
  if (input_scale_ != 1.0f) {
    cv::resize(depth_buf_small_,
               depth_buf_,
               cv::Size(),
               1.0 / input_scale_,
               1.0 / input_scale_,
               cv::INTER_NEAREST);
  }

  const auto &depth_size = GetDepthSize();
  
  if(depth_buf_.rows != depth_size.height || depth_buf_.cols != depth_size.width){
     int u_coner = (depth_buf_.cols - depth_size.width)/2.0;
     int v_coner = depth_buf_.rows - depth_size.height;
     cv::Rect rect(u_coner, v_coner, depth_size.width, depth_size.height);
     depth_buf_ = depth_buf_(rect);
  }
  
  if (depth_buf_.rows != depth_size.height || depth_buf_.cols != depth_size.width) {
    cerr << "Unexpected depth map size. Got " << depth_buf_.size() << ", but the calibration file specified " << depth_size << "." << endl;
    cerr << "Was using format [" << config_.depth_fname_format << "] in dir ["
         << config_.depth_folder << "]." << endl;
    return false;
  }

  frame_idx_index ++;
  if(mDatasetType == TUM){
    frame_idx_pair = timeStampVector[frame_idx_index];
  }
  frame_idx_int = frame_idx_index; 
  return true;
}

void Input::GetRightColor(cv::Mat3b **rgb){
  *rgb = &right_frame_color_buf_;
}

void Input::GetCvImages(cv::Mat3b **rgb, cv::Mat1s **raw_depth) {
  *rgb = &left_frame_color_buf_;
  *raw_depth = &depth_buf_;
}

void Input::GetCvStereoGray(cv::Mat1b **left, cv::Mat1b **right) {
  *left = &left_frame_gray_buf_;
  *right = &right_frame_gray_buf_;
}

void Input::GetCvStereoColor(cv::Mat3b **left_rgb, cv::Mat3b **right_rgb) {
  *left_rgb = &left_frame_color_buf_;
  *right_rgb = &right_frame_color_buf_;
}

/// @brief 读取id为frame_idx的left gray image
template<typename T>
void Input::ReadLeftGray(T frame_idx, cv::Mat1b &out) const {
  out = cv::imread(GetFrameName(dataset_folder_,
                                config_.left_gray_folder,
                                config_.fname_format,
                                frame_idx),
                                CV_LOAD_IMAGE_UNCHANGED);
}

/// @brief 读取id为frame_idx的right gray image
template<typename T>
void Input::ReadRightGray(T frame_idx, cv::Mat1b &out) const {
  out = cv::imread(GetFrameName(dataset_folder_,
                                config_.right_gray_folder,
                                config_.fname_format,
                                frame_idx),
                                CV_LOAD_IMAGE_UNCHANGED);

}

/// @brief 读取id为frame_idx的left color image,支持scale,最近邻插值
template<typename T>
void Input::ReadLeftColor(T frame_idx, cv::Mat3b &out) const {
  cv::Mat3b buf = cv::imread(GetFrameName(dataset_folder_,
                                          config_.left_color_folder,
                                          config_.fname_format,
                                          frame_idx));
  cv::resize(buf, out, cv::Size(), 1 / input_scale_, 1 / input_scale_, cv::INTER_NEAREST);
}

/// @brief 读取id为frame_idx的right color image，支持scale,最近邻插值
template<typename T>
void Input::ReadRightColor(T frame_idx, cv::Mat3b &out) const {
  cv::Mat3b buf = cv::imread(GetFrameName(dataset_folder_,
                                          config_.right_color_folder,
                                          config_.fname_format,
                                          frame_idx));
  cv::resize(buf, out, cv::Size(), 1 / input_scale_, 1 / input_scale_, cv::INTER_NEAREST);
}

} // namespace SparsetoDense
