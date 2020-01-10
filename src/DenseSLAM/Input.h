
#ifndef DENSESLAM_INPUT_H
#define DENSESLAM_INPUT_H

#include <string>
#include <highgui.h>
#include <memory>
#include <fstream>

#include "DepthProvider.h"
#include "Utils.h"
#include "../InfiniTAM-Global-Consistency-h/InfiniTAM/ITMLib/Objects/ITMRGBDCalib.h"

namespace SparsetoDense {

/// \brief Provides input from DynSLAM, in the form of RGBD frames.
/// Since DynSLAM currently operates with stereo input, this class also computes depth from stereo.
/// Currently, this is "computed" by reading the depth maps from disk, but the plan is to compute
/// depth on the fly in the future.
  
class Input { 
public:
  enum eSensor{
    MONOCULAR = 0,
    STEREO = 1,
    RGBD = 2
  };
  
  enum eDatasetType{
    KITTI = 0,
    TUM = 1,
    EUROC = 2
  };
 
  struct Config {
    std::string dataset_name;
    std::string left_gray_folder;
    std::string right_gray_folder;
    std::string left_color_folder;
    std::string right_color_folder;
    std::string fname_format;
    std::string calibration_fname;
    std::string frame_timestamp;

    /// \brief Minimum depth to keep when computing depth maps.
    float min_depth_m = -1.0f;
    /// \brief Maximum depth to keep when computing depth maps.
    float max_depth_m = -1.0f;

    // These are optional, and only used for precomputed depth.
    std::string depth_folder = "";
    std::string depth_fname_format = "";
    // Whether we read direct metric depth from the file, or just disparity values expressed in
    // pixels.
    bool read_depth = false;

    // Whether to read ground truth odometry information from an OxTS dump folder (e.g., KITTI
    // dataset), or from a single-file ground truth, as provided with the kitti-odometry dataset.
    // !! UNSUPPORTED AT THE MOMENT !!
    bool odometry_oxts = false;
    std::string odometry_fname = "";

    /// \brief The Velodyne LIDAR data (used only for evaluation).
    std::string velodyne_folder = "";
    std::string velodyne_fname_format = "";

    /// \brief Tracklet ground truth, only available in the KITTI-tracking benchmark in the format
    ///        that we support.
    std::string tracklet_folder = "";
  };

  /// We don't define the configs as constants here in order to make the code easier to read.
  /// Details and downloads: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
  static Config KittiOdometryConfig() {
    Config config;
    config.dataset_name           = "kitti-odometry";
    config.left_gray_folder       = "image_0";
    config.right_gray_folder      = "image_1";
    config.left_color_folder      = "image_2";
    config.right_color_folder     = "image_3";
    config.fname_format           = "%06d.png";
    config.calibration_fname      = "calib.txt";

    config.min_depth_m            =  0.5f;
    config.max_depth_m            = 30.0f;
    config.depth_folder           = "precomputed-depth/Frames";
    config.depth_fname_format     = "%04d.xml";
    config.read_depth             = true;

    config.odometry_oxts          = false;
    config.odometry_fname         = "ground-truth-poses.txt";

    config.velodyne_folder        = "velodyne";
    config.velodyne_fname_format  = "%06d.bin";
    config.frame_timestamp        = "";

    return config;
  };

  static Config KittiOdometryDispnetConfig() {
    Config config                 = KittiOdometryConfig();
    config.depth_folder           = "precomputed-depth-dispnet";
    config.depth_fname_format     = "%06d.pfm";
    config.read_depth             = false;
    
    return config;
  }
  
  static Config TUMOdometryConfig(){
    Config config;
    config.dataset_name        = "TUM-odometry";
    config.left_gray_folder    = "";
    config.right_gray_folder   = "";
    config.left_color_folder   = "rgb";
    config.right_gray_folder   = "";
    config.fname_format        = "%17f.png";   
    config.calibration_fname   = "calib.txt";
    
    config.min_depth_m         = 0.001f;
    config.max_depth_m         = 30.0f;
    config.depth_folder        = "depth";
    config.depth_fname_format  = "%17f.png";
    config.read_depth          = true;
    
    config.odometry_oxts       = false;
    config.odometry_fname      = "groundtruth.txt";
    
    config.velodyne_folder     =  "";
    config.velodyne_fname_format = "";
    
    config.frame_timestamp     = "associate.txt";        
    
    
    return config;
  }

 public:
  Input(const std::string &dataset_folder,
	const eSensor sensor,
	const eDatasetType datasetType,
        const Config &config,
        DepthProvider *depth_provider,
        const Eigen::Vector2i &frame_size,
        const StereoCalibration &stereo_calibration,
        int frame_offset,
        float input_scale)
      : dataset_folder_(dataset_folder),
        mSensor(sensor),
        mDatasetType(datasetType),
        config_(config),
        depth_provider_(depth_provider),
        frame_offset_(frame_offset),
        frame_idx_index(frame_offset),
        frame_width_(frame_size(0)),
        frame_height_(frame_size(1)),
        stereo_calibration_(stereo_calibration),
        depth_buf_(frame_size(1), frame_size(0)),
        input_scale_(input_scale),
        depth_buf_small_(static_cast<int>(round(frame_size(1) * input_scale)),
                         static_cast<int>(round(frame_size(0) * input_scale)))
    {
	if(mDatasetType == Input::TUM){
	   std::string fpath= dataset_folder + "/" + config.frame_timestamp;
	   if(utils::FileExists(fpath)){
	     getImageTimeStamp(fpath);
	   }
	   else{
	     std::runtime_error("the path to obtain frame timestamp doesn't exist!");
	  }	  
	  frame_idx_pair = timeStampVector[frame_idx_index];
	}
	else{
	  frame_idx_int = frame_idx_index;
	}
    }

  void getImageTimeStamp(std::string fpath) {
     std::ifstream in(fpath);
     std::string s;
     while(getline(in,s)){
        if(s[0] == '#') continue;
	std::string substring = s.substr(0,s.rfind("rgb")-1);
	std::string rgb_timestamp = substring.substr(0,substring.rfind(' '));
	//这里加18是因为depth_timestamp的位数为17位
	std::string depth_timestamp = substring.substr(substring.rfind(' ')+1, substring.rfind(' ')+18);
	timeStampVector.push_back(std::make_pair(rgb_timestamp, depth_timestamp));
     }
  }
  
  //判断是否还有剩余的图片
  bool HasMoreImages() const;

  /// \brief 讲读取器推进到下一帧，如果下一帧读取成功则返回True
  bool ReadNextFrame();

  /// \brief 返回最新RGB和深度图的指针
  void GetCvImages(cv::Mat3b **rgb, cv::Mat1s **raw_depth);

  /// \brief Returns pointers to the latest grayscale input frames.
  void GetCvStereoGray(cv::Mat1b **left, cv::Mat1b **right);

  /// \brief Returns pointers to the latest color input frames.
  void GetCvStereoColor(cv::Mat3b **left_rgb, cv::Mat3b **right_rgb);

  cv::Size2i GetRgbSize() const {
    return cv::Size2i(frame_width_, frame_height_);
  }

  cv::Size2i GetDepthSize() const {
    return cv::Size2i(frame_width_, frame_height_);
  }

  /// \brief Gets the name of the dataset folder which we are using.
  /// TODO(andrei): Make this more robust.
  std::string GetSequenceName() const {
    return dataset_folder_.substr(dataset_folder_.rfind('/') + 1);
  }

  std::string GetDatasetIdentifier() const {
    return config_.dataset_name + "-" + GetSequenceName();
  }

  DepthProvider* GetDepthProvider() const {
    return depth_provider_;
  }

  //this指向这个实例的地址，将传进来的depth_provider实例赋值给该实例
  void SetDepthProvider(DepthProvider *depth_provider) {
    this->depth_provider_ = depth_provider;
  }

  /// \brief Returns the current frame index from the dataset.
  /// \note May not be the same as the current DynSLAM frame number if an offset was used.
  int GetCurrentFrame_int() const {
      return frame_idx_int;
  }
  
  std::pair<std::string, std::string> GetCurrentFrame_str() const {
      return frame_idx_pair;
  }
  
  /// \brief Sets the out parameters to the RGB and depth images from the specified frame.
  template<typename T>
  void GetFrameCvImages(T frame_idx, std::shared_ptr<cv::Mat3b> &rgb, std::shared_ptr<cv::Mat1s> &raw_depth);

  const Config& GetConfig() const {
    return config_;
  }

  int GetFrameOffset() const {
    return frame_offset_;
  }
  
  eSensor GetSensorType() const {
    return mSensor;
  }
  
  void GetRightColor(cv::Mat3b &out) const;
  
  eDatasetType GetDatasetType() const{
    return mDatasetType;
  }

 private:
  //数据类型
  eSensor mSensor;
  eDatasetType mDatasetType;
  
  std::vector<std::pair<std::string,std::string>> timeStampVector; 
  
  std::string dataset_folder_;
  Config config_;
  DepthProvider *depth_provider_;
  const int frame_offset_;
  int frame_idx_index;
  
  int frame_idx_int = 0;
  std::pair<std::string, std::string> frame_idx_pair = std::make_pair("","");
  
  int frame_width_;
  int frame_height_;

  StereoCalibration stereo_calibration_;

  cv::Mat3b left_frame_color_buf_;
  cv::Mat3b right_frame_color_buf_;
  cv::Mat1s depth_buf_;

  // Store the grayscale information necessary for scene flow computation using libviso2, and
  // on-the-fly depth map computation using libelas.
  cv::Mat1b left_frame_gray_buf_;
  cv::Mat1b right_frame_gray_buf_;

  /// Used when evaluating low-resolution input.
  float input_scale_;
  cv::Mat1s depth_buf_small_;
//  cv::Mat1s raw_depth_small(static_cast<int>(round(GetDepthSize().height * input_scale_)),
//  static_cast<int>(round(GetDepthSize().width * input_scale_)));

  template<typename T>
  static std::string GetFrameName(const std::string &root, const std::string &folder, const std::string &fname_format, T frame_idx) {
    return root + "/" + folder + "/" + utils::Format(fname_format, frame_idx);
  }
 
  template<typename T>
  void ReadLeftGray(T frame_idx, cv::Mat1b &out) const;
  
  template<typename T>
  void ReadRightGray(T frame_idx, cv::Mat1b &out) const;
  
  template<typename T>
  void ReadLeftColor(T frame_idx, cv::Mat3b &out) const;
  
  template<typename T>
  void ReadRightColor(T frame_idx, cv::Mat3b &out) const;
};

} // namespace SparsetoDense

#endif //DENSESLAM_INPUT_H
