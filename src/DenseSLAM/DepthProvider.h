
#ifndef DENSESLAM_DEPTHPROVIDER_H
#define DENSESLAM_DEPTHPROVIDER_H

#include <limits>

#include <opencv/cv.h>
#include "Utils.h"

// Some necessary forward declarations
namespace SparsetoDense {
namespace utils {
std::string Type2Str(int type);
std::string Format(const std::string &fmt, ...);
}
}

namespace SparsetoDense {

/// \brief Contains the calibration parameters of a stereo rig, such as the AnnieWAY platform,
/// which is the one used to record the KITTI dataset.
struct StereoCalibration {
  float baseline_meters;
  float focal_length_px;

  StereoCalibration(float baseline_meters, float focal_length_px)
      : baseline_meters(baseline_meters), focal_length_px(focal_length_px) {}
};

/// \brief ABC for components computing depth from stereo image pairs.
/// \note The methods of this interface are designed according to the OpenCV API style, and
/// return their results into pre-allocated out parameters.
class DepthProvider {
 public:
  static constexpr float kMetersToMillimeters = 1000.0f;

 public:
  DepthProvider (const DepthProvider&) = default;//默认深拷贝构造函数
  DepthProvider (DepthProvider&&) = default;//默认移动构造函数
  DepthProvider& operator=(const DepthProvider&) = default;
  DepthProvider& operator=(DepthProvider&&) = default;
  virtual ~DepthProvider() = default;//虚析构构造函数

  /// \brief 从双目图像中计算深度图（Stereo->disparity->depth）
  virtual void DepthFromStereo(const cv::Mat &left,
                               const cv::Mat &right,
                               const StereoCalibration &calibration,
                               cv::Mat1s &out_depth,
                               float scale
  ) {
    if (input_is_depth_) {
      // Our input is designated as direct depth, not just disparity.
      DisparityMapFromStereo(left, right, out_depth);
      return;
    }

    // 先计算视差图，再计算深度
    DisparityMapFromStereo(left, right, out_disparity_);

    // This should be templated in a nicer fashion...
    if (out_disparity_.type() == CV_32FC1) {
      DepthFromDisparityMap<float>(out_disparity_, calibration, out_depth, scale);
    } 
    else if (out_disparity_.type() == CV_16SC1) {
      throw std::runtime_error("Cannot currently convert int16_t disparity to depth.");
    } 
    else {
      throw std::runtime_error(utils::Format(
          "Unknown data type for disparity matrix [%s]. Supported are CV_32FC1 and CV_16SC1.",
          utils::Type2Str(out_disparity_.type()).c_str()
      ));
    }
  }

  /// \brief 从双目图像中计算视差图
  virtual void DisparityMapFromStereo(const cv::Mat &left,
                                      const cv::Mat &right,
                                      cv::Mat &out_disparity) = 0;

  /// \brief 将视差图转为深度图，单位为m，d=baseline*fx/disparity
  virtual float DepthFromDisparity(const float disparity_px,
                                   const StereoCalibration &calibration) {
    return (calibration.baseline_meters * calibration.focal_length_px) / disparity_px;
  }

  // TODO-LOW(andrei): This can be sped up trivially using CUDA.
  /// \brief Computes a depth map from a disparity map using the `DepthFromDisparity` function at
  /// every pixel.
  /// \tparam T The type of the elements in the disparity input.
  /// \param disparity The disparity map.
  /// \param calibration The stereo calibration parameters used to compute depth from disparity.
  /// \param out_depth The output depth map, which gets populated by this method.
  /// \param scale Used to adjust the depth-from-disparity formula when using reduced-resolution
  ///              input. Unless evaluating the system's performance on low-res input, this should
  ///              be set to 1.
  /// 使用模板可以很好的适应输入的视差图的类型
  template<typename T>
  void DepthFromDisparityMap(const cv::Mat_<T> &disparity,
                             const StereoCalibration &calibration,
                             cv::Mat1s &out_depth,
                             float scale
  ) {
    assert(disparity.size() == out_depth.size());
    assert(!input_is_depth_ && "Should not attempt to compute depth from disparity when the read "
        "data is already a depth map, and not just a disparity map.");

    // max_depth是对于地图的质量是很重要的，如果max_depth太大，那么地图会有很多噪声，如果太小，地图上就是有路或者人行道
    int32_t min_depth_mm = static_cast<int32_t>(min_depth_m_ * kMetersToMillimeters);//kMetersToMillimeters=1000.0f
    int32_t max_depth_mm = static_cast<int32_t>(max_depth_m_ * kMetersToMillimeters);

    // InfiniTAM requires short depth maps, so we need to ensure our depth can actually fit in a
    // short.
    // std::numeric_limits<int16_t>::max()返回的是编译器允许的int16_t型数的最大值:2^{16}-1
    int32_t max_representable_depth = std::numeric_limits<int16_t>::max();
    if (max_depth_mm >= max_representable_depth) {
      throw std::runtime_error(utils::Format("Unsupported maximum depth of %f meters (%d mm, "
                                                 "larger than the %d limit).", max_depth_m_,
                                             max_depth_mm, max_representable_depth));
    }

    //遍历视差图
    for (int i = 0; i < disparity.rows; ++i) {
      for (int j = 0; j < disparity.cols; ++j) {
        T disp = disparity.template at<T>(i, j);
	//通过static_cast强制转换为int32_t类型
        int32_t depth_mm = static_cast<int32_t>(kMetersToMillimeters * scale * DepthFromDisparity(disp, calibration));

        if (abs(disp) < 1e-5) {
          depth_mm = 0;
        }

        if (depth_mm > max_depth_mm || depth_mm < min_depth_mm) {
          depth_mm = 0;
        }

        int16_t depth_mm_short = static_cast<int16_t>(depth_mm);
        out_depth.at<int16_t>(i, j) = depth_mm_short;
      }
    }
  }

  /// \brief The name of the technique being used for depth estimation.
  virtual const std::string &GetName() const = 0;

  //获取深度的最小值阈值
  float GetMinDepthMeters() const {
    return min_depth_m_;
  }

  //设置最小深度值阈值
  void SetMinDepthMeters(float min_depth_m) {
    this->min_depth_m_ = min_depth_m;
  }

  //获取深度的最大值阈值
  float GetMaxDepthMeters() const {
    return max_depth_m_;
  }

  //设置深度的最大值阈值
  void SetMaxDepthMeters(float max_depth_m) {
    this->max_depth_m_ = max_depth_m;
  }

 protected:
  /// \param input_is_depth 输入是深度图还是视差图
  /// \param min_depth_m 最小深度
  /// \param max_depth_m 最大深度
  explicit DepthProvider(bool input_is_depth, float min_depth_m, float max_depth_m) :
      input_is_depth_(input_is_depth),
      min_depth_m_(min_depth_m),
      max_depth_m_(max_depth_m) {}//显示构造函数，表明该构造函数不能被隐式转换

  /// \brief If true, then assume the read maps are depth maps, instead of disparity maps.
  /// In this case, the depth from disparity computation is no longer performed.
  bool input_is_depth_;
  /// \brief Buffer in which the disparity map gets saved at every frame.
  cv::Mat out_disparity_;

 private:
  float min_depth_m_;
  float max_depth_m_;
};

} // namespace SparsetoDense

#endif //DENSESLAM_DEPTHPROVIDER_H
