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
  
                                      
  virtual void GetDepth(int frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale) = 0;
  virtual void GetDepth(double frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale) = 0;

  /// \brief 将视差图转为深度图，单位为m，d=baseline*fx/disparity
  float DepthFromDisparity(const float disparity_px, const StereoCalibration &calibration) {
	return calibration.focal_length_px * calibration.baseline_meters / disparity_px;
  };
				   
  /// @brief 使用'DepthFromDisparity'函数从视差图计算深度图，应该可以用CUDA进行加速才对
  /// @param T 输入的视差图的类型
  /// @param disparity 视差图
  /// @param calibration 用来从视差图计算深度的双目标定参数
  /// \param out_depth 输出的深度图
  /// \param scale 当输入的图片是低像素的时候，用来调整depth-from-disparity方程，除非用低像素的图片来进行评估，在这种情况下，则scale=1
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
        //将深度保存为mm的精度，同时以16位进行存储
        int16_t depth_mm_short = static_cast<int16_t>(depth_mm);
        out_depth.at<int16_t>(i, j) = depth_mm_short;
      }
    }
  }

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
  /// @param input_is_depth 输入是深度图还是视差图
  /// @param min_depth_m 最小深度
  /// @param max_depth_m 最大深度
  /// @Note 构造函数放在protected类型中,通过PrecomputedDepthProvider进行构造
  explicit DepthProvider(bool input_is_depth, float min_depth_m, float max_depth_m) :
      input_is_depth_(input_is_depth),
      min_depth_m_(min_depth_m),
      max_depth_m_(max_depth_m) {}//显示构造函数，表明该构造函数不能被隐式转换

  /// @Note 如果input_is_depth=true_，那么输入为深度图，计算视差图的函数将不会被计算
  bool input_is_depth_;
  /// @brief Buffer in which the disparity map gets saved at every frame.
  cv::Mat out_disparity_;

 private:
  float min_depth_m_;
  float max_depth_m_;
};

} // namespace SparsetoDense

#endif //DENSESLAM_DEPTHPROVIDER_H
