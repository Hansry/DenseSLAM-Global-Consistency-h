#ifndef DENSESLAM_PRECOMPUTEDDEPTHPROVIDER_H
#define DENSESLAM_PRECOMPUTEDDEPTHPROVIDER_H

#include <string>

#include "DepthProvider.h"
#include "Input.h"


/// \brief 视差图和深度图都是从本地读取的，并不是通过双目进行实时的视差图计算
namespace SparsetoDense {

extern const std::string kDispNetName;
extern const std::string kPrecomputedElas;

/// \brief Reads precomputed disparity (default) or depth maps from a folder.
/// The depth maps are expected to be grayscale, and in short 16-bit or float 32-bit format.
class PrecomputedDepthProvider : public DepthProvider {
 public:
  PrecomputedDepthProvider(
      Input *input,
      const std::string &folder,
      const std::string &fname_format,
      bool input_is_depth,
      float min_depth_m,
      float max_depth_m)
      : DepthProvider(input_is_depth, min_depth_m, max_depth_m),
        input_(input),
        folder_(folder),
        fname_format_(fname_format) {}

  PrecomputedDepthProvider(const PrecomputedDepthProvider&) = delete; 
  PrecomputedDepthProvider(PrecomputedDepthProvider&&) = delete;
  PrecomputedDepthProvider& operator=(const PrecomputedDepthProvider&) = delete; 
  PrecomputedDepthProvider& operator=(PrecomputedDepthProvider&&) = delete;

  ~PrecomputedDepthProvider() override = default;

  /// \brief Loads the precomputed depth map for the specified frame into 'out_depth'.
  void GetDepth(int frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale) override;
  void GetDepth(double frame_idx, StereoCalibration& calibration, cv::Mat1s& out_depth, float scale) override;

 protected:
  /// \brief Reads a disparity or depth (depending on the data).
  template<typename T>
  void ReadPrecomputed(T frame_idx, cv::Mat &out) const;

 private:
  Input *input_;
  std::string folder_;
  /// \brief The printf-style format of the frame filenames, such as "frame-%04d.png" for frames
  /// which are called "frame-0000.png"-"frame-9999.png".
  std::string fname_format_;
};

} // namespace SparsetoDense

#endif //DENSESLAM_PRECOMPUTEDDEPTHPROVIDER_H
