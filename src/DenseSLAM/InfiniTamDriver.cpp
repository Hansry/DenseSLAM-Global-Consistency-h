

#include "InfiniTamDriver.h"

// TODO(andrei): Why not move to DynSLAM.cpp?
DEFINE_bool(enable_evaluation, true, "Whether to enable evaluation mode for DynSLAM. This means "
    "the system will load in LIDAR ground truth and compare its maps with it, dumping the results "
    "in CSV format.");

namespace SparsetoDense {
namespace drivers {

using namespace SparsetoDense::utils;

/// \brief Converts between the DynSlam preview type enums and the InfiniTAM ones.
ITMLib::Engine::ITMMainEngine::GetImageType GetItmVisualization(PreviewType preview_type) {
  switch(preview_type) {
    case PreviewType::kDepth:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_ORIGINAL_DEPTH;
    case PreviewType::kGray:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_SHADED;
    case PreviewType::kColor:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_VOLUME;
    case PreviewType::kNormal:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_NORMAL;
//     case PreviewType::kWeight:
//      return ITMLib::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_COLOUR_FROM_DEPTH_WEIGHT;
    case PreviewType::kRaycastImage:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_SCENERAYCAST; 
    case PreviewType::kLatestRaycast:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_SHADED;
    case PreviewType::kRaycastDepth:
      return ITMLib::Engine::ITMMainEngine::GetImageType::InfiniTAM_IMAGE_FREECAMERA_DEPTH;

    default:
      throw std::runtime_error(Format("Unknown preview type: %d", preview_type));
  }
}

//ITMPose指InfiniTAM中表示位姿的数据类型
ITMLib::Objects::ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix) {
  Matrix4f M;
  for(int i = 0; i < 16; ++i) {
    M.m[i] = static_cast<float>(pangolin_matrix.m[i]);
  }

  ITMLib::Objects::ITMPose itm_pose;
  itm_pose.SetM(M);
  itm_pose.Coerce();

  return itm_pose;
}

//这个函数主要为ITMLib的RGBD输入提供内参，其内参有专门的一个类
ITMLib::Objects::ITMRGBDCalib* CreateItmCalib(const Eigen::Matrix<double, 3, 4> &left_cam_proj,
                                              const Eigen::Vector2i &frame_size) {
  ITMLib::Objects::ITMRGBDCalib *calib = new ITMLib::Objects::ITMRGBDCalib;
  float kMetersToMillimeters = 1.0f / 1000.0f;

  ITMLib::Objects::ITMIntrinsics intrinsics;
  float fx = static_cast<float>(left_cam_proj(0, 0));
  float fy = static_cast<float>(left_cam_proj(1, 1));
  float cx = static_cast<float>(left_cam_proj(0, 2));
  float cy = static_cast<float>(left_cam_proj(1, 2));
  float sizeX = frame_size(0);
  float sizeY = frame_size(1);
  intrinsics.SetFrom(fx, fy, cx, cy, sizeX, sizeY);

  // all depth maps using the RGB inputs.
  calib->intrinsics_rgb = intrinsics;
  calib->intrinsics_d = intrinsics;

  // RGB和depth的变换矩阵，RGB和depth是同个相机，因此其相对位姿为单位矩阵
  Matrix4f identity; identity.setIdentity();
  calib->trafo_rgb_to_depth.SetFrom(identity);

  // These parameters are used by ITM to convert from the input depth, expressed in millimeters, to
  // the internal depth, which is expressed in meters.
  calib->disparityCalib.SetFrom(kMetersToMillimeters, 0.0f, ITMLib::Objects::ITMDisparityCalib::TRAFO_AFFINE);
  return calib;
}

//将OpenCV RGB Mat格式转成InfiniTAM所需的图片格式
void CvToItm(const cv::Mat3b &mat, ITMUChar4Image *out_itm) {
  Vector2i newSize(mat.cols, mat.rows);
  out_itm->ChangeDims(newSize);
  Vector4u *data_ptr = out_itm->GetData(MEMORYDEVICE_CPU);

  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      int idx = i * mat.cols + j;
      // Convert from OpenCV's standard BGR format to RGB.
      cv::Vec3b col = mat.at<cv::Vec3b>(i, j);
      data_ptr[idx].b = col[0];
      data_ptr[idx].g = col[1];
      data_ptr[idx].r = col[2];
      data_ptr[idx].a = 255u;
    }
  }

  // This does not currently work because the input images lack the alpha channel.
//    memcpy(data_ptr, mat.data, mat.rows * mat.cols * 4 * sizeof(unsigned char));
}

//将OpenCV Mat格式的深度转成InfiniTAM深度格式
void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_itm) {
  short *data_ptr = out_itm->GetData(MEMORYDEVICE_CPU);
  out_itm->ChangeDims(Vector2i(mat.cols, mat.rows));
  memcpy(data_ptr, mat.data, mat.rows * mat.cols * sizeof(short));
}

//将InfiniTAM rgb(a)图像转换成OpenCV Mat格式，丢弃alpha通道的信息
void ItmToCv(const ITMUChar4Image &itm, cv::Mat3b *out_mat) {
  const Vector4u *itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  for (int i = 0; i < itm.noDims[1]; ++i) {
    for (int j = 0; j < itm.noDims[0]; ++j) {
      out_mat->at<cv::Vec3b>(i, j) = cv::Vec3b(
          itm_data[i * itm.noDims[0] + j].b,
          itm_data[i * itm.noDims[0] + j].g,
          itm_data[i * itm.noDims[0] + j].r
      );
    }
  }
}

void ItmToCvMat(const ITMUChar4Image *itm, cv::Mat& out_mat) {
  const Vector4u *itm_data = itm->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  for (int i = 0; i < itm->noDims[1]; ++i) {
    for (int j = 0; j < itm->noDims[0]; ++j) {
      out_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(
          itm_data[i * itm->noDims[0] + j].b,
          itm_data[i * itm->noDims[0] + j].g,
          itm_data[i * itm->noDims[0] + j].r
      );
    }
  }
}

void ResizeRaycastDepthGUI(const ITMUChar4Image *itm, const Vector2i srcSize, const Vector2i dstSize, cv::Mat& dstImg){
  const Vector4u *itm_data = itm->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  cv::Mat srcImg(srcSize.height, srcSize.width, CV_8UC3);
  for (int i = 0; i < srcImg.rows; ++i) {
    for (int j = 0; j < srcImg.cols; ++j) {
       srcImg.at<cv::Vec3b>(i, j) = cv::Vec3b(
          itm_data[i * srcImg.cols + j].b,
          itm_data[i * srcImg.cols + j].g,
          itm_data[i * srcImg.cols + j].r
      );
    }
  }
//   cv::imshow("before resize srcImg: ", srcImg);
  cv::Size outputImgSize(dstSize.width, dstSize.height);
  cv::resize(srcImg, dstImg, outputImgSize, 0, 0, CV_INTER_CUBIC);
//   cv::imshow("resize srcImg:", dstImg);
//   cv::waitKey(0);
}

void Char4RGBToUC3(const unsigned char *pixels, cv::Mat &out_mat) {
  for (int i = 0; i < out_mat.rows; ++i) {
    for (int j = 0; j < out_mat.cols; j+=3) {
      out_mat.at<cv::Vec3b>(i, j) = cv::Vec3b(pixels[i * out_mat.cols + j], pixels[i*out_mat.cols + j+1], pixels[i*out_mat.cols + j+2]);
    }
  }
}

/// @brief 将深度图从Float转成int_16类型,这里将float转成short会乘上1000
void FloatDepthmapToShort(const float *pixels, cv::Mat1s &out_mat) {
  const int kMetersToMillimeters = 1000;
//   int count = 0;
  for (int i = 0; i < out_mat.rows; ++i) {
    for (int j = 0; j < out_mat.cols; ++j) {
      /// ITM internal: depth = meters, float
      /// Our preview:  depth = mm, short int
      out_mat.at<int16_t>(i, j) = static_cast<int16_t>(
          pixels[i * out_mat.cols + j] * kMetersToMillimeters
      );
    }
  }
}

/// @brief 将InfiniTAM 深度图转换成OpenCV Mat格式
void ItmToCv(const ITMShortImage &itm, cv::Mat1s *out_mat) {
  const int16_t *itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  memcpy(out_mat->data, itm_data, itm.noDims[0] * itm.noDims[1] * sizeof(int16_t));
}

void FloatDepthmapToInt16(const float *pixels, cv::Mat &out_mat) {
  const int kMetersToMillimeters = 1000;
//   int count = 0;
  for (int i = 0; i < out_mat.rows; ++i) {
    for (int j = 0; j < out_mat.cols; ++j) {
      /// ITM internal: depth = meters, float
      /// Our preview:  depth = mm, short int
      out_mat.at<int16_t>(i, j) = static_cast<int16_t>(
          pixels[i * out_mat.cols + j] * kMetersToMillimeters
      );
    }
  }
}

/// @brief 将InfiniTAM深度图从float类型转成short类型
void ItmDepthToCv(const ITMFloatImage &itm, cv::Mat1s *out_mat) {
  const float* itm_data = itm.GetData(MemoryDeviceType::MEMORYDEVICE_CPU);
  FloatDepthmapToShort(itm_data, *out_mat);
}

/// @brief 将InfiniTAM 4x4矩阵转成Eigen类型
Eigen::Matrix4f ItmToEigen(const Matrix4f &itm_matrix) {
  Eigen::Matrix4f res;
  res << itm_matrix.at(0, 0), itm_matrix(1, 0), itm_matrix.at(2, 0), itm_matrix.at(3, 0),
    itm_matrix.at(0, 1), itm_matrix.at(1, 1), itm_matrix.at(2, 1), itm_matrix.at(3, 1),
    itm_matrix.at(0, 2), itm_matrix.at(1, 2), itm_matrix.at(2, 2), itm_matrix.at(3, 2),
    itm_matrix.at(0, 3), itm_matrix.at(1, 3), itm_matrix.at(2, 3), itm_matrix.at(3, 3);
  return res;
}

/// @brief 将Eigen类型转成InfiniTAM 4x4的矩阵类型
Matrix4f EigenToItm(const Eigen::Matrix4f &eigen_matrix) {
  // Note the ordering, which is necessary since the input to the ITM matrix must be given in column-major format.
  // ITM matrix 的存储是列存储的
  Matrix4f res(eigen_matrix(0, 0), eigen_matrix(1, 0), eigen_matrix(2, 0), eigen_matrix(3, 0),
               eigen_matrix(0, 1), eigen_matrix(1, 1), eigen_matrix(2, 1), eigen_matrix(3, 1),
               eigen_matrix(0, 2), eigen_matrix(1, 2), eigen_matrix(2, 2), eigen_matrix(3, 2),
               eigen_matrix(0, 3), eigen_matrix(1, 3), eigen_matrix(2, 3), eigen_matrix(3, 3));
  return res;
}

/// @brief 得到InfiniTAM地图中在model_view视角下的图片,char类型
void InfiniTamDriver::GetImage(ITMUChar4Image *out,
                               SparsetoDense::PreviewType get_image_type,
                               const pangolin::OpenGlMatrix &model_view,
			       const ITMLocalMap *currentLocalMap){
  if (nullptr != this->view) {
    ITMLib::Objects::ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    if (get_image_type == PreviewType::kDepth) {
      std::cerr << "Warning: Cannot preview depth normally anymore." << std::endl;
      return;
    }
    
    ITMLib::Objects::ITMIntrinsics intrinsics = this->viewBuilder->GetCalib()->intrinsics_d;
    ITMLib::Engine::ITMMainEngine::GetImage(
        out,
	nullptr,
        GetItmVisualization(get_image_type),
        &itm_freeview_pose,
        &intrinsics,
	currentLocalMap);
  }
  // Otherwise: We're before the very first frame, so no raycast is available yet.
}

/// @brief 得到InfiniTAM地图中在model_view视角下的图片,float类型，可能对于深度图比较有用
void InfiniTamDriver::GetFloatImage(
    ITMFloatImage *out,
    SparsetoDense::PreviewType get_image_type,
    const pangolin::OpenGlMatrix &model_view,
    const ITMLocalMap *currentLocalMap
) {
  if (nullptr != this->view) {
    ITMLib::Objects::ITMPose itm_freeview_pose = PoseFromPangolin(model_view);

    if (get_image_type != PreviewType::kRaycastDepth) {
      std::cerr << "Warning: Can only preview depth as float." << std::endl;
      return;
    }

    ITMLib::Objects::ITMIntrinsics intrinsics = this->viewBuilder->GetCalib()->intrinsics_d;
    ITMLib::Engine::ITMMainEngine::GetImage(
        nullptr,
        out,
        GetItmVisualization(get_image_type),
        &itm_freeview_pose,
        &intrinsics,
	currentLocalMap);
  }
}

//将OpenCV中的rgb_image和raw_depth_image转换成InfiniTAM中对应的数据类型，并进行更新
void InfiniTamDriver::UpdateView(const cv::Mat3b &rgb_image,
                                 const cv::Mat1s &raw_depth_image,
				 const double timestamp) {
  CvToItm(rgb_image, rgb_itm_);
  CvToItm(raw_depth_image, raw_depth_itm_);
  
  this->viewBuilder->UpdateView(&view, rgb_itm_, raw_depth_itm_, timestamp, settings->useBilateralFilter);
  
}

} // namespace drivers
} // namespace SparsetoDense
