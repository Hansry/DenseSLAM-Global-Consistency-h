#include <chrono>
#include <thread>
#include <future>

#include "DenseSlam.h"

DEFINE_bool(dynamic_weights, false, "Whether to use depth-based weighting when performing fusion.");
DECLARE_bool(semantic_evaluation);
DECLARE_int32(evaluation_delay);

namespace SparsetoDense {

void DenseSlam::ProcessFrame(Input *input) {
  // Read the images from the first part of the pipeline
  if (! input->HasMoreImages()) {
    cout << "No more frames left in image source." << endl;
    return;
  }

  bool first_frame = (current_frame_no_ == 0);

  utils::Tic("Read input and compute depth");
  if(!input->ReadNextFrame()) {
    throw runtime_error("Could not read input from the data source.");
  }
  utils::Toc();

  /// \brief 利用左右图像计算稀疏场景光流
  future<void> ssf_and_vo = async(launch::async, [this, &input, &first_frame] {
    utils::Tic("Sparse Scene Flow");

    // Whether to use input from the original cameras. Unavailable with the tracking dataset.
    // When original gray images are not used, the color ones are converted to grayscale and passed
    // to the visual odometry instead.
    bool original_gray = false;

    // TODO-LOW(andrei): Reuse these buffers for performance.
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

    // TODO(andrei): Idea: compute only matches here, the make the instance reconstructor process
    // the frame and remove clearly-dynamic SF vectors (e.g., from tracks which are clearly dynamic,
    // as marked from a prev frame), before computing the egomotion, and then processing the
    // reconstructions. This may improve VO accuracy, and it could give us an excuse to also
    // evaluate ATE and compare it with the results from e.g., StereoScan, woo!
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

    bool external_odo = false;
    if (external_odo) {
      //new_pose为当前帧到世界坐标系(第一帧)下的位姿变换
      //Tcurrent_w = Tcurrent_previous * previous_w
      Eigen::Matrix4f new_pose = delta * pose_history_[pose_history_.size() - 1]; 
      
      static_scene_->SetPose(new_pose.inverse());
      pose_history_.push_back(new_pose);//将当前帧的位姿存储到vector中，方便下一次计算使用
    }
    else {
      // Used when we're *not* computing VO as part of the SF estimation process.
      // Tcurrent_w = Tcurrent_previous * previous_w
      Eigen::Matrix4f new_pose_sp = delta * pose_history_[pose_history_.size() - 1]; 
      std::cout << SparsetoDense::EigenToItm(new_pose_sp.inverse())<<std::endl;
      // new_pose_sp.inverse(): Tw_current
//       std::cout << "new_pose: " << SparsetoDense::EigenToItm(new_pose_sp) << std::endl;
      static_scene_->SetPose(new_pose_sp.inverse());
//       std::cout << "new_pose_inv: " << SparsetoDense::EigenToItm(new_pose_sp.inverse()) << std::endl;

      pose_history_.push_back(new_pose_sp);
      
//      将pose设置为单位阵，那么将一直在同一个地方上进行融合
//       Eigen::Matrix4f identify_pose;
//       identify_pose.setIdentity();
//       static_scene_->SetPose(identify_pose);
//       pose_history_.push_back(new_pose_sp);
//       Eigen::Matrix4f get_pose = static_scene_->GetPose();
    }

    if (! original_gray) {
      delete left_gray;
      delete right_gray;
    }
    utils::Toc("Visual Odometry", false);
  });

//  seg_result_future.wait();
// 'get' ensures any exceptions are propagated (unlike 'wait').
  ssf_and_vo.get();
  
  utils::Tic("Input preprocessing");
  input->GetCvImages(&input_rgb_image_, &input_raw_depth_image_);
  static_scene_->UpdateView(*input_rgb_image_, *input_raw_depth_image_);
  utils::Toc();
  
  // Perform the tracking after the segmentation, so that we may in the future leverage semantic
  // information to enhance tracking.
//   if (! first_frame) {
    if (current_frame_no_ % experimental_fusion_every_ == 0) {
      utils::Tic("Static map fusion");
      static_scene_->Integrate();
      static_scene_->PrepareNextStep();
      utils::TocMicro();

      // Idea: trigger decay not based on frame gap, but using translation-based threshold.
      // Decay old, possibly noisy, voxels to improve map quality and reduce its memory footprint.
//       utils::Tic("Map decay");
//       static_scene_->Decay();
//       utils::TocMicro();
    }
//   }
  
  // Final sanity check after the frame is processed: individual components should check for errors.
  // If something slips through and gets here, it's bad and we want to stop execution.
//   ITMSafeCall(cudaDeviceSynchronize());
//   cudaError_t last_error = cudaGetLastError();
//   if (last_error != cudaSuccess) {
//     cerr << "A CUDA error slipped by undetected from a component of DynSLAM!" << endl;
// 
//     // Trigger the regular error response.
//     ITMSafeCall(last_error);
//   }
//   std::cout << "the number of active map: "<< static_scene_->GetLocalActiveMapNumber()<<std::endl; 
//   std::cout << "the number of map: " << static_scene_->GetMapNumber()<<std::endl;
  current_frame_no_++;
}

void DenseSlam::SaveStaticMap(const std::string &dataset_name, const std::string &depth_name) const {
  string target_folder = EnsureDumpFolderExists(dataset_name);
  string map_fpath = utils::Format("%s/static-%s-mesh-%06d-frames.obj",
                                   target_folder.c_str(),
                                   depth_name.c_str(),
                                   current_frame_no_);
  cout << "Saving full static map to: " << map_fpath << endl;
  static_scene_->SaveSceneToMesh(map_fpath.c_str());
}

std::string DenseSlam::EnsureDumpFolderExists(const string &dataset_name) const {
  // TODO-LOW(andrei): Make this more cross-platform and more secure.
  string today_folder = utils::GetDate();
  string target_folder = "mesh_out/" + dataset_name + "/" + today_folder;
  if(system(utils::Format("mkdir -p '%s'", target_folder.c_str()).c_str())) {
    throw runtime_error(utils::Format("Could not create directory: %s", target_folder.c_str()));
  }

  return target_folder;
}

}
