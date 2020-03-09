#include "DenseSLAMGUI.h"
#include "InfiniTamDriver.h"
#include <sys/types.h>
#include <dirent.h>

const std::string kKittiOdometry = "kitti-odometry";
const std::string kKittiTracking = "kitti-tracking";
const std::string kKitti         = "kitti";
const std::string ORBVocPath = "../src/ORB-SLAM2-API-M/Vocabulary/ORBvoc.txt";

//将需要的命令行参数使用gflags的宏:DEFINE_xxxx(变量名，默认值，help-string)定义在文件当中，全局变量，存储在区全局区中
DEFINE_string(dataset_root, "", "The root folder of the dataset or dataset sequence to use.");
DEFINE_int32(sensor_type, 0, "The sensor type.");
DEFINE_int32(dataset_type, 0, "Determine to use which dataset.");
DEFINE_string(ORBvoc, ORBVocPath, "the path to load the ORBvoc.txt");
DEFINE_string(strSettingFile, "", "the path to load the setting file for ORBSLAM");
DEFINE_int32(frame_offset, 0, "The frame index from which to start reading the dataset sequence.");
DEFINE_int32(frame_limit, 0, "How many frames to process in auto mode. 0 = no limit.");
DEFINE_bool(voxel_decay, true, "Whether to enable map regularization via voxel decay (a.k.a. voxel garbage collection).");
DEFINE_int32(min_decay_age, 150, "The minimum voxel *block* age for voxels within it to be eligible for deletion (garbage collection)."); //kitti: 5
DEFINE_int32(max_decay_weight, 0.5, "The maximum voxel weight for decay. Voxels which have accumulated more than this many measurements will not be removed."); //kitti: 2
DEFINE_int32(kitti_tracking_sequence_id, -1, "Used in conjunction with --dataset_type kitti-tracking.");
DEFINE_bool(direct_refinement, false, "Whether to refine motion estimates for other cars computed sparsely with RANSAC using a semidense direct image alignment method.");
// TODO-LOW(andrei): Automatically adjust the voxel GC params when depth weighting is enabled.
DEFINE_bool(use_depth_weighting, false, "Whether to adaptively set fusion weights as a function of the inverse depth (w \\propto \\frac{1}{Z}). If disabled, "
                                        "all new measurements have a constant weight of 1.");
DEFINE_double(scale, 1.0, "Whether to run in reduced-scale mode. Used for experimental purposes. "
                          "Requires the (odometry) sequence to have been preprocessed using the "
                          "'scale_sequence.py' script.");
DEFINE_bool(use_dispnet, false, "Whether to use DispNet depth maps. Otherwise ELAS is used.");
DEFINE_int32(evaluation_delay, 0, "How many frames behind the current one should the evaluation be "
                                  "performed. A value of 0 signifies always computing the "
                                  "evaluation metrics on the most recent frames. Useful for "
                                  "measuring the impact of the regularization, which ``follows'' "
                                  "the camera with a delay of 'min_decay_age'. Warning: does not "
                                  "support dynamic scenes.");
DEFINE_bool(close_on_complete, true, "Whether to shut down automatically once 'frame_limit' is reached.");
DEFINE_bool(record, true, "Whether to record a video of the GUI and save it to disk. Using an "
                           "external program usually leads to better results, though.");
DEFINE_bool(chase_cam, true, "Whether to preview the reconstruction in chase cam mode, following "
                             "the camera from a third person view.");
DEFINE_int32(fusion_every, 1, "Fuse every kth frame into the map. Used for evaluating the system's "
                              "behavior under reduced temporal resolution.");
DEFINE_bool(autoplay, false, "Whether to start with autoplay enabled. Useful for batch experiments.");
DEFINE_bool(useOrbSLAMViewer, false, "Whether to launch the GUI of ORBSLAM2.");
DEFINE_bool(viewRaycastDepth, true, "Whether to view the raycast depth.");

// Note: the [RIP] tags signal spots where I wasted more than 30 minutes debugging a small, sillyzhe
// issue, which could easily be avoided in the future.

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
//backward::SignalHandling sh;
namespace SparsetoDense{
  
using namespace std;

using namespace SparsetoDense;
using namespace SparsetoDense::eval;
using namespace SparsetoDense::utils;
  
Eigen::Matrix<double, 3, 4> ReadProjection(const string &expected_label, istream &in, double downscale_factor) {
  Eigen::Matrix<double, 3, 4> matrix;
  string label;
  in >> label;
  assert(expected_label == label && "Unexpected token in calibration file.");

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> matrix(row, col);
    }
  }

// The downscale factor is used to adjust the intrinsic matrix for low-res input.
//  cout << "Adjusting projection matrix for scale [" << downscale_factor << "]." << endl;
//  matrix *= downscale_factor;
//  matrix(2, 2) = 1.0;

  return matrix;
};

/// \brief Reads the projection and transformation matrices for a KITTI-odometry sequence.
/// \note P0 = left-gray, P1 = right-gray, P2 = left-color, P3 = right-color
/// 读取图片的内参
void ReadKittiOdometryCalibration(const string &fpath,
                                  Eigen::Matrix<double, 3, 4> &left_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &right_gray_proj,
                                  Eigen::Matrix<double, 3, 4> &left_color_proj,
                                  Eigen::Matrix<double, 3, 4> &right_color_proj,
                                  Eigen::Matrix4d &velo_to_left_cam,
                                  double downscale_factor) {
  static const string kLeftGray = "P0:";
  static const string kRightGray = "P1:";
  static const string kLeftColor = "P2:";
  static const string kRightColor = "P3:";
  ifstream in(fpath);
  if (! in.is_open()) {
    throw runtime_error(SparsetoDense::utils::Format("Could not open calibration file: [%s]", fpath.c_str()));
  }

  //将相机的参数保存成Matrix形式
  left_gray_proj = ReadProjection(kLeftGray, in, downscale_factor);
  right_gray_proj = ReadProjection(kRightGray, in, downscale_factor);
  left_color_proj = ReadProjection(kLeftColor, in, downscale_factor);
  right_color_proj = ReadProjection(kRightColor, in, downscale_factor);

  string dummy;
  in >> dummy;
  if (dummy != "Tr:") {
    // Looks like a kitti-tracking sequence
    std::getline(in, dummy); // skip to the end of current line

    in >> dummy;
    assert(dummy == "Tr_velo_cam");
  }

  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 4; ++col) {
      in >> velo_to_left_cam(row, col);
    }
  }
  velo_to_left_cam(3, 0) = 0.0;
  velo_to_left_cam(3, 1) = 0.0;
  velo_to_left_cam(3, 2) = 0.0;
  velo_to_left_cam(3, 3) = 1.0;
}

void ReadTUMOdometryCalibration(const string &fpath,
                                Eigen::Matrix<double,3,4> &left_gray_proj,
				double downscale_factor){
	static const string kLeftColor = "RGB:";
	ifstream in(fpath);
	if(! in.is_open()){
	  throw runtime_error(SparsetoDense::utils::Format("Could not open calibration file: [%s]", fpath.c_str()));
	}
        left_gray_proj = ReadProjection(kLeftColor, in, downscale_factor);			  
}

/// \brief Probes a dataset folder to find the frame dimentsions.
/// \note This is useful for pre-allocating buffers in the rest of the pipeline.
/// \returns A (width, height), i.e., (cols, rows)-style dimension.
/// 返回图像的大小
Eigen::Vector2i GetFrameSize(const string &dataset_root, const Input::Config &config) {
  string lc_folder = dataset_root + "/" + config.left_color_folder;
  DIR *pDir;
  struct dirent* ptr;
  if(!(pDir = opendir(lc_folder.c_str()))){
    runtime_error("folder doesnt't exist in GetFRameSize image");
  }
  std::string fileName = "";
  while((ptr = readdir(pDir)) != 0){
    if(strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
      fileName = ptr->d_name;
      break;
    }
  }
  stringstream lc_fpath_ss;
  lc_fpath_ss << lc_folder << "/" << fileName;
  cv::Mat frame = cv::imread(lc_fpath_ss.str());
  return Eigen::Vector2i(
      frame.cols * 1.0f / FLAGS_scale,
      frame.rows * 1.0f / FLAGS_scale
  );
}

/// \brief Constructs a DynSLAM instance to run on a KITTI Odometry dataset sequence, using liviso2
///        for visual odometry.
/// \brief 构建DenseSLAM对象实例以运行KITTI里程计，libviso为视觉里程计(VO)中的开源算法，包括特征点匹配，位姿估计等
void BuildDenseSlamOdometry(const string &dataset_root,
                               DenseSlam **dense_slam_out,
                               Input **input_out,
			       Input::eSensor sensor_type,
			       ORB_SLAM2::System::eSensor sensor_type_orbslam,
			       Input::eDatasetType dataset_type) {
  Input::Config input_config;
  double downscale_factor = FLAGS_scale;//scale = 1(default)
  if (downscale_factor > 1.0 || downscale_factor <= 0.0) {
    throw runtime_error("Scaling factor must be > 0 and <= 1.0.");
  }
  float downscale_factor_f = static_cast<float>(downscale_factor);//强制转换为float类型

  //当使用深度的时候，会偏向于使用use_dispnet
  if(sensor_type == Input::STEREO){
    FLAGS_use_dispnet = true; 
  }
  //Odometry 判断是否使用Dispnet以及设置保存深度图文件夹的名称
  if(dataset_type == Input::KITTI){
     if (FLAGS_use_dispnet) {
        input_config = Input::KittiOdometryDispnetConfig();
     }
     else {
        input_config = Input::KittiOdometryConfig();
     }
  }
  else if(dataset_type == Input::TUM){
    input_config = Input::TUMOdometryConfig();
//     FLAGS_chase_cam = true;
  }
  else if(dataset_type == Input::ICLNUIM){
    input_config = Input::ICLNUIMOdometryConfig();
//     FLAGS_chase_cam = true;
  }
  else {
    runtime_error("Unspported dataset type!");
  }

  ///这里的参数用于KITTI
  Eigen::Matrix34d left_gray_proj;
  Eigen::Matrix34d right_gray_proj;
  Eigen::Matrix34d left_color_proj;
  Eigen::Matrix34d right_color_proj;
  Eigen::Matrix4d velo_to_left_gray_cam;//雷达到灰度相机0的转换关系

  // 读取图片的大小，方便为后面分配内存
  Eigen::Vector2i frame_size = GetFrameSize(dataset_root, input_config);
  if(dataset_type == SparsetoDense::Input::KITTI){
     // ReadKittiOdometryCalibration读取各个相机的内参以及雷达到灰度相机0的转换关系,挺重要的这里
     ReadKittiOdometryCalibration(dataset_root + "/" + input_config.calibration_fname,
                                  left_gray_proj, right_gray_proj, left_color_proj, right_color_proj,
                                  velo_to_left_gray_cam, downscale_factor);

     cout << "Read calibration from KITTI-style data..." << endl
          << "Frame size: " << frame_size << endl
          << "Proj (left, gray): " << endl << left_gray_proj << endl
          << "Proj (right, gray): " << endl << right_gray_proj << endl
          << "Proj (left, color): " << endl << left_color_proj << endl
          << "Proj (right, color): " << endl << right_color_proj << endl
          << "Velo: " << endl << velo_to_left_gray_cam << endl;
  }
  else if(dataset_type == SparsetoDense::Input::TUM){
      ReadTUMOdometryCalibration(dataset_root + "/" + input_config.calibration_fname, left_color_proj, downscale_factor);
      left_gray_proj = left_color_proj;
      right_gray_proj = left_color_proj;
      right_color_proj = left_color_proj;
      velo_to_left_gray_cam.setOnes();
      cout << "Read calibration from TUM data..." << endl
           << "Frame size: " << frame_size << endl
	   << "Proj: " << endl << left_color_proj << endl;
  }
  else if(dataset_type == SparsetoDense::Input::ICLNUIM){
      ReadTUMOdometryCalibration(dataset_root + "/" + input_config.calibration_fname, left_color_proj, downscale_factor);
      left_gray_proj = left_color_proj;
      right_gray_proj = left_color_proj;
      right_color_proj = left_color_proj;
      velo_to_left_gray_cam.setOnes();
      cout << "Read calibration from ICL-NUIM data..." << endl
           << "Frame size: " << frame_size << endl
	   << "Proj: " << endl << left_color_proj << endl;
  }
  else{
      runtime_error("Currently not supported the dataset type !");
  }
  
  VoxelDecayParams voxel_decay_params(
      FLAGS_voxel_decay,
      FLAGS_min_decay_age,
      FLAGS_max_decay_weight
  );

  //frame_offset=0 (default)
  int frame_offset = FLAGS_frame_offset;
  
  //基线
  float baseline_m = 0.537150654273f;
  float focal_length_px = left_gray_proj(0, 0);
  
  //depth = baseline*fx/disparity
  StereoCalibration stereo_calibration(baseline_m, focal_length_px);

  //输入，包含文件夹路径，内参，以及传感器的类型
  *input_out = new Input(
      dataset_root,
      sensor_type,
      dataset_type,
      input_config,
      nullptr,          // set the depth provider later
      frame_size,
      stereo_calibration,
      frame_offset,
      downscale_factor);
  
  if(sensor_type == Input::STEREO){
    input_config.read_depth = false; //读取的是视差图，需要将其转成深度图
  }
  else{
    if(FLAGS_use_dispnet){
       input_config.read_depth = false; //直接读取的是深度
    }
    else{
       input_config.read_depth = true; //直接读取的是深度图
    }

  }
  
  FLAGS_strSettingFile = dataset_root + "/orbslam_param.yaml";
  //通过预测深度，可以通过从disk读取，也可以实时进行计算
  DepthProvider *depth = new PrecomputedDepthProvider(
      *input_out,
      dataset_root + "/" + input_config.depth_folder,
      input_config.depth_fname_format,
      input_config.read_depth,
      input_config.min_depth_m,
      input_config.max_depth_m
  );
  (*input_out)->SetDepthProvider(depth);

  // 注意：InfiniTAM即使在深度/rgb输入的大小(如校准文件中所指定的)与输入图像的实际大小存在差异时仍能正常工作(但它会破坏预览)。
  // InfiniTAM中的设置
  ITMLib::Objects::ITMLibSettings *driver_settings = new ITMLib::Objects::ITMLibSettings();
//   driver_settings->groundTruthPoseFpath = dataset_root + "/" + input_config.odometry_fname;
//   driver_settings->groundTruthPoseOffset = frame_offset;
//   if (FLAGS_dynamic_weights) {
//     driver_settings->sceneParams.maxW = driver_settings->maxWDynamic;
//   }

  //与InfiniTAM的接口
  drivers::InfiniTamDriver *itmDriver = new InfiniTamDriver(
      driver_settings,
      SparsetoDense::drivers::CreateItmCalib(left_color_proj, frame_size),
      SparsetoDense::drivers::ToItmVec((*input_out)->GetRgbSize()),
      SparsetoDense::drivers::ToItmVec((*input_out)->GetDepthSize()),
      voxel_decay_params,
      FLAGS_use_depth_weighting);
   
  //与ORB_SLAM2的接口
  ORB_SLAM2::drivers::OrbSLAMDriver *orbDriver = new ORB_SLAM2::drivers::OrbSLAMDriver(
      FLAGS_ORBvoc,
      FLAGS_strSettingFile,
      sensor_type_orbslam,
      FLAGS_useOrbSLAMViewer
  );
  
//   const string seg_folder = dataset_root + "/" + input_config.segmentation_folder;
//   auto segmentation_provider = new instreclib::segmentation::PrecomputedSegmentationProvider(
//                                     seg_folder, frame_offset, static_cast<float>(downscale_factor));

  VisualOdometryStereo::parameters sf_params;
  // TODO(andrei): The main VO (which we're not using viso2 for, at the moment (June '17) and the
  // "VO" we use to align object instance frames have VASTLY different requirements, so we should
  // use separate parameter sets for them.
  sf_params.base = baseline_m;
  sf_params.match.nms_n = 3;          // Optimal from KITTI leaderboard: 3 (also the default)
  sf_params.match.half_resolution = 0;
  sf_params.match.multi_stage = 1;    // Default = 1 (= 0 => much slower)
  sf_params.match.refinement = 1;     // Default = 1 (per-pixel); 2 = sub-pixel, slower
  sf_params.ransac_iters = 500;       // Default = 200
  sf_params.inlier_threshold = 2.0;   // Default = 2.0 (insufficient for e.g., hill sequence)
//  sf_params.inlier_threshold = 2.7;   // May be required for the hill sequence
  sf_params.bucket.max_features = 15;    // Default = 2
  // VO is computed using the color frames.
  sf_params.calib.cu = left_color_proj(0, 2);
  sf_params.calib.cv = left_color_proj(1, 2);
  sf_params.calib.f  = left_color_proj(0, 0);

  //进行光流估计，以检测动态物体
  auto sparse_sf_provider = new instreclib::VisoSparseSFProvider(sf_params);
  
  Vector2i input_shape((*input_out)->GetRgbSize().width, (*input_out)->GetRgbSize().height);
  
  //将所有的对象集成到该系统中
  *dense_slam_out = new DenseSlam(
      itmDriver,
      orbDriver,
      sparse_sf_provider,
      input_shape,
      left_color_proj.cast<float>(),
      right_color_proj.cast<float>(),
      baseline_m,
      FLAGS_direct_refinement,
      FLAGS_fusion_every,
      FLAGS_min_decay_age
  );
}

} // namespace SparsetoDense

int main(int argc, char **argv) {
  gflags::SetUsageMessage("The GUI for Sparse to Dense simultaneous localization and mapping (Sparse to Dense).");
  //argc为参数个数，argv为具体参数，
  //第三个参数设为true：该函数处理完成后，argv中只保留argv[1],argc会被设置为1
  //第三个参数设为false:则argc和argv会被保留，但是注意函数会调整argv中的参数
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  
  //通过使用gflags，在后续代码中可以使用FLAGS_变量名访问对应的命令行参数，而不用像普通的传入argv等参数需要用argv[0],argv[1]等来调用。
  const string dataset_root = FLAGS_dataset_root;
  const int sensor_type_int = FLAGS_sensor_type;
  const int dataset_type_int = FLAGS_dataset_type;
  if (dataset_root.empty()) {
    std::cerr << "The --dataset_root=<path> flag must be set." << std::endl;
    return -1;
  }
  
  SparsetoDense::DenseSlam *dense_slam;
  SparsetoDense::Input *input;
  SparsetoDense::Input::eSensor sensor_type;
  SparsetoDense::Input::eDatasetType dataset_type;
  
  ORB_SLAM2::System::eSensor sensor_type_orbslam;
  if (sensor_type_int == 0){
    sensor_type = SparsetoDense::Input::MONOCULAR;
    sensor_type_orbslam = ORB_SLAM2::System::MONOCULAR;
    cout << "Monocular" << endl;
  }
  else if (sensor_type_int == 1 ){
    sensor_type = SparsetoDense::Input::STEREO;
    sensor_type_orbslam = ORB_SLAM2::System::STEREO;
    cout << "STEREO" << endl;
  }
  else{
    sensor_type = SparsetoDense::Input::RGBD;
    sensor_type_orbslam = ORB_SLAM2::System::RGBD;
    cout << "RGB-D" << endl;
  }
  
  if(dataset_type_int == 0){
    dataset_type = SparsetoDense::Input::KITTI;
  }
  else if (dataset_type_int == 1){
    dataset_type = SparsetoDense::Input::TUM;
  }
  else if (dataset_type_int == 2){
    dataset_type = SparsetoDense::Input::ICLNUIM;
  }
  else{
      runtime_error("Unspported dataset mode !");
//    dataset_type = SparsetoDense::Input::EUROC;
  }
  BuildDenseSlamOdometry(dataset_root, &dense_slam, &input, sensor_type, sensor_type_orbslam, dataset_type);
  
  SparsetoDense::gui::ParamSLAMGUI paramGUI;
  paramGUI.autoplay = FLAGS_autoplay;
  paramGUI.chase_cam = FLAGS_chase_cam;
  paramGUI.close_on_complete = FLAGS_close_on_complete;
  paramGUI.evaluation_delay = FLAGS_evaluation_delay;
  paramGUI.frame_limit = FLAGS_frame_limit;
  paramGUI.record = FLAGS_record;
  paramGUI.viewRaycastDepth = FLAGS_viewRaycastDepth;
  SparsetoDense::gui::PangolinGui pango_gui(dense_slam, input, paramGUI);
  pango_gui.Run();

  delete dense_slam;
  delete input;
}

