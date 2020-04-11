#include "DenseSLAMGUI.h"
#include "InfiniTamDriver.h"
#include <sys/types.h>
#include <dirent.h>
#include "Utils.h"

const std::string kKittiOdometry = "kitti-odometry";
const std::string kKittiTracking = "kitti-tracking";
const std::string kKitti         = "kitti";
const std::string ORBVocPath = "../src/ORB-SLAM2-API-M/Vocabulary/ORBvoc.txt";

//将需要的命令行参数使用gflags的宏:DEFINE_xxxx(变量名，默认值，help-string)定义在文件当中，全局变量，存储在区全局区中

//一些文件的读取
DEFINE_string(dataset_root, "", "The root folder of the dataset or dataset sequence to use.");
DEFINE_int32(sensor_type, 0, "The sensor type.");
DEFINE_int32(dataset_type, 0, "Determine to use which dataset.");

//常量路径
DEFINE_string(ORBvoc, ORBVocPath, "the path to load the ORBvoc.txt");
DEFINE_string(strSettingFile, "", "the path to load the setting file for ORBSLAM");

//一些不怎么需要改变的变量
DEFINE_bool(direct_refinement, false, "Whether to refine motion estimates for other cars computed sparsely with RANSAC using a semidense direct image alignment method.");
DEFINE_bool(use_dispnet, false, "Whether to use DispNet depth maps. Otherwise ELAS is used.");
DEFINE_bool(record, true, "Whether to record a video of the GUI and save it to disk. Using an external program usually leads to better results, though.");
DEFINE_double(scale, 1.0, "Whether to run in reduced-scale mode. Used for experimental purposes. "
                          "Requires the (odometry) sequence to have been preprocessed using the "
                          "'scale_sequence.py' script.");
DEFINE_int32(evaluation_delay, 0, "How many frames behind the current one should the evaluation be "
                                  "performed. A value of 0 signifies always computing the "
                                  "evaluation metrics on the most recent frames. Useful for "
                                  "measuring the impact of the regularization, which ``follows'' "
                                  "the camera with a delay of 'min_decay_age'. Warning: does not "
                                  "support dynamic scenes.");
DEFINE_bool(close_on_complete, true, "Whether to shut down automatically once 'frame_limit' is reached.");

// Note: the [RIP] tags signal spots where I wasted more than 30 minutes debugging a small, sillyzhe
// issue, which could easily be avoided in the future.

// Handle SIGSEGV and its friends by printing sensible stack traces with code snippets.
//backward::SignalHandling sh;
namespace SparsetoDense{
  
using namespace std;
using namespace SparsetoDense;
using namespace SparsetoDense::eval;
using namespace SparsetoDense::utils;


/// \brief Reads the projection and transformation matrices for a KITTI-odometry sequence.
/// \note P0 = left-gray, P1 = right-gray, P2 = left-color, P3 = right-color
/// 读取图片的内参
void ReadOdometryCalibration(const string &fpath,
                                  Eigen::Matrix<double, 3, 4> &left_gray_proj,
			          Eigen::Vector2i &frame_size,
                                  double downscale_factor) {
  cv::FileStorage calib(fpath.c_str(), cv::FileStorage::READ);
  if(!calib.isOpened()){
     throw runtime_error(SparsetoDense::utils::Format("Could not open calibration file: [%s]", fpath.c_str()));
  }
  
  left_gray_proj << calib["Camera.fx"],        0.0,         calib["Camera.cx"], 0.0,
                           0.0,        calib["Camera.fy"],  calib["Camera.cy"], 0.0,
                           0.0,                0.0,               1.0,          0.0;
			   
  frame_size << calib["Camera.width"], calib["Camera.height"];
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
//   if(sensor_type == Input::STEREO){
//     FLAGS_use_dispnet = true; 
//   }
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
  }
  else if(dataset_type == Input::ICLNUIM){
    input_config = Input::ICLNUIMOdometryConfig();
  }
  else {
    runtime_error("Unspported dataset type!");
  }

  ///这里的参数用于KITTI
  Eigen::Matrix34d left_color_proj;
  Eigen::Vector2i frame_size;
  
  if(dataset_type == SparsetoDense::Input::KITTI){
     // ReadKittiOdometryCalibration读取各个相机的内参,以及读取图片的大小，方便为后面分配内存
     ReadOdometryCalibration(dataset_root + "/" + input_config.calibration_fname,
                                  left_color_proj, frame_size, downscale_factor);

     cout << "Read calibration from KITTI-style data..." << endl
          << "Frame size: " << frame_size << endl
          << "Proj: " << endl << left_color_proj << endl;
  }
  else if(dataset_type == SparsetoDense::Input::TUM){
      ReadOdometryCalibration(dataset_root + "/" + input_config.calibration_fname, left_color_proj, frame_size, downscale_factor);
      cout << "Read calibration from TUM data..." << endl
           << "Frame size: " << frame_size << endl
	   << "Proj: " << endl << left_color_proj << endl;
  }
  else if(dataset_type == SparsetoDense::Input::ICLNUIM){
      ReadOdometryCalibration(dataset_root + "/" + input_config.calibration_fname, left_color_proj, frame_size, downscale_factor);
      cout << "Read calibration from ICL-NUIM data..." << endl
           << "Frame size: " << frame_size << endl
	   << "Proj: " << endl << left_color_proj << endl;
  }
  else{
      runtime_error("Currently not supported the dataset type !");
  }
  
  string param_path = dataset_root + "/" + input_config.calibration_fname;
  cv::FileStorage params(param_path.c_str(), cv::FileStorage::READ);
  
  int use_voxel_decay =  params["voxel_decay"];
  VoxelDecayParams voxel_decay_params(
      (bool)use_voxel_decay,
      params["min_decay_age"],
      params["max_decay_weight"]
  );
  
  int use_online_correction = params["online_correction"];
  OnlineCorrectionParams online_correction_params{
      (bool)use_online_correction,
      params["Online_correction_num"]
  };
  
  int use_post_processing = params["post_processing"];
  PostPocessParams post_processing_params{
     (bool)use_post_processing,
     params["filter_threshold"],
     params["filter_area"]
  };
  
  int save_raycastdepth = params["raycast_depth"];
  int enable_compositing_dense = params["compositing_dense"];
  SaveRaycastDepthParams raycast_depth_params{
    (bool)save_raycastdepth,
    (bool)enable_compositing_dense,
    params["delay_num"]
  };
  
  int save_raycastrgb = params["raycast_rgb"];
  int enable_compositing_dense_rgb = params["compositing_dense_rgb"];
  SaveRaycastRGBParams raycast_rgb_params{
    (bool)save_raycastrgb,
    (bool)enable_compositing_dense_rgb,
    params["delay_num_rgb"]
  };
  
  int use_depth_weighting = params["depth_weighting"];
  ITMLib::Engine::WeightParams  depth_weighting_params;
  depth_weighting_params.depthWeighting = (bool)use_depth_weighting;
  depth_weighting_params.maxNewW = params["maxNewW"];
  depth_weighting_params.maxDistance = params["maxDistance"];
  
  int use_orbslam_vo = params["orbslam_vo"];
  int use_orbslam_viewer = params["orbslam_viewer"];
  
  int frame_offset = params["frame_offset"];
  
  //基线
  float baseline_mm = params["Camera.bf"];
  float focal_length_px = left_color_proj(0, 0);
  
  input_config.max_depth_m = params["ThFarDepth"];
  input_config.min_depth_m = params["ThCloseDepth"];
  
  //depth = baseline*fx/disparity
  StereoCalibration stereo_calibration(baseline_mm, focal_length_px);

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
  

   if(FLAGS_use_dispnet){
      input_config.read_depth = false; //直接读取的是深度
   }
   else{
      input_config.read_depth = true; //直接读取的是深度图
   }
  
  FLAGS_strSettingFile = dataset_root + "/" + input_config.calibration_fname;
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
      depth_weighting_params);  
   
  //与ORB_SLAM2的接口
  ORB_SLAM2::drivers::OrbSLAMDriver *orbDriver = new ORB_SLAM2::drivers::OrbSLAMDriver(
      FLAGS_ORBvoc,
      FLAGS_strSettingFile,
      sensor_type_orbslam,
      (bool)use_orbslam_viewer
  );
  

  Vector2i input_shape((*input_out)->GetRgbSize().width, (*input_out)->GetRgbSize().height);
  
  //将所有的对象集成到该系统中
  *dense_slam_out = new DenseSlam(
      itmDriver,
      orbDriver,
      input_shape,
      left_color_proj.cast<float>(),
      baseline_mm,
      FLAGS_direct_refinement,
      post_processing_params,
      online_correction_params,
      raycast_depth_params,
      raycast_rgb_params,
      (bool)use_orbslam_vo
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
  
  string param_path = dataset_root + "/" + input->GetConfig().calibration_fname;
  cv::FileStorage params(param_path.c_str(), cv::FileStorage::READ);
  
  int use_GUI = params["GUI_SHOW"];
  if(!(bool)use_GUI){
    int count = 0;
    int totalcount = params["frame_limit"];
    while(input->HasMoreImages() || input->GetFrameIndex() < 980){
      SparsetoDense::utils::Tic("DenseSLAM frame");
      cout << endl << "[Starting frame " << dense_slam->GetCurrentFrameNo() << "]" << endl;
      dense_slam->ProcessFrame(input);
      int64_t frame_time_ms = SparsetoDense::utils::Toc(true);
      float fps = 1000.0f / static_cast<float>(frame_time_ms);
      cout << "[Finished frame " << dense_slam->GetCurrentFrameNo()-1 << " in " << frame_time_ms
           << "ms @ " << setprecision(4) << fps << " FPS (approx.)]"
           << endl;
      if(count > totalcount){
         break;
      }
      count ++;
     }

     dense_slam->SaveTUMTrajectory("/home/hansry/DenseSLAM-Global-Consistency-h/data/result.txt");
     cout << "No more images, Bye!" << endl;
      
     if(!dense_slam->todoList.empty()){
       for(int mapNO = 0; mapNO < dense_slam->todoList.size(); mapNO++){
	 int mapNum = dense_slam->todoList[mapNO].dataId;
	 ITMLib::Engine::ITMLocalMap* currLocalMap = dense_slam->GetStaticScene()->GetMapManager()->getLocalMap(mapNum);
	 dense_slam->SaveStaticMap(input->GetDatasetIdentifier(), currLocalMap, mapNum);   
       }
      }
     getchar();
  }
  else{
     int is_autoplay = params["auto_play"];
     int is_chase_cam = params["chase_cam"];
     int is_view_raycast_depth = params["view_ew_raycast_depth"];
  
     SparsetoDense::gui::ParamSLAMGUI paramGUI;
     paramGUI.autoplay = bool(is_autoplay);
     paramGUI.chase_cam = bool(is_chase_cam);
     paramGUI.viewRaycastDepth = (bool)is_view_raycast_depth;
     paramGUI.frame_limit = params["frame_limit"];
     paramGUI.close_on_complete = FLAGS_close_on_complete;
     paramGUI.evaluation_delay = FLAGS_evaluation_delay;
     paramGUI.record = FLAGS_record;
     SparsetoDense::gui::PangolinGui pango_gui(dense_slam, input, paramGUI);
     pango_gui.Run();
  }
  delete dense_slam;
  delete input;
}

