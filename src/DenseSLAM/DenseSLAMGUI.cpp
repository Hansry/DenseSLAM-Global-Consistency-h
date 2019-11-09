#include "DenseSLAMGUI.h"

namespace SparsetoDense{
namespace gui{

using namespace SparsetoDense::utils;

void PangolinGui::DrawPose(long int current_time_ms){
    //启动main_view_
    main_view_->Activate();

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);

    glMatrixMode(GL_PROJECTION);
    proj_.Load();

    glMatrixMode(GL_MODELVIEW);
    pane_cam_->GetModelViewMatrix().Load();

    auto phist = dense_slam_->GetPoseHistory();

    // Make the poses a little bit more visible (set to > 0.0f to enable).
    float frustum_root_cube_scale = 0.00f;

    //对所有位姿进行绘制
    const float kMaxFrustumScale = 0.66;
    Eigen::Vector3f color_white(1.0f, 1.0f, 1.0f);
    for (int i = 0; i < static_cast<int>(phist.size()) - 1; ++i) {
      float frustum_scale = max(0.15f, kMaxFrustumScale - 0.05f * (phist.size() - 1 - i));
      DrawPoseFrustum(phist[i], color_white, frustum_scale, frustum_root_cube_scale);
    }

    //对最新的位姿颜色进行加深
    if (! phist.empty()) {
      // Highlight the most recent pose.
      Eigen::Vector3f glowing_green(
          0.5f, 0.5f + static_cast<float>(sin(current_time_ms / 250.0) * 0.5 + 0.5) * 0.5f, 0.5f);
      DrawPoseFrustum(phist[phist.size() - 1], glowing_green, kMaxFrustumScale, frustum_root_cube_scale);
    }
}

void PangolinGui::DrawPoseFrustum(const Eigen::Matrix4f& pose, const Eigen::Vector3f& color, float frustum_scale, float frustum_root_cube_scale) const{
    glPushMatrix();
    
    //inv_pose = Tw_c
    Eigen::Matrix4f inv_pose = pose.inverse();
    glMultMatrixf(inv_pose.data());
    pangolin::glDrawColouredCube(-frustum_root_cube_scale, frustum_root_cube_scale);
    glPopMatrix();

    Eigen::Matrix34f projection = dense_slam_->GetLeftRgbProjectionMatrix();//投影矩阵
    const Eigen::Matrix3f Kinv = projection.block(0, 0, 3, 3).inverse();//投影矩阵的逆
    glColor3f(color(0), color(1), color(2));
    pangolin::glDrawFrustum(Kinv, width_, height_, inv_pose, frustum_scale);
}

void PangolinGui::Run(){
    // Default hooks for exiting (Esc) and fullscreen (tab).
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glColor3f(1.0, 1.0, 1.0);
      pangolin::GlFont &font = pangolin::GlFont::I();

      //自动运行
      if (autoplay_->Get()) {
        if (paramGUI.frame_limit == 0 || dense_slam_->GetCurrentFrameNo() < paramGUI.frame_limit) {
          ProcessFrame();
	  //cout<<dyn_slam_->GetDepthPreview()->clone()<<endl;
	  dense_slam_->orbslam_static_scene_trackRGBD(dense_slam_->GetRgbPreview()->clone(),
               dense_slam_->GetDepthPreview()->clone()/256.0, (double)dense_slam_->GetCurrentFrameNo());
        }
        else {
          cerr << "Warning: reached autoplay limit of [" << paramGUI.frame_limit << "]. Stopped."<< endl;
          *autoplay_ = false;
          if (paramGUI.close_on_complete) {
            cerr << "Closing as instructed. Bye!" << endl;
            pangolin::QuitAll();
            return;
          }
        }
      }

      long time_ms = utils::GetTimeMs();

      main_view_->Activate(*pane_cam_);
      glEnable(GL_DEPTH_TEST);
      glColor3f(1.0f, 1.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      glDisable(GL_DEPTH_TEST);
      glDepthMask(false);

      //是否在chase cam模式下预览重建，第三视角
      if (paramGUI.chase_cam) 
      {
        Eigen::Matrix4f cam_mv = dense_slam_->GetPose();
        pangolin::OpenGlMatrix pm(cam_mv);
        pm =
            // Good for odo 05
             pangolin::OpenGlMatrix::RotateY(M_PI * 0.5 * 0.05f) *
             pangolin::OpenGlMatrix::RotateX(M_PI * 0.5 * 0.03f) *
             pangolin::OpenGlMatrix::Translate(-0.5, 1.0, 15.0) *
             pm;
         pane_cam_->SetModelViewMatrix(pm);
	 orb_pane_cam_->SetModelViewMatrix(pm);
      }
//       else{
// 	Eigen::Matrix4f cam_mv = dense_slam_->GetPose().inverse();
//         pangolin::OpenGlMatrix pm(cam_mv);
//         pm =
//             // Good for odo 05
//              pangolin::OpenGlMatrix::RotateY(M_PI * 0.5 * 0.05f) *
//              pangolin::OpenGlMatrix::RotateX(M_PI * 0.5 * 0.03f) *
//              pangolin::OpenGlMatrix::Translate(-0.5, 1.0, 15.0) *
//              pm;
//          pane_cam_->SetModelViewMatrix(pm);
// 	 orb_pane_cam_->SetModelViewMatrix(pm);
//       }

      int evaluated_frame_idx = dense_slam_->GetCurrentFrameNo() - 1 - paramGUI.evaluation_delay;
      if (evaluated_frame_idx > 0) {
        //auto velodyne = dyn_slam_->GetEvaluation()->GetVelodyneIO();
//         int input_frame_idx = dense_slam_input_->GetFrameOffset() + evaluated_frame_idx;
// 
//         Eigen::Matrix4f epose = dense_slam_->GetPoseHistory()[evaluated_frame_idx + 1];
//         auto pango_pose = pangolin::OpenGlMatrix::ColMajor4x4(epose.data());
// 
        bool enable_compositing = (paramGUI.evaluation_delay == 0);
// 	
// 	//通过模型光线投影回来的深度图
// // 	const unsigned char *synthesized_depthmap = nullptr;
// //         synthesized_depthmap = dense_slam_->GetStaticMapRaycastDepthPreview(pango_pose, enable_compositing);
//         auto input_depthmap = shared_ptr<cv::Mat1s>(nullptr);
//         auto input_rgb = shared_ptr<cv::Mat3b>(nullptr);
//         dense_slam_input_->GetFrameCvImages(input_frame_idx, input_rgb, input_depthmap);
// 
//         /// Result of diffing our disparity maps (input and synthesized).
//         uchar diff_buffer[width_ * height_ * 4];
// 	//void* memset(void *str, int c, size_t n)复制字符c(一个无符号字符)到参数str所指向的字符串的前n个字符
// 	//diff_buffer为要填充的内存块，‘\0’为要被设置的值，sizeof(uchar)*width_*height_*4为要被设置为该值的字节数
//         memset(diff_buffer, '\0', sizeof(uchar) * width_ * height_ * 4);

        bool need_lidar = false;
        const unsigned char *preview = nullptr;
        const uint delta_max_visualization = 1;
        string message;
        switch(current_lidar_vis_) {
          case kNone://
            if (paramGUI.chase_cam) {
              message = "Chase cam preview";
            }
            else {
              message = "Free cam preview";
            }
            preview = dense_slam_->GetStaticMapRaycastPreview(pane_cam_->GetModelViewMatrix(),
//                  pango_pose,
                  static_cast<PreviewType>(current_preview_type_),
		  enable_compositing);
	    pane_texture_->Upload(preview, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            DrawPose(time_ms);
            break;

//           case kInputVsLidar:
//             message = utils::Format("Input depth vs. LIDAR | delta_max = %d", delta_max_visualization);
//             need_lidar = true;
//             UploadCvTexture(*input_depthmap, *pane_texture_, false, GL_SHORT);
//             break;
// 
//           case kFusionVsLidar:
//             message = utils::Format("Fused map vs. LIDAR | delta_max = %d", delta_max_visualization);
//             need_lidar = true;
//             FloatDepthmapToShort(synthesized_depthmap, depth_preview_buffer_);
//             UploadCvTexture(depth_preview_buffer_, *pane_texture_, false, GL_SHORT);
//             break;
// 
//           case kInputVsFusion:
//             message = "Input depth vs. fusion (green = OK, yellow = input disp > fused, cyan = input disp < fused";
//             DiffDepthmaps(*input_depthmap, synthesized_depthmap, width_, height_,
//                           delta_max_visualization, diff_buffer, dense_slam_->GetStereoBaseline(),
//                           dense_slam_->GetLeftRgbProjectionMatrix()(0, 0));
//             pane_texture_->Upload(diff_buffer, GL_RGBA, GL_UNSIGNED_BYTE);
//             pane_texture_->RenderToViewport(true);
//             break;
// 
//           default:
//           case kEnd:
//             throw runtime_error("Unexpected 'current_lidar_vis_' error visualization mode.");
//             break;
         }
         // Ensures we have a blank slate for the pane's overlay text.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1.0f, 1.0f, 1.0f);
//        main_view_->Activate();

//         if (need_lidar) {
//           pane_texture_->RenderToViewport(true);
//           bool visualize_input = (current_lidar_vis_ == kInputVsLidar);
//          eval::ErrorVisualizationCallback vis_callback(
//              delta_max_visualization,
//              visualize_input,
//              Eigen::Vector2f(main_view_->GetBounds().w, main_view_->GetBounds().h),
//              lidar_vis_colors_,
//              lidar_vis_vertices_);
//           auto vis_mode = eval::SegmentedCallback::LidarAssociation::kStaticMap;
//          auto seg = dyn_slam_->GetSpecificSegmentationForEval(input_frame_idx);
//           eval::SegmentedVisualizationCallback vis_callback(
//               delta_max_visualization,
//               visualize_input,
//               Eigen::Vector2f(main_view_->GetBounds().w, main_view_->GetBounds().h),
//               lidar_vis_colors_,
//               lidar_vis_vertices_,
//               seg.get(),
// //              dyn_slam_->GetInstanceReconstructor(),
//               nullptr,
//               vis_mode
//           );
//           if (vis_mode == eval::SegmentedCallback::LidarAssociation::kDynamicReconstructed) {
//             message += " | Reconstructed dynamic objects only ";
//           }
//           else if (vis_mode == eval::SegmentedCallback::LidarAssociation::kStaticMap) {
//             message += " | Static map only";
//           }
// 
//           bool compare_on_intersection = true;
//           bool kitti_style = true;
//           eval::EvaluationCallback eval_callback(delta_max_visualization,
//                                                  compare_on_intersection,
//                                                  kitti_style);
// 
//           if (velodyne->FrameAvailable(input_frame_idx)) {
//             auto visualized_lidar_pointcloud = velodyne->ReadFrame(input_frame_idx);
//             dyn_slam_->GetEvaluation()->EvaluateDepth(visualized_lidar_pointcloud,
//                                                       synthesized_depthmap,
//                                                       *input_depthmap,
//                                                       {&vis_callback, &eval_callback});
//             auto result = eval_callback.GetEvaluation();
//             DepthResult depth_result = current_lidar_vis_ == kFusionVsLidar ? result.fused_result
//                                                                             : result.input_result;
//             message += utils::Format(" | Acc (with missing): %.3lf | Acc (ignore missing): %.3lf",
//                                      depth_result.GetCorrectPixelRatio(true),
//                                      depth_result.GetCorrectPixelRatio(false));
//             vis_callback.Render();
//           }
//         }

      //  font.Text(message).Draw(-0.90f, 0.80f);
      }
      //*/
      //font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);
      
//      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      OrbSlamMapDrawer_->GetCurrentOpenGLCameraMatrix(OrbSlamTwc_);
      orbslam_view_->Activate(*orb_pane_cam_);
      if(dense_slam_->GetCurrentFrameNo()>=1){
	glClearColor(0.0f,0.0f,0.0f,0.0f);
	if(sparseMap_show_graph_->Get()){
	  OrbSlamMapDrawer_->DrawCurrentCamera(OrbSlamTwc_);
	}
	if(sparseMap_show_keyFrame_->Get()){
	  OrbSlamMapDrawer_->DrawKeyFrames(true,true);
	}
	if(sparseMap_show_points_->Get()){
	  OrbSlamMapDrawer_->DrawMapPoints();
	}
      }
      //font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);
      
      //激活rgb_view_以便绘制
      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      /// \brief 当前帧大于1，对RGB图片进行预览。 
      /// 若启动动态模式，则显示的检测动态物体的RGB图像，不启动动态检测的时候， 则显示的是ORB特征点的RGB图片
      if(dense_slam_->GetCurrentFrameNo() >= 1 && dense_slam_->IsDynamicMode()) {
	//是否显示原来的图片预览
        if (display_raw_previews_->Get()) {
          UploadCvTexture(*(dense_slam_->GetRgbPreview()), *pane_texture_, true, GL_UNSIGNED_BYTE);
        } else {
	  //得到静态RGB图片的预览
          UploadCvTexture(*(dense_slam_->GetStaticRgbPreview()), *pane_texture_, true, GL_UNSIGNED_BYTE); 
        }
        pane_texture_->RenderToViewport(true);
      }
      else if(dense_slam_->GetCurrentFrameNo() >= 1 && !dense_slam_->IsDynamicMode()){
	cv::Mat im = dense_slam_->GetOrbSlamFrameDrawerGlobal()->DrawFrame();
	UploadCvTexture(im, *pane_texture_, true, GL_UNSIGNED_BYTE);
	pane_texture_->RenderToViewport(true);
      }
     
      if (dense_slam_->GetCurrentFrameNo() > 1 && preview_sf_->Get()) {
          PreviewSparseSF(dense_slam_->GetLatestFlow().matches, rgb_view_);
      }

      //激活depth_viewe_以进行绘制，对深度图进行显示
      depth_view_.Activate();
      glColor3f(1.0, 1.0, 1.0);
      //模型光线投影回来的深度 && 启动动态模式
      if (!display_raw_previews_->Get() && dense_slam_->IsDynamicMode()) {
	UploadCvTexture(*(dense_slam_->GetStaticDepthPreview()), *pane_texture_, false, GL_SHORT);
      }
      else {
       UploadCvTexture(*(dense_slam_->GetDepthPreview()), *pane_texture_, false, GL_SHORT);
       //UploadCvTexture(*(dense_slam_->GetStaticDepthPreview()), *pane_texture_, false, GL_SHORT);
      }
      pane_texture_->RenderToViewport(true);

      //以第一视角观察orbslam以及稠密地图的第一视角
      if(dense_slam_->GetCurrentFrameNo()>=1 && !dense_slam_->IsDynamicMode()){
	  orb_trajectory_view_->Activate(*orb_Trajectory_pane_cam_);
	  glColor3f(1.0,1.0,1.0);
	  // OrbSlamMapDrawer_->DrawTracjectory();
          orb_Trajectory_pane_cam_->Follow(OrbSlamTwc_);
	if(sparseMap_show_graph_->Get()){
	  OrbSlamMapDrawer_->DrawCurrentCamera(OrbSlamTwc_);
	}
	if(sparseMap_show_points_->Get()){
	  OrbSlamMapDrawer_->DrawMapPoints();
        }
        
	bool enable_compositing_dense = (paramGUI.evaluation_delay == 0);
	const unsigned char *preview_dense = nullptr;
        dense_map_fpv_view_->Activate(*dense_map_pane_cam_);
	Eigen::Matrix4f cam_mv = dense_slam_->GetPose().inverse();
        pangolin::OpenGlMatrix pm(cam_mv);
        dense_map_pane_cam_->SetModelViewMatrix(pm);
	glColor3f(1.0,1.0,1.0);
        preview_dense = dense_slam_->GetStaticMapRaycastPreview(dense_map_pane_cam_->GetModelViewMatrix(),
                  static_cast<PreviewType>(current_preview_type_),enable_compositing_dense);
        pane_texture_dense_->Upload(preview_dense, GL_RGBA, GL_UNSIGNED_BYTE);
        pane_texture_dense_->RenderToViewport(true);
        
      }

      // Swap frames and Process Events
      pangolin::FinishFrame();

      //将GUI的视屏保存到本地中
      if (paramGUI.record) {
        const string kRecordingRoot = "../recordings/";
        if (! utils::FileExists(kRecordingRoot)) {
          throw std::runtime_error(utils::Format(
              "Recording enabled but the output directory (%s) could not be found!",
              kRecordingRoot.c_str()));
        }
        string frame_fname = utils::Format("recorded-frame-%04d", dense_slam_->GetCurrentFrameNo());
        pangolin::SaveWindowOnRender(kRecordingRoot + "/" + frame_fname);
      }
    }
}

void PangolinGui::PreviewSparseSF(const std::vector<RawFlow,Eigen::aligned_allocator<RawFlow> >& flow, const pangolin::View& view){
    // pangolin::GlFont &font = pangolin::GlFont::I();
    Eigen::Vector2f frame_size(width_, height_);
//    font.Text("libviso2 scene flow preview").Draw(-0.90f, 0.89f);

    // We don't need z-checks since we're rendering UI stuff.
    glDisable(GL_DEPTH_TEST);
    for(const RawFlow &match : flow) {
      Eigen::Vector2f bounds(segment_view_.GetBounds().w, segment_view_.GetBounds().h);

      // Very hacky way of making the lines thicker
      for (int xof = -1; xof <= 1; ++xof) {
        for (int yof = -1; yof <= 1; ++yof) {
          Eigen::Vector2f of(xof, yof);
          Eigen::Vector2f gl_pos = PixelsToGl(match.curr_left + of, frame_size, bounds);
          Eigen::Vector2f gl_pos_old = PixelsToGl(match.prev_left + of, frame_size, bounds);

          Eigen::Vector2f delta = gl_pos - gl_pos_old;
          float magnitude = 15.0f * static_cast<float>(delta.norm());

          glColor4f(max(0.2f, min(1.0f, magnitude)), 0.4f, 0.4f, 1.0f);
          pangolin::glDrawCircle(gl_pos.cast<double>(), 0.010f);
          pangolin::glDrawLine(gl_pos_old[0], gl_pos_old[1], gl_pos[0], gl_pos[1]);
        }
      }
    }
    glEnable(GL_DEPTH_TEST);
}

void PangolinGui::DiffDepthmaps(const cv::Mat1s& input_depthmap, 
				const float* rendered_depth, 
				int width, 
				int height, 
				int delta_max, 
				uchar* out_image, 
				float baseline_m, 
				float focal_length_px){
      for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        int in_idx = (i * width + j);
        int out_idx = (i * width + j) * 4;
        float input_depth_m = input_depthmap.at<short>(i, j) / 1000.0f;
        float rendered_depth_m = rendered_depth[in_idx];

        float input_disp = baseline_m * focal_length_px / input_depth_m;
        float rendered_disp = baseline_m * focal_length_px / rendered_depth_m;

        if (input_depth_m == 0 || fabs(rendered_depth_m < 1e-5)) {
          continue;
        }

        float delta = input_disp - rendered_disp;
        float abs_delta = fabs(delta);
        if (abs_delta > delta_max) {
          // Visualize SIGNED delta to highlight areas where a particular method tends to
          // consistently over/underestimate.
          if (delta > 0) {
            out_image[out_idx + 0] = 0;
            out_image[out_idx + 1] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 2] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
          }
          else {
            out_image[out_idx + 0] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 1] = min(255, static_cast<int>(50 + (abs_delta - delta) * 10));
            out_image[out_idx + 2] = 0;
          }
        }
        else {
          out_image[out_idx + 0] = 0;
          out_image[out_idx + 1] = 255;
          out_image[out_idx + 2] = 0;
        }
      }
    }
}

void PangolinGui::PreviewLidar(const Eigen::MatrixX4f& lidar_points, 
			       const Eigen::Matrix34f& P, 
			       const Eigen::Matrix4f& Tr, 
			       const pangolin::View& view){
    // convert every velo point into 2D as: x_i = P * Tr * X_i
    if (lidar_points.rows() == 0) {
      return;
    }
    size_t idx_v = 0;
    size_t idx_c = 0;
    glDisable(GL_DEPTH_TEST);
    for (int i = 0; i < lidar_points.rows(); ++i) {
      Eigen::Vector4f point = lidar_points.row(i);
      float reflectance = lidar_points(i, 3);
      point(3) = 1.0f;                // Replace reflectance with the homogeneous 1.

      Eigen::Vector4f p3d = Tr * point;
      p3d /= p3d(3);
      float Z = p3d(2);

      lidar_vis_vertices_[idx_v++] = p3d(0);
      lidar_vis_vertices_[idx_v++] = p3d(1);
      lidar_vis_vertices_[idx_v++] = p3d(2);

      float intensity = min(8.0f / Z, 1.0f);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(intensity * 255);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(intensity * 255);
      lidar_vis_colors_[idx_c++] = static_cast<uchar>(reflectance * 255);
    }

    float fx = P(0, 0);
    float fy = P(1, 1);
    float cx = P(0, 2);
    float cy = P(1, 2);
    auto proj = pangolin::ProjectionMatrix(width_, height_, fx, fy, cx, cy, 0.01, 1000);

    pangolin::OpenGlRenderState state(
        proj, pangolin::IdentityMatrix().RotateX(M_PI)
    );
    state.Apply();

    // TODO-LOW(andrei): For increased performance (unnecessary), consider just passing ptr to
    // internal eigen data. Make sure its row-major though, and the modelview matrix is set properly
    // based on the velo-to-camera matrix.
    pangolin::glDrawColoredVertices<float>(idx_v / 3, lidar_vis_vertices_, lidar_vis_colors_, GL_POINTS, 3, 3);
    glEnable(GL_DEPTH_TEST);
}

void PangolinGui::CreatePangolinDisplays(){
         
    pangolin::CreateWindowAndBind("DenseSLAM GUI",
                                  kUiWidth + width_,
                                  // One full-height pane with the main preview, plus 3 * 0.5
                                  // height ones for various visualizations.
                                  static_cast<int>(ceil(height_ * 3)));
    
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    /***************************************************************************
     * GUI Buttons
     **************************************************************************/
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(kUiWidth));

    auto next_frame = [this]() {
      *(this->autoplay_) = false;
      this->ProcessFrame();
    };
    pangolin::Var<function<void(void)>> next_frame_button("ui.[N]ext Frame", next_frame);
    pangolin::RegisterKeyPressCallback('n', next_frame);

    //保存静态地图，即没有运动中的物体
    auto save_map = [this]() {
      Tic("Static map mesh generation");
      if (dense_slam_->GetCurrentFrameNo() < 2) {
        cerr << "Warning: no map to save!" << endl;
      }
      else {
        dense_slam_->SaveStaticMap(dense_slam_input_->GetDatasetIdentifier(),
                                 dense_slam_input_->GetDepthProvider()->GetName());
        cout << "Mesh generated OK. Writing asynchronously to the disk..." << endl;
        Toc();
      }
    };

    pangolin::Var<function<void(void)>> save_map_button("ui.[S]ave Static Map", save_map);
    pangolin::RegisterKeyPressCallback('s', save_map);
    
    auto ORB_SLAM2 = [this](){
      cout<< "the ORBSLAM has no integrate to this system yet"<< endl;
    };
    pangolin::Var<function<void(void)>> orb_slam2_button("ui.or[B] slam", ORB_SLAM2);
    pangolin::RegisterKeyPressCallback('b', ORB_SLAM2);
    
    
    reconstructions = new pangolin::Var<string>("ui.Rec", "");
//     pangolin::Var<function<void(void)>> previous_object("ui.Previous Object [z]", [this]() {
//       SelectPreviousVisualizedObject();
//     });
//     pangolin::RegisterKeyPressCallback('z', [this]() { SelectPreviousVisualizedObject(); });
    
    
//     pangolin::Var<function<void(void)>> next_object("ui.Ne[x]t Object", [this]() {
//       SelectNextVisualizedObject();
//     });
//     pangolin::RegisterKeyPressCallback('x', [this]() { SelectNextVisualizedObject(); });
//     auto save_object = [this]() {
//       dyn_slam_->SaveDynamicObject(dyn_slam_input_->GetDatasetIdentifier(),
//                                      dyn_slam_input_->GetDepthProvider()->GetName(),
//                                      visualized_object_idx_);
//     };
//     pangolin::Var<function<void(void)>> save_active_object("ui.Save Active [O]bject", save_object);
//     pangolin::RegisterKeyPressCallback('o', save_object);

    auto quit = [this]() {
//       dense_slam_->WaitForJobs();
      pangolin::QuitAll();
    };
    pangolin::Var<function<void(void)>> quit_button("ui.[Q]uit", quit);
    pangolin::RegisterKeyPressCallback('q', quit);

    auto previous_preview_type = [this]() {
      if (--current_preview_type_ < 0) {
        current_preview_type_ = (PreviewType::kEnd - 1);
      }
    };
    auto next_preview_type = [this]() {
      if (++current_preview_type_ >= PreviewType::kEnd) {
        current_preview_type_ = 0;
      }
    };
    pangolin::Var<function<void(void)>> ppt("ui.Previous Preview Type [j]", previous_preview_type);
    pangolin::RegisterKeyPressCallback('j', previous_preview_type);
    pangolin::Var<function<void(void)>> npt("ui.Next Preview Type [k]", next_preview_type);
    pangolin::RegisterKeyPressCallback('k', next_preview_type);

    pangolin::RegisterKeyPressCallback('0', [&]() {
      if(++current_lidar_vis_ >= VisualizeError::kEnd) {
        current_lidar_vis_ = 0;
      }
    });
    pangolin::RegisterKeyPressCallback('9', [&]() {
      if(--current_lidar_vis_ < 0) {
        current_lidar_vis_ = (VisualizeError::kEnd - 1);
      }
    });

//     pangolin::Var<function<void(void)>> collect("ui.Map Voxel [G]C Catchup", [&]() {
//       dense_slam_->StaticMapDecayCatchup();
//     });
//     pangolin::RegisterKeyPressCallback('g', [&]() { dense_slam_->StaticMapDecayCatchup(); });

    /***************************************************************************
     * GUI Checkboxes
     **************************************************************************/
    autoplay_ = new pangolin::Var<bool>("ui.[A]utoplay", paramGUI.autoplay, true);
    pangolin::RegisterKeyPressCallback('a', [this]() {
      *(this->autoplay_) = ! *(this->autoplay_);
    });
    
    display_raw_previews_ = new pangolin::Var<bool>("ui.Raw Previews", false, true);
    preview_sf_ = new pangolin::Var<bool>("ui.Show Scene Flow", false, true);
    sparseMap_show_points_ = new pangolin::Var<bool>("ui.Sparse Map Show Points", true, true);
    sparseMap_show_keyFrame_ = new pangolin::Var<bool>("ui.Sparse Map Show KeyFrames", true, true);
    sparseMap_show_graph_ = new pangolin::Var<bool>("ui.Sparse Map Show Graph",true, true);
    

    pangolin::RegisterKeyPressCallback('r', [&]() {
      *display_raw_previews_ = !display_raw_previews_->Get();
    });

    // This constructs an OpenGL projection matrix from a calibrated camera pinhole projection
    // matrix. They are quite different, and the conversion between them is nontrivial.
    // See https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/ for more info.
    // 从相机的针孔模型投影矩阵构建OpenGL的投影矩阵，这俩个矩阵不太一样，它们之间的转换并不简单
    const Eigen::Matrix34f real_cam_proj = dense_slam_->GetLeftRgbProjectionMatrix();
    float near = 0.01;
    float far = 1000.0f;
    // -y is up
    // ProjectionMatrixRDF_TopLeft(int w, int h, GLprecision fu, GLprecision fv, GLprecision u0, 
    //                             GLprecision v0, GLprecision zNear, GLprecision zFar );
    proj_ = pangolin::ProjectionMatrixRDF_TopLeft(width_, height_,
                                                  real_cam_proj(0, 0), real_cam_proj(1, 1),
                                                  real_cam_proj(0, 2), real_cam_proj(1, 2),
                                                  near, far);

    pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
	//显示的时候，设置查看模型的角度
        pangolin::ModelViewLookAtRDF(3.0,  -2.5, 10,
                                     3.0,  -2.5, 50,
                                     0, 1, 0));
    //pane_cam_->Follow()
    
    
    orb_pane_cam_ = new pangolin::OpenGlRenderState(
        proj_,
	pangolin::ModelViewLookAtRDF(3.0, -2.5, 15,
	                             3.0, -2.5, 50,
			             0, 1, 0));
   
    
    orb_Trajectory_pane_cam_ = new pangolin::OpenGlRenderState(
       proj_,
       pangolin::ModelViewLookAtRDF(0, -1.5, 15,
	                             0, -1.5, 50,
			             0, 1, 0));
    
    instance_cam_ = new pangolin::OpenGlRenderState(
        proj_,
        pangolin::ModelViewLookAtRDF(
          -0.8, -0.20,  -3,
          -0.8, -0.20,  15,
          0, 1, 0)
    );
    
    dense_map_pane_cam_ = new pangolin::OpenGlRenderState(
      proj_,
      pangolin::ModelViewLookAtRDF(0, -1.5, 15,
	                             0, -1.5, 50,
			             0, 1, 0));

    float aspect_ratio = static_cast<float>(width_) / height_;//宽高比
    rgb_view_ = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view_ = pangolin::Display("depth").SetAspect(aspect_ratio);
    if(dense_slam_->IsDynamicMode()){
        segment_view_ = pangolin::Display("segment").SetAspect(aspect_ratio);
        object_view_ = pangolin::Display("object").SetAspect(aspect_ratio);
    }
    float camera_translation_scale = 1.0f;
    float camera_zoom_scale = 1.0f;
    
    if(dense_slam_->IsDynamicMode()){
       object_reconstruction_view_ = pangolin::Display("object_3d").SetAspect(aspect_ratio)
        .SetHandler(new DSHandler3D(
            instance_cam_,
            pangolin::AxisY,
            camera_translation_scale,
            camera_zoom_scale
        ));
    }
    // These objects remain under Pangolin's management, so they don't need to be deleted by the current class.
    // 构建的地图主要显示
    main_view_ = &(pangolin::Display("main").SetAspect(aspect_ratio));
    
    //SetHandler主要是方便用鼠标和键盘控制 
    main_view_->SetHandler(
        new DSHandler3D(pane_cam_,
                        pangolin::AxisY,
                        camera_translation_scale,
                        camera_zoom_scale));
    
    // 构建OrbSLAM地图的显示
    orbslam_view_ = &(pangolin::Display("orbslam"));
    //orbslam_view_->SetHandler(new DSHandler3D(orb_pane_cam_));
    //orbslam_view_->SetHandler(new pangolin::Handler3D(*orb_pane_cam_));
    
    orbslam_view_->SetHandler(
    new DSHandler3D(orb_pane_cam_,
                    pangolin::AxisY,
                    camera_translation_scale,
                    camera_zoom_scale));
    
    //用于显示OrbSLAM的位姿
    if(!dense_slam_->IsDynamicMode()){
    orb_trajectory_view_ = &(pangolin::Display("orbslam_trajectory"));
    orb_trajectory_view_->SetHandler(
      new DSHandler3D(orb_Trajectory_pane_cam_,
	              pangolin::AxisY,
		      camera_translation_scale,
		      camera_zoom_scale));
    dense_map_fpv_view_ = &(pangolin::Display("dense_map_fpv_view"));
    dense_map_fpv_view_->SetHandler(
       new DSHandler3D(orb_Trajectory_pane_cam_,
	   pangolin::AxisY,
           camera_translation_scale,
           camera_zoom_scale));
    }
    //主要的一些细节部分
    detail_views_ = &(pangolin::Display("detail"));

    // Add labels to our data logs (and automatically to our plots).
    if(dense_slam_->IsDynamicMode()){
       data_log_.SetLabels({"Active tracks",
                         "Free GPU Memory (100s of MiB)",
                         "Static map memory usage (100s of MiB)",
                         "Static map memory usage without decay (100s of Mib)",
                        });

    // OpenGL 'view' of data such as the number of actively tracked instances over time.
       float tick_x = 1.0f;
       float tick_y = 1.0f;
       plotter_ = new pangolin::Plotter(&data_log_, 0.0f, 200.0f, -0.1f, 25.0f, tick_x, tick_y);
       plotter_->Track("$i");  // This enables automatic scrolling for the live plots.
    }
    // TODO(andrei): Maybe wrap these guys in another controller, make it an equal layout and
    // automagically support way more aspect ratios?
    
    //main_views:指的是融合后的地图
    //detail_views:指的是static_rgb、 static_depth、 segment_view_、object_view
    //SetBounds(bottom, top, left, right)
    main_view_->SetBounds(pangolin::Attach::Pix(height_ *2.0), pangolin::Attach::Pix(height_ * 3.0), pangolin::Attach::Pix(kUiWidth*1.2), pangolin::Attach::Pix(kUiWidth*1.2+width_));
    orbslam_view_->SetBounds(pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(height_ * 2.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
    if(dense_slam_->IsDynamicMode()){
    detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
    detail_views_->SetLayout(pangolin::LayoutEqual)
      .AddDisplay(rgb_view_)
      .AddDisplay(depth_view_)
      .AddDisplay(segment_view_)
      .AddDisplay(object_view_)
      .AddDisplay(*plotter_)
      .AddDisplay(object_reconstruction_view_);
    }
    else{
      detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_*(1.0/3.0)));
      detail_views_->SetLayout(pangolin::LayoutEqual)
      .AddDisplay(rgb_view_)
      .AddDisplay(depth_view_);
      orb_trajectory_view_->SetBounds(pangolin::Attach::Pix(height_*0.32), pangolin::Attach::Pix(height_*1.0), 
				      pangolin::Attach::Pix(kUiWidth+width_*(1.0/3.0)),
				      pangolin::Attach::Pix(kUiWidth+width_*(2.0/3.0)));
      dense_map_fpv_view_->SetBounds(pangolin::Attach::Pix(height_*0.32), pangolin::Attach::Pix(height_*1.0), 
				      pangolin::Attach::Pix(kUiWidth+width_*(2.0/3.0)),
				      pangolin::Attach::Pix(kUiWidth+width_));
    }
    // Internally, InfiniTAM stores these as RGBA, but we discard the alpha when we upload the
    // textures for visualization (hence the 'GL_RGB' specification).
    this->pane_texture_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RGB,GL_UNSIGNED_BYTE);
    this->pane_texture_mono_uchar_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RED, GL_UNSIGNED_BYTE);
    this->pane_texture_dense_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RED, GL_UNSIGNED_BYTE);
                                                             
    cout << "Pangolin UI setup complete." << endl;
}

void PangolinGui::ProcessFrame(){
        cout << endl << "[Starting frame " << dense_slam_->GetCurrentFrameNo() + 1 << "]" << endl;
//   active_object_count_ = dyn_slam_->GetInstanceReconstructor()->GetActiveTrackCount();

    if (! dense_slam_input_->HasMoreImages() && paramGUI.close_on_complete) {
      cerr << "No more images, and I'm instructed to shut down when that happens. Bye!" << endl;
      pangolin::QuitAll();
      return;
    }

    size_t free_gpu_memory_bytes;
    size_t total_gpu_memory_bytes;
    cudaMemGetInfo(&free_gpu_memory_bytes, &total_gpu_memory_bytes);

    const double kBytesToGb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    double free_gpu_gb = static_cast<float>(free_gpu_memory_bytes) * kBytesToGb;
//     data_log_.Log(
// //        active_object_count_,
//         static_cast<float>(free_gpu_gb) * 10.0f,   // Mini-hack to make the scales better
//         dense_slam_->GetStaticMapMemoryBytes() * 10.0f * kBytesToGb,
//         (dense_slam_->GetStaticMapMemoryBytes() + dense_slam_->GetStaticMapSavedDecayMemoryBytes()) * 10.0f * kBytesToGb
//     );

    Tic("DynSLAM frame");
    // Main workhorse function of the underlying SLAM system.
    dense_slam_->ProcessFrame(this->dense_slam_input_);
    int64_t frame_time_ms = Toc(true);
    float fps = 1000.0f / static_cast<float>(frame_time_ms);
    cout << "[Finished frame " << dense_slam_->GetCurrentFrameNo() << " in " << frame_time_ms
         << "ms @ " << setprecision(4) << fps << " FPS (approx.)]"
         << endl;
}
}//gui
}//SparsetoDense