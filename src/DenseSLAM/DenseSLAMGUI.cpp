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

    /// NOTE 对所有位姿进行绘制
    float kMaxFrustumScale = 0.66;
    Eigen::Vector3f color_blue(0.0f, 0.0f, 255.0f);
    
    //对于大范围的KITTI数据集，我们将所有的位姿画出来
    if(dense_slam_input_->GetDatasetType() == Input::KITTI){
        for (int i = 0; i < static_cast<int>(phist.size()) - 1; ++i) {
//             float frustum_scale = max(0.15f, kMaxFrustumScale - 0.05f * (phist.size() - 1 - i));
            DrawPoseFrustum(phist[i], color_blue, kMaxFrustumScale, frustum_root_cube_scale);
        }
    }
    else if(dense_slam_input_->GetDatasetType() == Input::TUM || dense_slam_input_->GetDatasetType() == Input::ICLNUIM){
      kMaxFrustumScale = 0.3;
         //这代码只画出最新5帧的位姿
      for (int i = static_cast<int>(phist.size())-5; i < static_cast<int>(phist.size()) - 1; ++i) {
//          for (int i = 0; i < static_cast<int>(phist.size()) - 1; ++i) {
            DrawPoseFrustum(phist[i], color_blue, kMaxFrustumScale, frustum_root_cube_scale);
      }
    }
    else{
      runtime_error("Currently not support this dataset type!");
    }

    /// NOTE 对最新的位姿颜色进行加深
    if (! phist.empty() ) {
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

      /// NOTE 是否在chase cam模式下预览重建，第三视角
      if (paramGUI.chase_cam) 
      {
        Eigen::Matrix4f cam_mv = dense_slam_->GetPose().inverse();
        pangolin::OpenGlMatrix pm(cam_mv);
	if(dense_slam_input_->GetDatasetType() == Input::KITTI){
        pm =
            // Good for odo 05
             pangolin::OpenGlMatrix::RotateY(M_PI * 0.5 * 0.05f) *
             pangolin::OpenGlMatrix::RotateX(M_PI * 0.5 * 0.03f) *
             pangolin::OpenGlMatrix::Translate(-0.5, 2.0, 20.0) *
             pm;
	}
	else if(dense_slam_input_->GetDatasetType() == Input::TUM || dense_slam_input_->GetDatasetType() == Input::ICLNUIM){
	 pm =
            // Good for odo 05
            pangolin::OpenGlMatrix::RotateY(M_PI * 0.5 * 0.05f) *
            pangolin::OpenGlMatrix::RotateX(M_PI * 0.5 * 0.03f) *
            pangolin::OpenGlMatrix::Translate(-0.5, 0.0, 2.0) *
            pm;
	}
	else{
	  runtime_error("Currently not support this dataset type!");
	}
         pane_cam_->SetModelViewMatrix(pm);
	 orb_pane_cam_->SetModelViewMatrix(pm);
      }

      int evaluated_frame_idx = dense_slam_->GetCurrentFrameNo() - 1 - paramGUI.evaluation_delay;
      if (evaluated_frame_idx > 0) { 
	
        bool enable_compositing = (paramGUI.evaluation_delay == 0);
// 	//void* memset(void *str, int c, size_t n)复制字符c(一个无符号字符)到参数str所指向的字符串的前n个字符
// 	//diff_buffer为要填充的内存块，‘\0’为要被设置的值，sizeof(uchar)*width_*height_*4为要被设置为该值的字节数
//         memset(diff_buffer, '\0', sizeof(uchar) * width_ * height_ * 4);

        const unsigned char *preview = nullptr;
        string message;
        switch(current_lidar_vis_) {
          case kNone://
            if (paramGUI.chase_cam) {
              message = "Chase cam preview";
            }
            else {
              message = "Free cam preview";
            }
            preview = dense_slam_->GetMapRaycastPreview(pane_cam_->GetModelViewMatrix(), static_cast<PreviewType>(current_preview_type_), enable_compositing);
	    pane_texture_->Upload(preview, GL_RGBA, GL_UNSIGNED_BYTE);
            pane_texture_->RenderToViewport(true);
            DrawPose(time_ms);
            break;
         }
         // Ensures we have a blank slate for the pane's overlay text.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1.0f, 1.0f, 1.0f);
      }
      
      //*/
      //font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);
      
//    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      OrbSlamMapDrawer_->GetCurrentOpenGLCameraMatrix(OrbSlamTwc_);
      orbslam_view_->Activate(*orb_pane_cam_);
      if(dense_slam_->GetCurrentFrameNo()>=1){
	glClearColor(0.0f,0.0f,0.0f,0.0f);
	bool showGraph = true;
	if(dense_slam_input_->GetDatasetType() == Input::TUM || dense_slam_input_->GetDatasetType() == Input::ICLNUIM){
	    OrbSlamMapDrawer_->SetViewKeyframe(5);
	    showGraph = false;
	}
	if(sparseMap_show_graph_->Get()){
	  OrbSlamMapDrawer_->DrawCurrentCamera(OrbSlamTwc_);
	}
	if(sparseMap_show_keyFrame_->Get()){
	  OrbSlamMapDrawer_->DrawKeyFrames(true,showGraph);
	}
	if(sparseMap_show_points_->Get()){
	  OrbSlamMapDrawer_->DrawMapPoints();
	}
      }
      //font.Text("Frame #%d", dyn_slam_->GetCurrentFrameNo()).Draw(-0.90f, 0.90f);
      
      //激活rgb_view_以便绘制
      rgb_view_.Activate();
      glColor3f(1.0f, 1.0f, 1.0f);
      /// NOTE  当前帧大于1，对RGB图片进行预览。 
      /// 显示的是ORB特征点的RGB图片
      if(dense_slam_->GetCurrentFrameNo() >= 1){
      cv::Mat im = dense_slam_->GetOrbSlamFrameDrawerGlobal()->DrawFrame();
	UploadCvTexture(im, *pane_texture_, true, GL_UNSIGNED_BYTE);
	pane_texture_->RenderToViewport(true);
      }
        
      bool enable_compositing_dense = (paramGUI.evaluation_delay == 0);
      /// NOTE 激活depth_viewe_以进行绘制，对深度图进行显示
      depth_view_.Activate();
      glColor3f(1.0,1.0,1.0);
      /// 可视化光线投影的深度图
      if (dense_slam_->GetDepthPreview() != nullptr) {
           UploadCvTexture(*(dense_slam_->GetDepthPreview()), *pane_texture_, false, GL_SHORT);
      }
      pane_texture_->RenderToViewport(true);
      
      raycast_depth_view_.Activate();
      glColor3f(1.0,1.0,1.0);
      cv::Size2i tempRaycastDepthSize = dense_slam_input_->GetDepthSize();
      cv::Mat1s *tempRaycastShort = new cv::Mat1s(tempRaycastDepthSize.height, tempRaycastDepthSize.width);
      Eigen::Matrix4f cam_raycast = dense_slam_->GetPose().inverse();
      pangolin::OpenGlMatrix pm_raycast(cam_raycast);
      /// NOTE 光线投影回来的深度图
      const float* tempRaycastDepth = dense_slam_ -> GetRaycastDepthPreview(pm_raycast, static_cast<PreviewType>(current_preview_depth_type), enable_compositing_dense);
      if(tempRaycastDepth != nullptr ){
	SparsetoDense::FloatDepthmapToShort(tempRaycastDepth, *tempRaycastShort);
	if(tempRaycastDepth != nullptr){
	    UploadCvTexture(*tempRaycastShort,*pane_texture_, false, GL_SHORT);
	 }
      }
      delete tempRaycastShort;
      pane_texture_->RenderToViewport(true);
      
      /// NOTE 以第一视角观察orbslam以及稠密地图的第一视角
      /*
      if(dense_slam_->GetCurrentFrameNo()>=1){
// 	  orb_trajectory_view_->Activate(*orb_Trajectory_pane_cam_);
// 	  glColor3f(1.0,1.0,1.0);
// 	  // OrbSlamMapDrawer_->DrawTracjectory();
//           orb_Trajectory_pane_cam_->Follow(OrbSlamTwc_);
	if(sparseMap_show_graph_->Get()){
	  OrbSlamMapDrawer_->DrawCurrentCamera(OrbSlamTwc_);
	}
	if(sparseMap_show_points_->Get()){
	  OrbSlamMapDrawer_->DrawMapPoints();
        }
        
	const unsigned char *preview_dense = nullptr;
        dense_map_fpv_view_->Activate(*dense_map_pane_cam_);
	
	Eigen::Matrix4f cam_mv = dense_slam_->GetPose().inverse();
	
        pangolin::OpenGlMatrix pm(cam_mv);
        dense_map_pane_cam_->SetModelViewMatrix(pm);
	glColor3f(1.0,1.0,1.0);
        preview_dense = dense_slam_->GetMapRaycastPreview(dense_map_pane_cam_->GetModelViewMatrix(),
                  static_cast<PreviewType>(current_preview_type_),enable_compositing_dense);
        pane_texture_dense_->Upload(preview_dense, GL_RGBA, GL_UNSIGNED_BYTE);
        pane_texture_dense_->RenderToViewport(true); 
      }
      */
      
      *(NumFrame) = Format("%d", dense_slam_->GetCurrentFrameNo());
      *(NumKeyFrame) = Format("%d", dense_slam_->GetKeyFrameNum());
      *(AvgFusionTime) = Format("%f%s", (float)dense_slam_->GetTotalFusionTime()/(float)dense_slam_->GetKeyFrameNum(), " ms");
//       *(NumActiveLocalMap) = Format("%d", dense_slam_->GetNumActiveLocalMap());
      *(NumLocalMap) = Format("%d", dense_slam_->GetNumLocalMap());
      if(dense_slam_->GettodoList().size()>0){
         *(CurrentLocalMapStartKeyframeNo) = Format("%d", dense_slam_->GettodoList().back().startKeyframeTimeStamp);
	 *(CurrentLocalMapEndKeyframeNo) = Format("%d", dense_slam_->GettodoList().back().endKeyframeTimeStamp);
      }
      // Swap frames and Process Events
      pangolin::FinishFrame();

      /// NOTE 将GUI的视屏保存到本地中
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

    //保存地图
    auto save_map = [this]() {
      Tic("Dense map mesh generation");
      if (dense_slam_->GetCurrentFrameNo() < 2) {
        cerr << "Warning: no map to save!" << endl;
      }
      else {
        dense_slam_->SaveStaticMap(dense_slam_input_->GetDatasetIdentifier(),
				  dense_slam_->GetCurrentLocalMap(),
				  dense_slam_->GetNumLocalMap());
        cout << "Mesh generated OK. Writing asynchronously to the disk..." << endl;
        Toc();
      }
    };

    pangolin::Var<function<void(void)>> save_map_button("ui.[S]ave Dense and Sparse Map ", save_map);
    pangolin::RegisterKeyPressCallback('s', save_map);
    
    auto ORB_SLAM2 = [this](){
      cout<< "the ORBSLAM has no integrate to this system yet"<< endl;
    };
    pangolin::Var<function<void(void)>> orb_slam2_button("ui.or[B] slam", ORB_SLAM2);
    pangolin::RegisterKeyPressCallback('b', ORB_SLAM2);

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
    
    /***************************************************************************
     * GUI Checkboxes
     **************************************************************************/
    NumLocalMap = new pangolin::Var<string>("ui.Number of Submaps: ", "");
//  NumActiveLocalMap = new pangolin::Var<string>("ui.Number of Active Submaps: ", "");
    
    NumFrame = new pangolin::Var<string>("ui.Number of Frames: ", "");
    NumKeyFrame = new pangolin::Var<string>("ui.Number of KeyFrames: ", "");
    AvgFusionTime = new pangolin::Var<string>("ui.Avg Fusion Time: ", "");
    
    CurrentLocalMapStartKeyframeNo = new pangolin::Var<string> ("ui.Curr LocalMap Start Keyframe No.", "");
    CurrentLocalMapEndKeyframeNo = new pangolin::Var<string> ("ui.Curr LocalMap End Keyframe No.", "");

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

    float aspect_ratio = static_cast<float>(width_) / height_;//宽高比
    rgb_view_ = pangolin::Display("rgb").SetAspect(aspect_ratio);
    depth_view_ = pangolin::Display("depth").SetAspect(aspect_ratio);
    raycast_depth_view_ = pangolin::Display("raycast_depth").SetAspect(aspect_ratio);
    float camera_translation_scale = 0.5f;
    float camera_zoom_scale = 0.5f;
    
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
    orbslam_view_->SetHandler(
    new DSHandler3D(orb_pane_cam_,
                    pangolin::AxisY,
                    camera_translation_scale,
                    camera_zoom_scale));
    
    //主要的一些细节部分
    detail_views_ = &(pangolin::Display("detail"));

    // Add labels to our data logs (and automatically to our plots).
    data_log_memory.SetLabels({
                      "Free GPU Memory (100 MB)",
		      "Total GPU Memory (100 MB)",
		      "Dense map allocated memory (100 MB)",
                      "Dense map memory usage (100 MB)",
                     });
    
    // OpenGL 'view' of data such as the number of actively tracked instances over time.
    // tick指将横纵坐标分为多少份
    float tick_x = 30.0f;
    float tick_y_memory = 15.0f;
//     float tick_y_track =  250.0f;
    plotter_memory = new pangolin::Plotter(&data_log_memory, 0.0f, 200.0f, 0.0f, 25.0f, tick_x, tick_y_memory);
    plotter_memory->Track("$i");  // This enables automatic scrolling for the live plots.
       
    //main_views:指的是融合后的地图
    //detail_views:指的是static_rgb、 static_depth、 segment_view_、object_view
    //SetBounds(bottom, top, left, right)
    if(dense_slam_input_->GetDatasetType() == Input::KITTI){
       if(dense_slam_input_->GetDepthSize().width == 912){
	  main_view_->SetBounds(pangolin::Attach::Pix(height_ *2.5), pangolin::Attach::Pix(height_ * 4.5), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth*1.2+width_));
          orbslam_view_->SetBounds(pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(height_ * 2.5), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
       }
       else{
          main_view_->SetBounds(pangolin::Attach::Pix(height_ *2.0), pangolin::Attach::Pix(height_ * 3.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth*1.2+width_));
          orbslam_view_->SetBounds(pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(height_ * 2.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
       }
       detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
       detail_views_->SetLayout(pangolin::LayoutEqual)
         .AddDisplay(rgb_view_)
         .AddDisplay(depth_view_)
         .AddDisplay(*plotter_memory)
	 .AddDisplay(raycast_depth_view_);
//          .AddDisplay(*plotter_track);
    }
    else if(dense_slam_input_->GetDatasetType() == Input::TUM || dense_slam_input_->GetDatasetType() == Input::ICLNUIM){
      main_view_->SetBounds(pangolin::Attach::Pix(height_ *1.0), pangolin::Attach::Pix(height_ * 2.0), pangolin::Attach::Pix(kUiWidth+width_), pangolin::Attach::Pix(kUiWidth+2*width_));
      orbslam_view_->SetBounds(pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(height_ * 2.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+width_));
      detail_views_->SetBounds(0.0, pangolin::Attach::Pix(height_ * 1.0), pangolin::Attach::Pix(kUiWidth), pangolin::Attach::Pix(kUiWidth+2*width_));
      detail_views_->SetLayout(pangolin::LayoutEqual)
         .AddDisplay(rgb_view_)
         .AddDisplay(depth_view_)
         .AddDisplay(*plotter_memory)
	 .AddDisplay(raycast_depth_view_);
//          .AddDisplay(*plotter_track);
    }
    else{
      runtime_error("Currently not support dataset type !");
    }
    
    // Internally, InfiniTAM stores these as RGBA, but we discard the alpha when we upload the
    // textures for visualization (hence the 'GL_RGB' specification).
    this->pane_texture_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RGB,GL_UNSIGNED_BYTE);
    this->pane_texture_mono_uchar_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RED, GL_UNSIGNED_BYTE);
    this->pane_texture_dense_ = new pangolin::GlTexture(width_, height_, GL_RGB, false, 0, GL_RED, GL_UNSIGNED_BYTE);
                                                             
    cout << "Pangolin UI setup complete." << endl;
}

void PangolinGui::ProcessFrame(){
    cout << endl << "[Starting frame " << dense_slam_->GetCurrentFrameNo() << "]" << endl;
    bool hasMoreImages = dense_slam_input_->HasMoreImages();
    if (!hasMoreImages && dense_slam_input_->GetFrameIndex() > 980){
      dense_slam_->SaveTUMTrajectory("/home/hansry/DenseSLAM-Global-Consistency-h/data/result.txt");
      cout << "No more images, Bye!" << endl;
      
      if(!dense_slam_->todoList.empty()){
	for(int mapNO = 0; mapNO < dense_slam_->todoList.size(); mapNO++){
	  int mapNum = dense_slam_->todoList[mapNO].dataId;
	  ITMLib::Engine::ITMLocalMap* currLocalMap = dense_slam_->GetStaticScene()->GetMapManager()->getLocalMap(mapNum);
	  dense_slam_->SaveStaticMap(dense_slam_input_->GetDatasetIdentifier(), currLocalMap, mapNum);
	}
      }
      getchar();
    }
    
    if(!hasMoreImages){
      cout << "[Finished frame " << dense_slam_->GetCurrentFrameNo() << "]" << endl;
      return;
    }
    Tic("DenseSLAM frame");
    // Main workhorse function of the underlying SLAM system.
    dense_slam_->ProcessFrame(this->dense_slam_input_);
    size_t free_gpu_memory_bytes;
    size_t total_gpu_memory_bytes;
    cudaMemGetInfo(&free_gpu_memory_bytes, &total_gpu_memory_bytes);
    
    const double kBytesToGb = 1.0 / 1024.0 / 1024.0 / 1024.0;
    double free_gpu_gb = static_cast<float>(free_gpu_memory_bytes) * kBytesToGb;    
    double total_gpu_gb = static_cast<float>(total_gpu_memory_bytes) * kBytesToGb;  
    
    size_t currLocalMapUsedMemory = dense_slam_->GetStaticMapMemoryBytes();
    size_t currLocalMapAlloMemory = dense_slam_->GetCurrMapAllocatedMapMemoryBytes();
    double curr_LocalMap_used_gpu_gb = static_cast<float>(currLocalMapUsedMemory) * kBytesToGb;
    double curr_LocalMap_allo_gpu_gb = static_cast<float>(currLocalMapAlloMemory) * kBytesToGb;

    //由于计算出来的free_gpu_gb等均为以G为单位的，因此乘上10.24转化为以（100MB的显示）
    data_log_memory.Log(
      static_cast<float>(free_gpu_gb) * 10.24f, 
      static_cast<float>(total_gpu_gb) * 10.24f,
      static_cast<float>(curr_LocalMap_allo_gpu_gb) * 10.24f, 
      static_cast<float>(curr_LocalMap_used_gpu_gb) * 10.24f
    );
    
    int orbslamTrackIntensity = dense_slam_->mTrackIntensity;
    float PDThreshold = dense_slam_->GetPDThreadhold();
    data_log_track.Log(
      orbslamTrackIntensity,
      PDThreshold
    );

    int64_t frame_time_ms = Toc(true);
    float fps = 1000.0f / static_cast<float>(frame_time_ms);
    cout << "[Finished frame " << dense_slam_->GetCurrentFrameNo()-1 << " in " << frame_time_ms
         << "ms @ " << setprecision(4) << fps << " FPS (approx.)]"
         << endl;
}
}//gui
}//SparsetoDense