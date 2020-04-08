#ifndef DENSESLAM_INFINITAMDRIVER_H
#define DENSESLAM_INFINITAMDRIVER_H

#include <iostream>

#include <opencv/cv.h>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <gflags/gflags.h>

#include "../InfiniTAM-Global-Consistency-h/InfiniTAM/ITMLib/Engine/ITMMainEngine.h"
#include "Defines.h"
#include "Input.h"
#include "PreviewType.h"
#include "VoxelDecayParams.h"

DECLARE_bool(enable_evaluation);

namespace SparsetoDense {
namespace drivers {

// 将OpenCV转换到InfiniTAM向量的程序
template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Vec<T, 2> in) {
  return ORUtils::Vector2<T>(in[0], in[1]);
}

template<typename T>
ORUtils::Vector2<T> ToItmVec(const cv::Size_<T> in) {
  return ORUtils::Vector2<T>(in.width, in.height);
}

template<typename T>
ORUtils::Vector3<T> ToItmVec(const cv::Vec<T, 3> in) {
  return ORUtils::Vector3<T>(in[0], in[1], in[2]);
}

template<typename T>
ORUtils::Vector4<T> ToItmVec(const cv::Vec<T, 4> in) {
  return ORUtils::Vector4<T>(in[0], in[1], in[2], in[3]);
}

ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const std::string &fpath);
//ITMLib::Objects::ITMRGBDCalib ReadITMCalibration(const string &fpath);

/// \brief 将OpenCV RGB Mat转成InfiniTAM image
void CvToItm(const cv::Mat3b &mat, ITMUChar4Image *out_itm);

/// \brief 将Opencv Mat格式的深度图转换为InfiniTAM格式的深度图
void CvToItm(const cv::Mat1s &mat, ITMShortImage *out_itm);

/// \brief 将InfiniTAM rgb(a)图像转换为OpenCV RGB mat格式，丢弃掉alpha通道的信息
void ItmToCv(const ITMUChar4Image &itm, cv::Mat3b *out_mat);

/// \brief 将InfiniTAM深度图转换为Opencv Mat格式的深度图
void ItmToCv(const ITMShortImage &itm, cv::Mat1s *out_mat);

/// \brief 将InfiniTAM浮点类型的深度图转成OpenCV格式的深度图
void ItmDepthToCv(const ITMFloatImage &itm, cv::Mat1s *out_mat);

/// @brief 将深度图从Float转成Short类型,这里将float转成short会乘上1000
void FloatDepthmapToShort(const float *pixels, cv::Mat1s &out_mat);

/// \brief 将深度图从Float转到Short
void FloatDepthmapToShortForRaycast(const float* pixels, cv::Mat1s &out_mat);

/// \brief 将InfiniTAM 4x4的矩阵转换为Eigen类型
Eigen::Matrix4f ItmToEigen(const Matrix4f &itm_matrix);

Matrix4f EigenToItm(const Eigen::Matrix4f &eigen_matrix);

ITMLib::Objects::ITMPose PoseFromPangolin(const pangolin::OpenGlMatrix &pangolin_matrix);

ITMLib::Objects::ITMRGBDCalib* CreateItmCalib(
    const Eigen::Matrix<double, 3, 4> &left_cam_proj,
    const Eigen::Vector2i &frame_size
);


/// \brief DenseSLAM和InfiniTAM的接口
class InfiniTamDriver : public ITMLib::Engine::ITMMainEngine{
public:
  
  InfiniTamDriver(  //InfiniTamDriver的构造函数
      const ITMLib::Objects::ITMLibSettings *settings,
      const ITMLib::Objects::ITMRGBDCalib *calib,
      const Vector2i &img_size_rgb,
      const Vector2i &img_size_d,
      const VoxelDecayParams &voxel_decay_params,
      bool use_depth_weighting)
      : ITMLib::Engine::ITMMainEngine(settings, calib, img_size_rgb, img_size_d),
        rgb_itm_(new ITMUChar4Image(img_size_rgb, true, true)),
        raw_depth_itm_(new ITMShortImage(img_size_d, true, true)),
        rgb_cv_(new cv::Mat3b(img_size_rgb.height, img_size_rgb.width)),
        raw_depth_cv_(new cv::Mat1s(img_size_d.height, img_size_d.width)),
        rgb_cv_local_map_(new cv::Mat3b(img_size_rgb.height, img_size_rgb.width)),
        raw_depth_cv_local_map_(new cv::Mat1s(img_size_d.height, img_size_d.width)),
        last_egomotion_(new Eigen::Matrix4f),
        voxel_decay_params_(voxel_decay_params)
  {
    last_egomotion_->setIdentity();
//     fusion_weight_params_.depthWeighting = use_depth_weighting;
  }

  virtual ~InfiniTamDriver() {
    delete rgb_itm_;
    delete raw_depth_itm_;
    delete rgb_cv_;
    delete raw_depth_cv_;
    delete last_egomotion_;
  }

  void UpdateView(const cv::Mat3b &rgb_image, const cv::Mat1s &raw_depth_image, double timestamp);
  
//   ITMLib::ITMTrackingState::TrackingResult trackingResult(const cv::Mat3b &rgb_image,
//                                                     const cv::Mat1s &raw_depth_image);

  // used by the instance reconstruction
  ITMActiveMapManager* GetActivateDataManger() const {
    return this->mActiveDataManger;
  }
  
  ITMVoxelMapGraphManager* GetMapManager() const {
    return this->mapManager;
  }
  
  void SetView(ITMLib::Objects::ITMView *view) {
    this->view = view;
  }
  
  /// @brief 对主子地图进行跟踪后得到的俩帧之间的相对变换
  void TrackPrimaryLocalMap() {
     ITMLocalMap* primaryLocalMap = this->GetPrimaryLocalMap();
     TrackLocalMap(primaryLocalMap);
  }
  
  //对对应的子地图进行跟踪后得到的俩帧之间的相对变换
  void TrackLocalMap(ITMLocalMap* currLocalMap){
    // 上一时刻主子图坐标系到当前帧坐标系的位姿变换，T_{w,p}
    Matrix4f old_pose = currLocalMap->trackingState->pose_d->GetInvM();
    Matrix4f old_pose_inv;
    //old_pose_inv: T_{o,w}
    old_pose.inv(old_pose_inv);
    this->trackingController->Track(currLocalMap->trackingState, this->view);
    //new_pose: T_{w,n}
    Matrix4f new_pose = currLocalMap->trackingState->pose_d->GetInvM();
    //last_egomotion_ = T_{o,w}*T_{w,n} = T_{o,n} 
    //最近俩帧之间的位姿变换
    *(this->last_egomotion_) = ItmToEigen(old_pose_inv * new_pose);
  }

  // 显示地设置跟踪状态，不仅要更新最新俩帧的相对位姿(last_egomotion_)，同时也需要更新最新的姿态，世界坐标系到当前坐标系的变换(new_pose)
  // this->GetTrackingState()也是对primary local map而言的
  // 在这里，我们认为new_pose是基于世界坐标系而言的
  void SetPose(const Eigen::Matrix4f &new_pose) {
    const ITMLocalMap* primaryLocalMap = this->GetPrimaryLocalMap();
    SetPoseLocalMap(primaryLocalMap, new_pose);
  }
  
  void SetPoseLocalMap(const ITMLocalMap* currLocalMap, const Eigen::Matrix4f &new_pose){
    const Matrix4f Tcurrmap_w = currLocalMap->estimatedGlobalPose.GetM();
    //Tprevious_c = (Tprevious_currmap) * (T_currmap_w * Tw_c)
    *(this->last_egomotion_)  = ItmToEigen(currLocalMap->trackingState->pose_d->GetInvM()).inverse()*(ItmToEigen(Tcurrmap_w) * new_pose);
    currLocalMap->trackingState->pose_d->SetInvM(Tcurrmap_w * EigenToItm(new_pose));
  }

  /// @brief 融合主局部地图
  void Integrate() const{ 
      ITMLocalMap* primaryLocalMap = this->GetPrimaryLocalMap();
      IntegrateLocalMap(primaryLocalMap);
  }
  
  /// @brief 融合局部地图
  void IntegrateLocalMap(const ITMLocalMap* currLocalMap) const{
    // this->denseMapper->SetFusionWeightParams(fusion_weight_params_);
    this->denseMapper->ProcessFrame(
      this->view, currLocalMap->trackingState, currLocalMap->scene, currLocalMap->renderState);
  }
  
  /// @brief 对局部地图进行反融合
  void DeIntegrateLocalMap(const ITMLocalMap* currLocalMap) const{
    this->denseMapper->DeProcessFrame(
      this->view, currLocalMap->trackingState, currLocalMap->scene, currLocalMap->renderState);
  }

  /// @brief 对于主局部地图而言，调用了PrepareNextStepLocalMap
  void PrepareNextStep() {
     ITMLocalMap* PrimaryLocalMap = GetPrimaryLocalMap();
      PrepareNextStepLocalMap(PrimaryLocalMap);
  }
  
  /// @brief 对于局部地图而言  
  void PrepareNextStepLocalMap(const ITMLocalMap* currLocalMap){
     const ITMRenderState_VH *renderState_vh = (ITMRenderState_VH*) (currLocalMap->renderState);
     if(renderState_vh->noVisibleEntries>0){
        
        this->trackingController->Prepare(currLocalMap->trackingState, 
					  currLocalMap->scene, 
					  this->view, 
					  currLocalMap->renderState);
	
	ItmToCv(*(this->view->rgb), rgb_cv_local_map_);
	ItmDepthToCv(*(this->view->depth), raw_depth_cv_local_map_);
    }
  }

  const ITMLib::Objects::ITMLibSettings* GetSettings() const {
    return this->settings;
  }

  void GetImage(
      ITMUChar4Image *out,
      SparsetoDense::PreviewType get_image_type,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix(),
      const ITMLocalMap *currentLocalMap = nullptr
  );

  void GetFloatImage(
      ITMFloatImage *out,
      SparsetoDense::PreviewType get_image_type,
      const pangolin::OpenGlMatrix &model_view = pangolin::IdentityMatrix(),
      const ITMLocalMap *currentLocalMap = nullptr
  );
  
  ITMLib::Engine::ITMSwappingEngine<ITMVoxel, ITMVoxelIndex>* GetSwappingEngine(){
    return this->denseMapper->GetSwappingEngine();
  }
  
  /// @brief 返回世界坐标系到当前帧的变换，Tw->c,the transform from world to current
  Eigen::Matrix4f GetPose() const{
      /// 得到主子地图到
      const ITMLocalMap* PrimaryLocalMap = this->GetPrimaryLocalMap();
      return GetLocalMapPose(PrimaryLocalMap);
  }

  Eigen::Matrix4f GetLocalMapPose(const ITMLocalMap* currentLocalMap) const{
      Matrix4f Tw_LocalMap = currentLocalMap->estimatedGlobalPose.GetInvM();
//       return ItmToEigen(currentLocalMap->trackingState->pose_d->GetInvM());
      return ItmToEigen( Tw_LocalMap * (currentLocalMap->trackingState->pose_d->GetInvM()));
  }
  
  /// \brief 返回前一帧到当前帧的变换， transform of from previous to current 
  Eigen::Matrix4f GetLastEgomotion() const {
      return *last_egomotion_;
  }

  ///@brief 得到当前活跃子地图的个数
  int GetLocalActiveMapNumber() const {
      return this->mActiveDataManger->numActiveLocalMaps();
  }

  ///@brief 得到所有子地图的个数
  int GetMapNumber() const {
     return this->mapManager->numLocalMaps();
  }
  
  /// @brief 通过剔除低权重的voxels（较老的voxels）对地图进行正则化(Regularizes),可以减少由于噪声深度图引起的鬼影(artifacts),
  ///        目前只支持GPU和Voxel hashing存储类型
  void Decay(const ITMLocalMap* currentLocalMap) {
    if (voxel_decay_params_.enabled) {
      denseMapper->Decay(currentLocalMap->scene, 
                         currentLocalMap->renderState, 
                         voxel_decay_params_.max_decay_weight,
                         voxel_decay_params_.min_decay_age, 
                         true);
    }
  }

  /// \brief Goes through all remaining visible lists until present time, triggering garbage
  /// collection for all of them. Should not be used mid-sequence.
  /// 遍历所有包括当前时间的可视列表，对这些可视列表进行Decay
  void DecayCatchup(const ITMLocalMap* currentLocalMap) {
    if (voxel_decay_params_.enabled) {
      std::cout << "Will perform voxel decay (GC) for all remaining frames up to the present. "
           << "Continuing to reconstruct past this point may lead to artifacts!" << std::endl;

      for (int i = 0; i < voxel_decay_params_.min_decay_age; ++i) {
        denseMapper->Decay(currentLocalMap->scene,
                           currentLocalMap->renderState,
                           voxel_decay_params_.max_decay_weight,
                           // Set the actual min age to 0 to clean everything up real good.
                           0,
                           false);
      }

      std::cout << "Decay (GC) catchup complete." << std::endl;
    }
  }

  /// \brief Aggressive decay which ignores the minimum age requirement and acts on ALL voxels.
  /// Typically used to clean up finished reconstructions. Can be much slower than `Decay`, even by
  /// a few orders of magnitude if used on the full static map.
  /// 通常用于清理已完成的地图重建，可以比‘Decay’慢得多，即使在高出几个数量级的完整静态地图上
  /*
  void Reap(int max_decay_weight) {
    if (voxel_decay_params_.enabled) {
      denseMapper->Decay(scene, renderState_live, max_decay_weight, 0, true);
    }
  }
  */
  
  size_t GetVoxelSizeBytes() const {
     return sizeof(ITMVoxel);
  }

  /// @brief 得到的是primary local map的内存
  size_t GetPrimaryLocalMapUsedMemoryBytes() {
     ITMLocalMap *primaryLocalMap = this->GetPrimaryLocalMap();
     return GetLocalMapUsedMemoryBytes(primaryLocalMap);
  }

/// @brief 得到的是local map的内存
  size_t GetLocalMapUsedMemoryBytes(ITMLocalMap* currentLocalMap) {
     int num_used_blocks = currentLocalMap->scene->index.getNumAllocatedVoxelBlocks() - currentLocalMap->scene->localVBA.lastFreeBlockId;
     return GetVoxelSizeBytes() * SDF_BLOCK_SIZE3 * num_used_blocks;
  }
  
  size_t GetCurrMapAllocateMemoryBytes(ITMLocalMap* currentLocalMap){
     int num_allocated_blocks = currentLocalMap->scene->index.getNumAllocatedVoxelBlocks();
     return GetVoxelSizeBytes() * SDF_BLOCK_SIZE3 * num_allocated_blocks;
  }
  
  void ResetPrimaryLocalMap() {
   this->denseMapper->ResetScene(this->GetPrimaryLocalMap()->scene);
  }
  
  void ResetLocalMap(ITMLocalMap *currentLocalMap) {
    this->denseMapper->ResetScene(currentLocalMap->scene);
  }
  
  const ITMVisualisationEngine<ITMVoxel, ITMVoxelIndex> *GetVisualizationEngine() const {
     return this->visualisationEngine; 
  }

  size_t GetSavedDecayMemoryBytes() const {
    size_t block_size_bytes = GetVoxelSizeBytes() * SDF_BLOCK_SIZE3;
    size_t decayed_block_count = denseMapper->GetDecayedBlockCount();
    return decayed_block_count * block_size_bytes;
  }  
  
//   void WaitForMeshDump() {
//     if (write_result.valid()) {
//       write_result.wait();
//     }
//   }

  VoxelDecayParams& GetVoxelDecayParams() {
    return voxel_decay_params_;
  }

  bool IsDecayEnabled() {
    return voxel_decay_params_.enabled;
  }

  const VoxelDecayParams& GetVoxelDecayParams() const {
    return voxel_decay_params_;
  }

  /*
  bool IsUsingDepthWeights() const {
    return fusion_weight_params_.depthWeighting;
  }
  */



SUPPORT_EIGEN_FIELDS;

 private:
  ITMUChar4Image *rgb_itm_;
  ITMShortImage  *raw_depth_itm_;
//   ITMLib::Engine::WeightParams fusion_weight_params_;

  cv::Mat3b *rgb_cv_;
  cv::Mat1s *raw_depth_cv_;
  
  cv::Mat3b *rgb_cv_local_map_;
  cv::Mat1s *raw_depth_cv_local_map_;

  Eigen::Matrix4f *last_egomotion_;//前一帧到当前帧的变换矩阵

  // Parameters for voxel decay (map regularization). 类似于像素滤波？
  VoxelDecayParams voxel_decay_params_;
};

} // namespace drivers
} // namespace SparsetoDense


#endif //DENSESLAM_INFINITAMDRIVER_H
