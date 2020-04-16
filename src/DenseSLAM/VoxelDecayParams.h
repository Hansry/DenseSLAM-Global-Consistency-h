
#ifndef DENSESLAM_VOXELDECAYPARAMS_H
#define DENSESLAM_VOXELDECAYPARAMS_H

namespace SparsetoDense {

struct VoxelDecayParams {
  /// Whether to enable voxel decay.
  bool enabled;
  /// Voxels older than this are eligible for decay.
  int min_decay_age;
  /// Voxels with a weight smaller than this are decayed, provided that they are old enough.
  int max_decay_weight;

  VoxelDecayParams(bool enabled, int min_decay_age, int max_decay_weight)
      : enabled(enabled), min_decay_age(min_decay_age), max_decay_weight(max_decay_weight) {}
};

struct SlideWindowParams {
  bool enabled;
  
  /// Voxels older than this are eligible for deletion.
  int max_age;
  
  SlideWindowParams(bool enabled, int max_age)
       : enabled(enabled), max_age(max_age) {}
};

struct OnlineCorrectionParams {
  bool enabled;
  int CorrectionNum;
  int StartToCorrectionNum;
  
  OnlineCorrectionParams(bool enabled, int CorrectionNum, int StartToCorrectionNum) 
        : enabled(enabled), CorrectionNum(CorrectionNum), StartToCorrectionNum(StartToCorrectionNum) {}
};

struct PostPocessParams{
  bool enabled;
  bool show_post_processing;
  float filterThreshold;
  float filterArea;
  
  PostPocessParams(bool enabled, bool show_post_processing, float filterThreshold, float filterArea)
       : enabled(enabled), show_post_processing(show_post_processing), filterThreshold(filterThreshold), filterArea(filterArea) {}
};

struct SaveRaycastDepthParams{
  bool enabled;
  bool compositing_dense;
  int delayNum;
  
  SaveRaycastDepthParams(bool enabled, bool compositing_dense, int delayNum)
        : enabled(enabled), compositing_dense(compositing_dense), delayNum(delayNum) {}
};

struct SaveRaycastRGBParams{
  bool enabled;
  bool compositing_dense;
  int delayNum;
  
  SaveRaycastRGBParams(bool enabled, bool compositing_dense, int delayNum)
        : enabled(enabled), compositing_dense(compositing_dense), delayNum(delayNum) {}
};

} //namespace SparsetoDense

#endif //DENSESLAM_VOXELDECAYPARAMS_H
