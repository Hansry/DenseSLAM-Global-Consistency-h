
#ifndef DYNSLAM_PREVIEWTYPE_H
#define DYNSLAM_PREVIEWTYPE_H

namespace SparsetoDense {

enum PreviewType {
  kDepth, kGray, kColor, kNormal, kWeight, kRaycastImage, kLatestRaycast, kRaycastDepth, kEnd
};

} // namespace SparsetoDense

#endif //DYNSLAM_PREVIEWTYPE_H
