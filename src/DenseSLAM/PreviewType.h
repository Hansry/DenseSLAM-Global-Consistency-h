
#ifndef DENSESLAM_PREVIEWTYPE_H
#define DENSESLAM_PREVIEWTYPE_H

namespace SparsetoDense {

enum PreviewType {
  kDepth, kGray, kColor, kNormal, kWeight, kRaycastImage, kLatestRaycast, kRaycastDepth, kEnd
};

} // namespace SparsetoDense

#endif //DENSESLAM_PREVIEWTYPE_H
