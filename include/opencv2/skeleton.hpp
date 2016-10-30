#ifndef OPENCV_SKELETON_HPP
#define OPENCV_SKELETON_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/**
  @defgroup skeleton Skeleton
  @{
    @defgroup skeletonize Skeletonization
       Functions here create skeletonization on images.
    @defgroup skeleton_c C API
  @}
*/

namespace cv
{

/** @addtogroup skeleton
@{
*/

//! @addtogroup skeletonize
//! @{

//! type of skeletonization operation
enum SkeletonTypes{
    SKEL_MORPHOLOGICAL   = 0,
    SKEL_ZHANGSUEN       = 1,
    SKEL_GUOHALL         = 2
};

/** @brief Returns a skeletonized image from a binary image.

The function computes the given skeleton on a binary image.

@param src input image
@param dst output image of the same size and type as src.
@param type skeletonization algorithm to use to calculate skeleton
 */
CV_EXPORTS_W void skeletonize( InputArray src, OutputArray dst, int type = SKEL_ZHANGSUEN);

//! @} skeletonize_filter
namespace skeleton{
/** @brief Returns the version of this module.
*/
CV_EXPORTS_W String version();
}
//! @} skeleton

} // cv
#endif
