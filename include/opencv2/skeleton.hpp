#ifndef OPENCV_SKELETON_HPP
#define OPENCV_SKELETON_HPP

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

/**
  @defgroup skeleton Skeleton
  @{
    @defgroup skeletonize Skeletonization
       Functions to skeletonize an image.
    @defgroup features Skeleton Features
       Functions to manipulate skeletons.
    @defgroup misc Miscellaneous
       Miscellaneous functions.
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
@param type algorithm to calculate skeleton
 */
CV_EXPORTS_W void skeletonize( InputArray src, OutputArray dst, int type = SKEL_ZHANGSUEN);

//! @} skeletonize

namespace skeleton{
//! @addtogroup features
//! @{
/** @brief Returns an image where branch points are 1 and all others are 0.

The function find the branch points of a skeleton.

@param skel input skeleton
@param branch output image of the same size and type as skel containing branch points only.
 */
CV_EXPORTS_W void branchPoints(InputArray skel, OutputArray branch);

//! @} features

//! @addtogroup misc
//! @{
/** @brief Returns the version of this module.
*/
CV_EXPORTS_W String version();
}
//! @} misc
//! @} skeleton

} // cv
#endif
