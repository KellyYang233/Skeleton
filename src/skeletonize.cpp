#include "precomp.hpp"
#include "lut.h"

namespace cv
{

static void expandBorder(InputOutputArray _src)
{
    Mat src = Mat::zeros(_src.size() + Size(2, 2), _src.type());
    _src.getMat().copyTo(src(Rect(1, 1, _src.size().width, _src.size().height)));
    _src.assign(src);
}

static void removeBorder(InputOutputArray _src)
{
    Mat src;
    _src.getMat()(Rect(1, 1, _src.size().width-2, _src.size().height-2)).copyTo(src);
    _src.assign(src);
}


static int applyLUTNeighbors(InputArray _src, OutputArray _dst, uchar *lut)
{
    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(src.channels() == 1);
    int numChanges = 0;
    for(int i = 1; i < src.rows-1; i++){
        uchar *dstptr = dst.ptr<uchar>(i);
        uchar *up = src.ptr<uchar>(i-1);
        uchar *cur = src.ptr<uchar>(i);
        uchar *down = src.ptr<uchar>(i+1);
        for(int j = 1; j < src.cols-1; j++){
            uchar curj = cur[j];
            if(cur[j] == 0){
                dstptr[j] = 0;
            }else{
                dstptr[j] = lut[  1*  up[j-1] +   2*  up[j] +   4*  up[j+1] +
                                128* cur[j-1] +                 8* cur[j+1] +
                                 64*down[j-1] +  32*down[j] +  16*down[j+1]];
            }
            if(dstptr[j] != curj){
                numChanges++;
            }
        }
    }
    return numChanges;
}

static void simpleLut(InputArray _src, OutputArray _dst, uchar *lut)
{
    Mat src;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);
    expandBorder(src);
    Mat dst(src.size(), src.type());
    applyLUTNeighbors(src, dst, lut);
    removeBorder(dst);
    _dst.assign(dst);
}

static void twoPassThinning(InputArray _src, OutputArray _dst, uchar *lut1, uchar *lut2)
{
    Mat src;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);
    expandBorder(src);
    Mat dst(src.size(), src.type());

    while(applyLUTNeighbors(src, dst, lut1) + applyLUTNeighbors(dst, src, lut2));

    removeBorder(src);
    _dst.assign(src);
}


static void morphologicalThinning(InputArray _src, OutputArray _dst)
{
    Mat src, eroded, opening;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);
    Mat skel = Mat::zeros(src.size(), src.type());
    Mat element = getStructuringElement(MORPH_CROSS, Size(3, 3));

    do{
        erode(src, eroded, element);
        dilate(eroded, opening, element);
        skel |= src - opening;
        eroded.copyTo(src);
    } while (countNonZero(src) != 0);

    _dst.assign(skel);
}

static void zhangSuenThinning(InputArray _src, OutputArray _dst)
{
    twoPassThinning(_src, _dst, ZHANGSUEN_LUT1, ZHANGSUEN_LUT2);
}

static void guoHallThinning(InputArray _src, OutputArray _dst)
{
    twoPassThinning(_src, _dst, GUOHALL_LUT1, GUOHALL_LUT2);
}

void skeletonize(InputArray _src, OutputArray _dst, int type)
{
    switch(type){
        case SKEL_MORPHOLOGICAL:
            morphologicalThinning(_src, _dst);
            break;
        case SKEL_ZHANGSUEN:
            zhangSuenThinning(_src, _dst);
            break;
        case SKEL_GUOHALL:
            guoHallThinning(_src, _dst);
            break;
    }
}

void skeleton::branchPoints(InputArray _src, OutputArray _dst)
{
    simpleLut(_src, _dst, BRANCH_LUT);
}

void skeleton::endPoints(InputArray _src, OutputArray _dst)
{
    simpleLut(_src, _dst, ENDPTS_LUT);
}
}
