#include "precomp.hpp"
#include "lut.h"

namespace cv
{

static int applyLUTNeighbors(InputArray _src, OutputArray _dst, uchar *lut)
{
    Mat src = _src.getMat();
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(src.channels() == 1);
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    int numChanges = 0;
    uchar *dstptr = dst.ptr<uchar>(0);
    for(int j = 0; j < src.cols; j++){
        dstptr[j] = 0;
    }
    for(int i = 1; i < src.rows-1; i++){
        dstptr = dst.ptr<uchar>(i);
        dstptr[0] = 0;
        dstptr[src.cols-1] = 0;
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
    dstptr = dst.ptr<uchar>(src.rows-1);
    for(int j = 0; j < src.cols; j++){
        dstptr[j] = 0;
    }
    return numChanges;
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
    uchar lut1[256] = ZHANGSUEN_LUT1;
    uchar lut2[256] = ZHANGSUEN_LUT2;

    Mat src, dst;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);

    while(applyLUTNeighbors(src, dst, lut1) + applyLUTNeighbors(dst, src, lut2));
    _dst.assign(src);
}

static void guoHallThinning(InputArray _src, OutputArray _dst)
{
    uchar lut1[256] = GUOHALL_LUT1;
    uchar lut2[256] = GUOHALL_LUT2;

    Mat src, dst;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);

    while(applyLUTNeighbors(src, dst, lut1) + applyLUTNeighbors(dst, src, lut2));
    _dst.assign(src);
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
    uchar lut[256] = BRANCH_LUT;

    Mat src;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);
    applyLUTNeighbors(src, _dst, lut);
}

void skeleton::endPoints(InputArray _src, OutputArray _dst)
{
    uchar lut[256] = ENDPTS_LUT;

    Mat src;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);
    applyLUTNeighbors(src, _dst, lut);
}
}
