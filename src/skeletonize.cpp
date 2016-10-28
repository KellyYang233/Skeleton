#include "precomp.hpp"

#include <iostream>

namespace cv
{

static int countNeighbors(Mat mat, int x, int y)
{
    int count = 0, power = 1;
    int iOrder[8] = {-1, 0, 1, 1, 1, 0, -1, -1};
    int jOrder[8] = {-1, -1, -1, 0, 1, 1, 1, 0};
    for(int pos = 0; pos < 8; pos++){
	int i = iOrder[pos], j = jOrder[pos];
        if(x+i < 0 || x+i >= mat.rows || y+j < 0 || y+j >= mat.cols){
        }else{
            count += power * mat.at<uchar>(x+i, y+j);
        }
        power *= 2;
    }
    return count;
}

static int applyLUTNeighbors(InputArray _src, OutputArray _dst, uchar *lut)
{
    Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();
    int numChanges = 0;
    for( int i = 0; i < src.rows; i++ ){
        for( int j = 0; j < src.cols; j++ ){
	    if(src.at<uchar>(i, j) == 0){
                dst.at<uchar>(i, j) = 0;
            }else{
                dst.at<uchar>(i, j) = lut[countNeighbors(src, i, j)];
            }
            if(dst.at<uchar>(i, j) != src.at<uchar>(i, j)){
                numChanges++;
            }
        }
    }
    return numChanges;
}

static void zhangSuenThinning(InputArray _src, OutputArray _dst)
{
    uchar lut1[256] = {
        1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,
        1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1
    };
    uchar lut2[256] = {
        1,1,1,0,1,1,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,
        1,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,
        1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        0,0,1,0,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,0,0,1,1,0,1,1,1
    };

    Mat src, dst;
    threshold(_src, src, 0, 1, THRESH_BINARY);
    src.convertTo(src, CV_8UC1);

    int count;
    do{
        count = applyLUTNeighbors(src, dst, lut1);
        dst.copyTo(src);
        count += applyLUTNeighbors(src, dst, lut2);
    } while(count > 0);
    _dst.assign(dst);
}

void skeletonize(InputArray _src, OutputArray _dst, int type)
{
    switch(type){
        case SKEL_ZHANGSUEN:
            zhangSuenThinning(_src, _dst);
            break;
    }
}
}
