#include "precomp.hpp"
#include "lut.h"

namespace cv
{

static float traverse(InputOutputArray _skel, Point endPoint, InputArray _branchPoints)
{
    //skel is assumed to have a border of 0's
    Mat branchPointsMat = _branchPoints.getMat();
    Point *branchPoints = branchPointsMat.ptr<Point>(0);
    Mat skel = _skel.getMat();
    Point nbrs[8] = {Point(-1, 0), Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, -1), Point(-1, 1), Point(1, 1), Point(1, -1)};
    Point curPoint = endPoint, lastPoint;
 
    float len = 0; 
    do{
        skel.at<uchar>(curPoint) = 0;
        lastPoint = curPoint;
        for(int i = 0; i < 8; i++){
            if(skel.at<uchar>(curPoint + nbrs[i]) > 0){
                curPoint += nbrs[i];
                len += 1;
                break;
            }
        }

        bool done = false;
        for(int i = 0; i < _branchPoints.rows(); i++){
            if(curPoint == branchPoints[i]){
                done = true;
                break;
            }
	}
        if(done){
            break;
        }
    } while(curPoint != lastPoint);
        
    return 0; 
}

}
