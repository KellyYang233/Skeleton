#include "precomp.hpp"
#include "lut.h"

#include <vector>

namespace cv
{

static bool pointInPoints(std::vector<Point> points, Point point)
{
    return std::find(points.begin(), points.end(), point) != points.end();
}

static void setMatAtPoints(InputOutputArray _skel, InputArray _points, int val)
{
    Mat skel = _skel.getMat();
    Mat points = _points.getMat();
    for(MatIterator_<Point> ptIter = points.begin<Point>(); ptIter != points.end<Point>(); ptIter++){
        skel.at<uchar>(*ptIter) = val;
    }
}

static void traverse(InputOutputArray _skel, Point endPoint, InputArray _branchPoints, OutputArray _points)
{
    //skel is assumed to have a border of 0's
    Mat_<Point> points;
    Mat branchPoints = _branchPoints.getMat();
    Mat skel = _skel.getMat();
    Point nbrs[8] = {Point(-1, 0), Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, -1), Point(-1, 1), Point(1, 1), Point(1, -1)};
    Point curPoint = endPoint, lastPoint;
    skel.at<uchar>(curPoint) = 0;
    points.push_back(curPoint);
 
    do{
        lastPoint = curPoint;
        for(int i = 0; i < 8; i++){
            Point newPoint = curPoint + nbrs[i];
            if(skel.at<uchar>(newPoint) > 0){
                curPoint = newPoint;
                skel.at<uchar>(curPoint) = 0;
                points.push_back(curPoint);
                break;
            }
        }

        if(pointInPoints(branchPoints, curPoint)){
            break;
        }
    } while(curPoint != lastPoint);
    setMatAtPoints(skel, points, 1);
    _points.assign(points);
}

static void skelToPoints(InputArray _skel, OutputArray _points)
{
    Mat skel = _skel.getMat();
    Mat_<Point> points;
    for(int i = 0; i < skel.rows; i++){
        for(int j = 0; j < skel.cols; j++){
            if(skel.at<uchar>(i, j) > 0){
                points.push_back(Point(j, i));
            }
        }
    }
    _points.assign(points);
}

void skeleton::prune(InputOutputArray _skel, int minBranchLength)
{
    Mat terminalPoints, endPoints, skel, tskel;
    skel = Mat::zeros(_skel.size() + Size(2, 2), _skel.type());
    tskel = skel(Rect(1, 1, skel.size().width-2, skel.size().height-2));
    _skel.copyTo(tskel);
    skel.convertTo(skel, CV_8UC1);

    skeleton::branchPoints(skel, terminalPoints);
    skelToPoints(terminalPoints, terminalPoints);
    skeleton::endPoints(skel, endPoints);
    skelToPoints(endPoints, endPoints);

    for(int i = 0; i < endPoints.rows; i++){
        Point endPoint = endPoints.at<Point>(i);
        terminalPoints.push_back(endPoint);
    }

    std::vector<Mat> branches;

    for(MatIterator_<Point> endPtIter = endPoints.begin<Point>(); endPtIter != endPoints.end<Point>(); endPtIter++){
        Mat points;
        traverse(skel, *endPtIter, terminalPoints, points);
        branches.push_back(points);
    }

    for(std::vector<Mat>::iterator branchIter = branches.begin(); branchIter != branches.end(); branchIter++){
        Mat branch = *branchIter;
        Point finalPoint;
        if(0 < branch.rows && branch.rows < minBranchLength && !pointInPoints(endPoints, finalPoint=branch.at<Point>(branch.rows-1))){
            setMatAtPoints(skel, branch, 0);
            skel.at<uchar>(finalPoint) = 1;
        }
    }

    tskel.copyTo(_skel);
}

void skeleton::prune(InputOutputArray _skel, float minBranchLength)
{
    int skelLength = countNonZero(_skel);
    int minLength = minBranchLength*skelLength;
    prune(_skel, minLength);
}

}
