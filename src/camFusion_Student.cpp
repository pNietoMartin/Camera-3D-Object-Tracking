
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))   enclosingBoxes.push_back(it2);

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
   // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for(const auto & match : kptMatches) {

        const auto &currKeyPoint = kptsCurr[match.trainIdx].pt;

        if (boundingBox.roi.contains(currKeyPoint)) boundingBox.kptMatches.emplace_back(match);
        
    }

    double sum = 0;

    std::cout << "Distance of matches:" << std::endl;

    for (const auto& it : boundingBox.kptMatches) {

        cv::KeyPoint keypointsCurrent = kptsCurr.at(it.trainIdx);
        cv::KeyPoint keypointsPrevious = kptsPrev.at(it.queryIdx);
        
        double dist = cv::norm(keypointsCurrent.pt - keypointsPrevious.pt);
        sum += dist;

        std::cout << dist << " ";
    }

    std::cout << std::endl;
    double mean = sum / boundingBox.kptMatches.size();

    constexpr double ratio = 1.5;

    for (auto it = boundingBox.kptMatches.begin(); it < boundingBox.kptMatches.end();) {
       
        cv::KeyPoint keypointsCurrent = kptsCurr.at(it->trainIdx);
        cv::KeyPoint keypointsPrevious = kptsPrev.at(it->queryIdx);
        double dist = cv::norm(keypointsCurrent.pt - keypointsPrevious.pt);

        if (dist >= mean * ratio) boundingBox.kptMatches.erase(it);
        
        else  it++;
        
    }

}



void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
   
    vector<double> distRatios;
    
    for (auto iterator1 = kptMatches.begin(); iterator1 != kptMatches.end() - 1; iterator1++) {

        cv::KeyPoint keypointOuterCurrent = kptsCurr.at (iterator1->trainIdx);
        cv::KeyPoint keypointOuterPrevious = kptsPrev.at (iterator1->queryIdx);

        for (auto iterator2 = kptMatches.begin() + 1; iterator2 != kptMatches.end(); iterator2++) {

            double minDist = 100.0;

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint keypointInnerCurrent = kptsCurr.at(iterator2->trainIdx);
            cv::KeyPoint keypointInnerPrevious = kptsPrev.at(iterator2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(keypointOuterCurrent.pt - keypointInnerCurrent.pt);
            double distPrev = cv::norm(keypointOuterPrevious.pt - keypointInnerPrevious.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist) {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } 
    }     

    if (distRatios.empty()) {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());

    std::cout << "Distance Ratios:" << std::endl;
    
    for (const auto& dist : distRatios)      std::cout << dist << " ";
    
    std::cout << std::endl;


    long medIndex = floor(distRatios.size() / 2.0);
    
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex];

    std::cout << "medDistRatio = " << medDistRatio << std::endl;

    double dT = 1 / frameRate;
    TTC = - dT / (1 - medDistRatio);
    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
  
 
     int laneWidth = 3;

    /*In Europe, laws and road widths vary by country;
    The minimum widths of lanes are generally between 2.5 to 3.25 metres 
    It has been assumed a widht of 3 m. 
    SOURCE: https://www.academia.edu/12488747/Narrower_Lanes_Safer_Streets_Accepted_Paper_for_CITE_Conference_Regina_June_2015_
    */

    std::vector<float> pointsPrevious;
    std::vector<float> pointsCurrent;

    for(auto iterator = lidarPointsPrev.begin(); iterator != lidarPointsPrev.end(); ++iterator)
    {
        if(abs(iterator->y) < laneWidth/2)      pointsPrevious.push_back(iterator->x);
    }

    for(auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {
        if(abs(it->y) < laneWidth/2)      pointsCurrent.push_back(it->x);
    }


    float minPointPrevious, minPointCurrent;
    int previousSize = pointsPrevious.size();
    int currentSize = pointsCurrent.size();


    if(previousSize > 0 && currentSize > 0)
    {
        for(int i = 0; i<previousSize; i++)  minPointPrevious += pointsPrevious[i];

        for(int j=0; j<currentSize; j++)  minPointCurrent += pointsCurrent[j];
        
    }
    else 
    {
        TTC = NAN;
        return;
    }

    minPointPrevious = minPointPrevious /previousSize;
    minPointCurrent = minPointCurrent /currentSize;

    std::cout<<"Lidar minimum point of previous box: "<<minPointPrevious<<std::endl;
    std::cout<<"Lidar minimum point of current box: "<<minPointCurrent<<std::endl;

    float dt = 1/frameRate;
    TTC = minPointCurrent * dt / (minPointPrevious - minPointCurrent);

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{


    int previousSize = prevFrame.boundingBoxes.size();
    int currentSize = currFrame.boundingBoxes.size();
    int pointCountMatrix [previousSize][currentSize];

    for (auto iterator = matches.begin(); iterator != matches.end() - 1; ++iterator ){

        cv::KeyPoint query = prevFrame.keypoints [iterator->queryIdx];
        auto queryPoint = cv::Point (query.pt.x, query.pt.y) ;
        bool queryFinding = false;

        cv::KeyPoint training = currFrame.keypoints [iterator->trainIdx];
        auto trainPoint = cv::Point(training.pt.x, training.pt.y);
        bool trainFinding = false;

        std::vector<int> queryId, trainId;

        for (int i = 0; i < previousSize; i++) 
        {
            if (prevFrame.boundingBoxes[i].roi.contains(queryPoint))            
             {
                queryFinding = true;
                queryId.push_back(i);
             }
        }
        for (int i = 0; i < currentSize; i++) 
        {
            if (currFrame.boundingBoxes[i].roi.contains(trainPoint))            
            {
                trainFinding= true;
                trainId.push_back(i);
            }
        }

        if (queryFinding && trainFinding) 
        {
            for (auto idPrevious: queryId)
                for (auto idCurrent: trainId)
                     pointCountMatrix[idPrevious][idCurrent] += 1;
        }
    }
   
    for (int i = 0; i < previousSize; i++)
    {  
        int maxCount = 0;
        int idMax = 0;

        for (int j = 0; j < currentSize; j++)
         {
             if (pointCountMatrix[i][j] > maxCount)
             {  
                  maxCount = pointCountMatrix[i][j];
                  idMax = j;
             }
         }
        bbBestMatches[i] = idMax;
    } 
}
