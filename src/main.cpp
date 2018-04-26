#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ros/ros.h>

using namespace cv;
using namespace std;

int computeFrequentFinger(vector<int> fingerVector) {
    sort(fingerVector.begin(), fingerVector.end());
    int mostFrequentFinger;
    int thisNumberFrequency = 1;
    int highestFrequency = 1;
    mostFrequentFinger = fingerVector[0];
    for(int i = 1; i < fingerVector.size(); i++) {
        if(fingerVector[i - 1] != fingerVector[i]) {
            if(thisNumberFrequency > highestFrequency) {
                mostFrequentFinger = fingerVector[i - 1];  
                highestFrequency = thisNumberFrequency;
            }
            thisNumberFrequency = 0;
        }
        thisNumberFrequency++;
    }
    if(thisNumberFrequency > highestFrequency) {
        mostFrequentFinger = fingerVector[fingerVector.size() - 1];   
    }
    return mostFrequentFinger;
}

int main( int argc, char** argv)
{
    ros::init(argc, argv, "wtfname");

    VideoCapture cam(0);
    if(!cam.isOpened()){
        cout<<"ERROR not opened "<< endl;
        return -1;
    }

    Mat img;
    Mat img_threshold;
    Mat img_gray;
    Mat img_roi;
    namedWindow("Original_image",CV_WINDOW_AUTOSIZE);
    namedWindow("Gray_image",CV_WINDOW_AUTOSIZE);
    namedWindow("Thresholded_image",CV_WINDOW_AUTOSIZE);
    namedWindow("ROI",CV_WINDOW_AUTOSIZE);

    char a[40];
    int finger =0;
    int previousfinger = 0;
    int frameNumber = 0;
    vector<int> fingerVector;
    int mostFrequentFinger = -1;
    while(1){
        bool b=cam.read(img);
        if(!b){
            cout<<"ERROR : cannot read"<<endl;
            return -1;
        }

        Rect roi(340,100,270,270);//starting (x,y) and width, height of the rectangle
        img_roi=img(roi);
        cvtColor(img_roi,img_gray,CV_RGB2GRAY);//convert to grayscale image 
        
        /* 
        GaussianBlur:
        Size(19,19): Gaussian kernel size 
        ksize.width and ksize.height can differ but they both must be positive and odd
        0.0: Gaussian kernel standard deviation in X direction
        /0: Gaussian kernel standard deviation in Y direction, if it is set to 0,
        sd.y = sd.x
        */
        GaussianBlur(img_gray,img_gray,Size(19,19),0.0,0);


        threshold(img_gray,img_threshold,0,255,THRESH_BINARY_INV+THRESH_OTSU);

        vector<vector<Point> >contours;
        /*
        Vec4i is a type holidng 4 int, (x1,y1,x2,y2)
        each line is a Vec4i, the first two elements are the line's start point (x1,y1) 
        and last two are the line's end point(x2,y2)
        */
        vector<Vec4i>hierarchy;
        findContours(img_threshold,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,Point());
        if(contours.size()>0){
            size_t indexOfBiggestContour = -1;
            size_t sizeOfBiggestContour = 0;

            for (size_t i = 0; i < contours.size(); i++){
	           if(contours[i].size() > sizeOfBiggestContour){
		       sizeOfBiggestContour = contours[i].size();
		       indexOfBiggestContour = i;
	          }
            }
            vector<vector<int> >hull(contours.size());
            vector<vector<Point> >hullPoint(contours.size());
            vector<vector<Vec4i> >defects(contours.size());
            vector<vector<Point> >defectPoint(contours.size());
            vector<vector<Point> >contours_poly(contours.size());
            Point2f rect_point[4];
            vector<RotatedRect>minRect(contours.size());
            vector<Rect> boundRect(contours.size());
            for(size_t i=0;i<contours.size();i++){
                if(contourArea(contours[i])>5000){
                    convexHull(contours[i],hull[i],true);
                    convexityDefects(contours[i],hull[i],defects[i]);
                    if(indexOfBiggestContour==i){
                        minRect[i]=minAreaRect(contours[i]);
                        for(size_t k=0;k<hull[i].size();k++){
                            int ind=hull[i][k];
                            hullPoint[i].push_back(contours[i][ind]);
                        }
                        finger =0;

                        for(size_t k=0;k<defects[i].size();k++){
                            if(defects[i][k][3]>13*256){
                             /*   int p_start=defects[i][k][0];   */
                                int p_end=defects[i][k][1];
                                int p_far=defects[i][k][2];
                                defectPoint[i].push_back(contours[i][p_far]);
                                circle(img_roi,contours[i][p_end],3,Scalar(0,255,0),2);
                                finger++;
                            }

                        }
                        fingerVector.push_back(finger);
                        // Frame number TBD
                        if (fingerVector.size() > 20){
                            mostFrequentFinger = computeFrequentFinger(fingerVector);
                            if(mostFrequentFinger == 1) {
                                strcpy(a, "You are ready to proceed to next action");
                                previousfinger = 1;
                            }
                            else if(mostFrequentFinger == 2) {
                                if (previousfinger == 1) {
                                    strcpy(a, "Some action to be decided");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 3) {
                                if (previousfinger == 1) {
                                    strcpy(a, "Some action to be decided");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 4) {
                                if (previousfinger == 1) {
                                    strcpy(a, "Some action to be decided");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 5) {
                                if (previousfinger == 1) {
                                    strcpy(a, "Some action to be decided");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else {
                                if (previousfinger == 1) {
                                    strcpy(a, "Some action to be decided");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            fingerVector.clear();
                        }

                        putText(img,a,Point(70,70),CV_FONT_HERSHEY_SIMPLEX,3,Scalar(255,0,0),2,8,false);
                        drawContours(img_threshold, contours, i,Scalar(255,255,0),2, 8, vector<Vec4i>(), 0, Point() );
                        drawContours(img_threshold, hullPoint, i, Scalar(255,255,0),1, 8, vector<Vec4i>(),0, Point());
                        drawContours(img_roi, hullPoint, i, Scalar(0,0,255),2, 8, vector<Vec4i>(),0, Point() );
                        approxPolyDP(contours[i],contours_poly[i],3,false);
                        boundRect[i]=boundingRect(contours_poly[i]);
                        rectangle(img_roi,boundRect[i].tl(),boundRect[i].br(),Scalar(255,0,0),2,8,0);
                        minRect[i].points(rect_point);
                        for(size_t k=0;k<4;k++){
                            line(img_roi,rect_point[k],rect_point[(k+1)%4],Scalar(0,255,0),2,8);
                        }
                    }
                }
            }
            imshow("Original_image",img);
            imshow("Gray_image",img_gray);
            imshow("Thresholded_image",img_threshold);
            imshow("ROI",img_roi);
            if(waitKey(30)==27){
                  return -1;
             }
        }
    }
    return 0;
}