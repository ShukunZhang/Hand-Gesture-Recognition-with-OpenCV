#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
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

//only push back when the screen count down to 3 2 1 

int main(int argc, char** argv)
{
    ros::init(argc, argv, "hand");
    ros::NodeHandle nh;
    ros::Rate loop_rate(5);
    //open default webcam and check for error
    VideoCapture cam(0);
    if (!cam.isOpened()) {
            cout<<"ERROR not opened "<< endl;
            return -1;
    }

    Mat img;
    Mat img_threshold;
    Mat img_gray;
    Mat img_roi;
    namedWindow("Original_image",CV_WINDOW_AUTOSIZE);
   // namedWindow("Gray_image",CV_WINDOW_AUTOSIZE);
    namedWindow("Thresholded_image",CV_WINDOW_AUTOSIZE);
    //namedWindow("ROI",CV_WINDOW_AUTOSIZE);

    char a[40];
    strcpy(a, "Welcome my friends!");
    int fingerCount = 0;
    int previousfinger = 0;
    int frameNumber = 0;
    vector<int> fingerVector;
    int mostFrequentFinger = -1;

    int right = 0;
    int left = 0;

    /* Learning */
    bool gestureLearned = false;
    bool firstTime = true;
    bool learning = false;
    int learningAction = 1;
    int labels[250];
    for (int i = 0; i < 250; i++) {
        labels[i] = i / 50;
    }
    Mat labelsMat(250, 1, CV_32FC1, labels);

    // TODO: replace 3 with the number I want
    size_t trainingData[250][3];
    int trainingImageNumber = 0;

    CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    CvSVM SVM;

    while (ros::ok()) {
        bool readimg = cam.read(img);
        if (!readimg) {
            cout<<"ERROR : cannot read"<<endl;
            return -1;
        }

        Rect roi(150, 50, 450, 350);//starting (x,y) and width, height of the rectangle
        flip(img, img, 1);
        img_roi = img(roi);
        cvtColor(img_roi, img_gray, CV_RGB2GRAY);//convert to grayscale image 

        /* 
        GaussianBlur:
        Size(19,19): Gaussian kernel size 
        ksize.width and ksize.height can differ but they both must be positive and odd
        0.0: Gaussian kernel standard deviation in X direction
        /0: Gaussian kernel standard deviation in Y direction, if it is set to 0,
        sd.y = sd.x
        */
        GaussianBlur(img_gray, img_gray, Size(19, 19), 0.0, 0);

        /*
        threshold:
        0: threshold value
        255: maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV.
        THRESH_OTSU: The function determines the optimal threshold value using 
        the Otsu’s algorithm and uses it instead of the specified thresh value.
        ThRESH_BINARY_INV: if(src(x.y) > thresh){src(x,y) = 0} else {maxval}
        */
        threshold(img_gray, img_threshold, 0, 255, THRESH_BINARY_INV + THRESH_OTSU);
        
        vector<vector<Point> >contours;
        /*
        Vec4i is a type holidng 4 int, (x1,y1,x2,y2)
        each line is a Vec4i, the first two elements are the line's start point (x1,y1) 
        and last two are the line's end point(x2,y2)
        */
        vector<Vec4i>hierarchy;
        /* 
        findContours:
        hierarchy: It has as many elements as the number of contours. For each i-th 
        contour contours[i], the elements hierarchy[i][0], hiearchy[i][1], hiearchy[i][2],
        and hiearchy[i][3] are set to 0-based indices in contours of the next and previous
        contours at the same hierarchical level, the first child contour and the parent contour.
        For full explanation on contour hierarchy, please refer to the following link:
        https://docs.opencv.org/3.4.0/d9/d8b/tutorial_py_contours_hierarchy.html
        CV_RETR_EXTERNAL: (contour retrieval mode)retrieves only the extreme outer contours. 
        It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours. 
        CV_CHAIN_APPROX_SIMPLE: (contour approximation method) compresses horizontal, vertical,
        and diagonal segments and leaves only their end points. For example, an up-right 
        rectangular contour is encoded with 4 points.
        Point():offset by which every contour point is shifted.
        */
        findContours(img_threshold, contours,hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point());
        if (contours.size() > 0) {
            size_t indexOfBiggestContour = -1;
            size_t sizeOfBiggestContour = 0;
                    
            //find the largest contour(hand)
            for (size_t i = 0; i < contours.size(); i++) {
                if (contours[i].size() > sizeOfBiggestContour) { 
                    sizeOfBiggestContour = contours[i].size();
                    indexOfBiggestContour = i;
                }
            }
       
            vector<vector<int> >hull(contours.size()); 
            vector<vector<Point> >hullPoint(contours.size());

            vector<vector<Vec4i> >defects(contours.size());
            // vector<vector<Point> >defectPoint(contours.size());

            vector<vector<Point> >contours_poly(contours.size());
            Point2f rect_point[4];
            vector<RotatedRect>minRect(contours.size());
            vector<Rect> boundRect(contours.size());

            for (size_t i = 0;i < contours.size(); i++) {
                //select the most probable hand contour based
                //on the contour area
                if (contourArea(contours[i]) > 5000) {
                    /*
                    convexHull:
                    The convex hull of a set X of points in the Euclidean plane
                    is the smallest convex set that contains X. 
                    Here Hull[i] is defined as vector<int>, so it will contain
                    the indexes of convexhull points in contours[i]
                    True: clockwise orientaion of point storage. 
                    */
                    convexHull(contours[i], hull[i], true);
                    /*
                    convexityDefects:
                    convexity defect is a cavity in an object (blob, contour) 
                    segmented out from an image. That means an area that do not
                    belong to the object but located inside of the convex hull
                    Here defect[i] is define as vector<Vec4i>. It contains:
                    1.index of point of the contour where the defect begins
                    2.index of point of the contour where the defect ends
                    3.index of the farthest from the convex hull point within the defect
                    4.distance between the farthest point and the convex hull
                    */
                    convexityDefects(contours[i],hull[i],defects[i]);
                    if (indexOfBiggestContour == i) { // hand == true
                        /*
                        Type RotateRect:
                        Each rectangle is specified by the center point (mass center), 
                        length of each side, and the rotation angle in degrees.
                        minAreaRect:
                        Finds a rotated rectangle of the minimum area enclosing 
                        the input 2D point set.
                        */
                        minRect[i] = minAreaRect(contours[i]);
                        for (size_t k = 0; k < hull[i].size(); k++) {
                            //get the index of the convexhull point
                            //from coutour and store the points in hullPoint
                            int contour_point = hull[i][k]; 
                            hullPoint[i].push_back(contours[i][contour_point]);
                        }
                                        
                        //make sure finger count is reset to 0 for every frame
                        fingerCount = 0;

                        drawContours(img_threshold, contours, i, Scalar(255,255,0), 2, 8, vector<Vec4i>(), 0, Point());
                        drawContours(img_threshold, hullPoint, i, Scalar(255,255,0), 1, 8, vector<Vec4i>(), 0, Point());
                        drawContours(img_roi, hullPoint, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());
                        approxPolyDP(contours[i], contours_poly[i], 3, false);
                        boundRect[i] = boundingRect(contours_poly[i]);
                        rectangle(img_roi, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
                        minRect[i].points(rect_point);
                        for (size_t k = 0; k < 4; k++) {
                            line(img_roi, rect_point[k], rect_point[(k+1)%4], Scalar(0, 255, 0), 2, 8);
                        }

                        if (gestureLearned == false) {  
                            if (firstTime == false) {
                                // before learning
                                for (size_t k = 0; k < defects[i].size(); k++) {
                                    if (defects[i][k][3] > 50 * 256) { 
                                        int p_start = defects[i][k][0];
                                        int p_end = defects[i][k][1];
                                        int p_far = defects[i][k][2];
                                        //defectPoint[i].push_back(contours[i][p_far]);
                                        circle(img_roi, contours[i][p_end], 3, Scalar(0,255,0), 2);
                                        circle(img_roi, contours[i][p_start], 3, Scalar(0,255,0), 2);
                                        circle(img_roi, contours[i][p_far], 3, Scalar(0,0,255), 2);
                   
                                        if (contours[i][p_end].x > contours[i][p_far].x) {
                                            right += 1;
                                        } else {
                                            left += 1;
                                        }
                                        fingerCount++;
                                    }
                                }
                            } else {
                                if (learning) { // first time learning
                                    cout << trainingImageNumber << endl;
                                    if (learningAction == 1) {
                                        strcpy(a, "First Action: Going Forward");
                                        cout << "forward\n";
                                    } else if (learningAction == 2) {
                                        strcpy(a, "Second Action: Going Backward");
                                        cout << "back\n";
                                    } else if (learningAction == 3) {
                                        strcpy(a, "Second Action: 360 Degree Spin");
                                        cout << "360\n";
                                    } else if (learningAction == 4) {
                                        strcpy(a, "Second Action: Turn Left");
                                        cout << "left\n";
                                    } else if (learningAction == 5) {
                                        strcpy(a, "Second Action: Turn Right");
                                        cout << "right\n";
                                    } else {
                                        strcpy(a, "Learning Finish!");
                                        Mat trainingDataMat(250, 3, CV_32FC1, trainingData);
                                        SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
                                        gestureLearned = true;
                                        putText(img, a, Point(20,40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 2, 8, false);
                                        loop_rate.sleep();
                                        cout << "break1\n";
                                        break;
                                    }
                                    trainingData[trainingImageNumber][0] = hullPoint[i].size();
                                    trainingData[trainingImageNumber][1] = defects[i].size();
                                    trainingData[trainingImageNumber][2] = contours[i].size();
                                    trainingImageNumber++;
                                    if (trainingImageNumber % 50 == 0)
                                        learningAction++;
                                    putText(img, a, Point(20,40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 2, 8, false);
                                    // loop_rate.sleep();
                                    break;
                                }
                            }
                        } else { // gestureLearned == true
                            cerr << "hi i am here\n";
                            Mat sampleMat = (Mat_<size_t>(1, 3) << hullPoint[i].size(), defects[i].size(), contours[i].size());
                            float response = SVM.predict(sampleMat);
                            if (response == 4.0) {
                                left += 1;
                            } else if (response == 5.0) {
                                right += 1;
                                response = 4.0;
                            }
                            fingerCount = int(response);
                        }
                        
                        fingerVector.push_back(fingerCount);

                        // Action Control
                        if (fingerVector.size() > 20) {
                            if (right > left) {
                                right = 1;
                                left = 0;
                            } else {
                                left = 1;
                                right = 0;
                            }
                            mostFrequentFinger = computeFrequentFinger(fingerVector);
                            if(mostFrequentFinger == 0) {
                                if (firstTime == false) {
                                    strcpy(a, "You are ready to proceed to next action");
                                    previousfinger = 0;
                                } else {
                                    strcpy(a, "You are ready to LEARN!!");
                                    putText(img, a, Point(20,40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 2, 8, false);
                                    cout << "Do you want the dog to learn your gesture? 1 for YES, 0 for NO\n";
                                    bool choice;
                                    cin >> choice;
                                    loop_rate.sleep();
                                    previousfinger = 0;
                                    /* TODO: learning */
                                    if (choice) {
                                        learning = true;
                                        cout << "learning set to true\n";
                                    } else {
                                        gestureLearned = false;
                                        firstTime = false;
                                    }    
                                }
                            }
                            else if (mostFrequentFinger == 1) {
                                if (previousfinger == 0) {
                                    strcpy(a, "TWO fingers! Going Forward!");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 2) {
                                if (previousfinger == 0) {
                                    strcpy(a, "THREE fingers! Going Backward!");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 3) {
                                if (previousfinger == 0) {
                                    strcpy(a, "FOUR fingers! 360 degree turn!");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            else if(mostFrequentFinger == 4) {
                                if (previousfinger == 0) {
                                    if(right) {
                                        strcpy(a, "Turning Right!");
                                        previousfinger = mostFrequentFinger;
                                    }
                                    else if (left) {
                                        strcpy(a, "Turning Left!"); 
                                        previousfinger = mostFrequentFinger;
                                    }
                                    else {
                                        strcpy(a, "High Five!");                                                
                                        previousfinger = mostFrequentFinger;
                                    }
                                }
                            }
                            else {
                                if (previousfinger == 0) {
                                    strcpy(a, "invalid move");
                                    previousfinger = mostFrequentFinger;
                                }
                            }
                            fingerVector.clear();
                            right = 0;
                            left = 0;
                            putText(img, a, Point(20,40), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,0,0), 2, 8, false);
                        }

                    }

                }
            }

            imshow("Original_image",img);
            //imshow("Gray_image",img_gray);
            imshow("Thresholded_image",img_threshold);
            //imshow("ROI",img_roi);
            if (waitKey(30) == 27) {
                    return -1;
            }
        }
    }
    return 0;
}