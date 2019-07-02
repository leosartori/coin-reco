/**
    Coin Detector and Recognizer - Exam of Computer Vision 2018/2019 UNIPD

    @author Leonardo Sartori (leonardo.sartori.1@studenti.unipp.it)
    @version 1.0
*/

#include <Recognizer.h>

using namespace cv;
using namespace std;

Recognizer::Recognizer(string image_path) {

    //Load the image
    this->image_path = image_path;
    this->image = imread(image_path, CV_LOAD_IMAGE_UNCHANGED);
    this->image_out = this->image.clone();

    if (image.empty()) {
        cout << "Error: image not found" << endl;
    }

    //display the image
    namedWindow("Original", WINDOW_NORMAL);
    imshow("Original", this->image);
    waitKey(0);
    cvDestroyWindow("Original");

}


void Recognizer::preprocess(int choice, string path){
    // selection of circle detection method
    switch(choice){
        case 1:
            hough_preprocess(path);
            break;
        case 2:{
            double canny_threshold = 800;
            double circle_threshold = 0.2;
            int numIterations = image.size().height * image.size().width / 100;
            ransac_preproc(path, canny_threshold, circle_threshold, numIterations);
            break;
        }
        default:
            cout << "Invalid choice, using Hough preprocessing" << endl;
            hough_preprocess(path);
            break;
    }
}


void Recognizer::hough_preprocess(string path) {

    Mat img_gray;
    cvtColor(image, img_gray, CV_BGR2GRAY);

    // GAUSSIAN BLUR: in order to reduce noise and false circles
    Mat gray_blur;
    GaussianBlur(img_gray, gray_blur, Size(15, 15), 10);
    namedWindow("Blur", WINDOW_NORMAL);
    imshow("Blur", gray_blur);
    waitKey(0);
    cvDestroyWindow("Blur");

    // Otsu thresholding: to obtain binary image from histogram analysis
    Mat thresh;
    double otsu_tresh = threshold(gray_blur, thresh, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    double high_thresh_val = otsu_tresh, lower_thresh_val = otsu_tresh / 2;

    //cout << "high: " << high_thresh_val << " low: " << lower_thresh_val << endl;

    // Alternative to Otsu: adaptive thresholding, divides pixels in above or below variable threshold (black or white)
    // adaptiveThreshold(gray_blur, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 1);

    namedWindow("Thresh", WINDOW_NORMAL);
    imshow("Thresh", thresh);
    waitKey(0);
    cvDestroyWindow("Thresh");

    // Opening: erosion followed by dilation. It is useful in removing noise
    Mat kernel = Mat::ones(3, 3, CV_8U);
    Mat opening;
    morphologyEx(thresh, opening, MORPH_OPEN, kernel, Point(-1, -1), 5);
    namedWindow("Opening", WINDOW_NORMAL);
    imshow("Opening", opening);
    waitKey(0);
    cvDestroyWindow("Opening");

    // Canny: edges detector
    Mat edges;
    Canny(img_gray, edges, high_thresh_val * 3, lower_thresh_val * 4, 3, true);
    namedWindow("Canny", WINDOW_NORMAL);
    imshow("Canny", edges);
    waitKey(0);
    destroyWindow("Canny");

    // HOUGH TRANSFORM

    int max_radius = min(image.size().height, image.size().width) / 4;
    int min_dist = min(image.size().height, image.size().width) / 5.5;
    int min_radius = min_dist * 1.2;

    // OK for more coins with occlusions
    //HoughCircles(closing,coin,CV_HOUGH_GRADIENT,1,50,1000,10,10,0);

    // PERFECT ON 1.jpg:
    HoughCircles(edges, coin, CV_HOUGH_GRADIENT, 1, 500, high_thresh_val, 10, 100, max_radius);
    // DECENT ON everything
    //HoughCircles(edges, coin, CV_HOUGH_GRADIENT, 1.5, min_dist, high_thresh_val * 3, 30, min_radius, max_radius);

    cout << endl << "The number of coins is: " << coin.size() << endl;
    if (coin.size() > MAX_COINS){
        cout <<"But only stronger " << MAX_COINS << "will be analyzed" << endl << endl;
        coin.resize(MAX_COINS);
    }

    if (coin.size() > 0) {

        // DEBUG
        //for (size_t i = 0; i < coin.size(); i++)
        //    cout << i << ": " << coin[i];

        // clean the directory used to perform classification by Python script
        String rm = "exec rm -r " + path + "*";
        system(rm.c_str());
        String mkdir = "exec mkdir " + path + "coins";
        system(mkdir.c_str());

        // To draw and save as images the detected circles.
        save_and_draw(path + "coins/");

        // Display circles found
        namedWindow("Coin Counter", WINDOW_NORMAL);
        imshow("Coin Counter", image_out);
        waitKey(0); // Wait for infinite time for a key press.

    }
}


void Recognizer::save_and_draw(string path){

    for (size_t i = 0; i < coin.size(); i++) {
        // get center
        Point center(cvRound(coin[i][0]), cvRound(coin[i][1]));
        // get the radius
        int radius = cvRound(coin[i][2]);

        // cout << "Circle " << i + 1 << " : " << center << " Diameter : " << 2 * radius << endl;

        // check if subimages are inside the original image (detected circle can go outside it)
        int x1 = max(0, center.x - radius);
        int x2 = max(0, center.y - radius);
        int x3 = min(image.size().width, center.x + radius);
        int x4 = min(image.size().height, center.y + radius);

        // create coin subimage
        Mat single_coin = image(Rect(Point(x1, x2), Point(x3, x4)));
        coins_img.push_back(single_coin);

        // save the coin frame
        imwrite(path + to_string(i) + ".jpg", single_coin);

        //Draw circle center
        circle(image_out, center, 3, Scalar(0, 255, 0), -1, 8, 0);
        //Draw circle outline.
        circle(image_out, center, radius, Scalar(0, 0, 255), 3, 8, 0);

        //namedWindow("Coin Crop", WINDOW_NORMAL);
        //imshow("Coin Crop", single_coin);
        //waitKey(0);
    }
    cout <<"Single coin images saved in " << path << endl;
}


void Recognizer::ransac_preproc(string path, double canny_threshold, double circle_threshold, int numIterations)
{
    // TODO: remove continue

    // Edge Detection
    Mat edges;
    Canny(image, edges, MAX(canny_threshold/2,1), canny_threshold, 3);

    // Create point set from Canny Output
    vector<Point2d> points;
    for(int r = 0; r < edges.rows; r++)
    {
        for(int c = 0; c < edges.cols; c++)
        {
            if(edges.at<unsigned char>(r,c) == 255)
            {
                points.push_back(Point2d(c,r));
            }
        }
    }

    // 4 point objects to hold the random samples
    Point2d pointA;
    Point2d pointB;
    Point2d pointC;
    Point2d pointD;

    // distances between points
    double AB;
    double BC;
    double CA;
    double DC;

    // variables for line equations y = mx + b
    double m_AB;
    double b_AB;
    double m_BC;
    double b_BC;

    // variables for line midpoints
    double XmidPoint_AB;
    double YmidPoint_AB;
    double XmidPoint_BC;
    double YmidPoint_BC;

    // variables for perpendicular bisectors
    double m2_AB;
    double m2_BC;
    double b2_AB;
    double b2_BC;

    // RANSAC
    RNG rng; // random number generator
    const int min_point_separation = 10; // if points randomly chosen are closer than that, skip them
    const int colinear_tolerance = 1; // make sure points are not on a line
    const int radius_tolerance = 10; // tolerance on points considered on the circumference
    const int points_threshold = 500; // prints warning if we are using a number of total points below that
    const double min_radius = 10; //minimum radius for a circle to not be rejected

    // used to find intersection
    int x,y;
    // circle variables
    Point2d center;
    double radius;

    // Iterate
    for(int iteration = 0; (iteration < numIterations) && (coin.size() < MAX_COINS); iteration++)
    {
        //cout << "RANSAC iteration: " << iteration << endl;

        // get 4 random points
        pointA = points[rng.uniform((int)0, (int)points.size())];
        pointB = points[rng.uniform((int)0, (int)points.size())];
        pointC = points[rng.uniform((int)0, (int)points.size())];
        pointD = points[rng.uniform((int)0, (int)points.size())];

        // calc lines
        AB = norm(pointA - pointB);
        BC = norm(pointB - pointC);
        CA = norm(pointC - pointA);
        DC = norm(pointD - pointC);

        // one or more random points are too close together
        if(AB < min_point_separation || BC < min_point_separation || CA < min_point_separation || DC < min_point_separation) continue;

        //find line equations for AB and BC
        //AB
        m_AB = (pointB.y - pointA.y) / (pointB.x - pointA.x + 0.000000001); //avoid divide by 0
        b_AB = pointB.y - m_AB*pointB.x;

        //BC
        m_BC = (pointC.y - pointB.y) / (pointC.x - pointB.x + 0.000000001); //avoid divide by 0
        b_BC = pointC.y - m_BC*pointC.x;


        //test colinearity (ie the points are not all on the same line)
        if(abs(pointC.y - (m_AB*pointC.x + b_AB + colinear_tolerance)) < colinear_tolerance) continue;

        //find perpendicular bisector
        //AB
        //midpoint
        XmidPoint_AB = (pointB.x + pointA.x) / 2.0;
        YmidPoint_AB = m_AB * XmidPoint_AB + b_AB;
        //perpendicular slope
        m2_AB = -1.0 / m_AB;
        //find b2
        b2_AB = YmidPoint_AB - m2_AB*XmidPoint_AB;

        //BC
        //midpoint
        XmidPoint_BC = (pointC.x + pointB.x) / 2.0;
        YmidPoint_BC = m_BC * XmidPoint_BC + b_BC;
        //perpendicular slope
        m2_BC = -1.0 / m_BC;
        //find b2
        b2_BC = YmidPoint_BC - m2_BC*XmidPoint_BC;

        //find intersection = circle center
        x = (b2_AB - b2_BC) / (m2_BC - m2_AB);
        y = m2_AB * x + b2_AB;
        center = Point2d(x,y);
        radius = norm(center - pointB);

        // check if radius is larger enough
        if (radius < min_radius) continue;

        //check if the 4 point is on the circle
        if(abs(norm(pointD - center) - radius) > radius_tolerance) continue;

        // vote
        vector<int> votes;
        vector<int> no_votes;
        for(int i = 0; i < (int)points.size(); i++)
        {
            double vote_radius = norm(points[i] - center);

            // the point is at distance similar to radius
            if(abs(vote_radius - radius) < radius_tolerance)
            {
                votes.push_back(i);
            }
            else
            {
                no_votes.push_back(i);
            }
        }

        // check votes vs circle_threshold (number of points near the circumference / points on circumference)
        if( (float)votes.size() / (2.0*CV_PI*radius) >= circle_threshold )
        {
            coin.push_back(Vec3f(x,y,radius));

            // remove points from the set so they can't vote on multiple coin
            vector<Point2d> new_points;
            for(int i = 0; i < (int)no_votes.size(); i++)
            {
                new_points.push_back(points[no_votes[i]]);
            }
            points.clear();
            points = new_points;
        }

        // stop RANSAC if there are few points left
        if((int)points.size() < points_threshold)
            cout << "Too little points remaining, RANSAC unstable: restart reducing iterations" << endl;
    }

    cout << "Circles found: " << coin.size() << endl;

    if (!coin.empty()){

        // clean the directory used to perform classification by Python script
        String rm = "exec rm -r " + path + "*";
        system(rm.c_str());
        String mkdir = "exec mkdir " + path + "coins";
        system(mkdir.c_str());

        // To draw and save as images the detected circles.
        save_and_draw(path + "coins/");

        // Display circles found
        namedWindow("Coin Counter", WINDOW_NORMAL);
        imshow("Coin Counter", image_out);
        waitKey(0); // Wait for infinite time for a key press.
    }
}


vector<string> Recognizer::predict(string path){

    if (coins_img.empty()){
        cout << "No coins detected, aborting recogniction" << endl;
    }
    else{

        // run Python script to creates CSV with predictions, about subimages previously found
        system("python test.py");
        cout << endl;

        // read predictions from CSV created by test.py
        ifstream f(path + "pred.csv");
        if (!f.is_open()){
            cout << "error opening file." << endl;
        }
        else{
            string single_pred;
            while(getline(f, single_pred, ',')) {
                pred.push_back(single_pred);
            }
        }
    }

    return pred;
}