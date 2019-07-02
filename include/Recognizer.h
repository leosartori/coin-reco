/**
    Coin Detector and Recognizer - Exam of Computer Vision 2018/2019 UNIPD

    @author Leonardo Sartori (leonardo.sartori.1@studenti.unipp.it)
    @version 1.0
*/

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
using namespace cv;

// Methods
class Recognizer{

    public:

        /**
        Constructor
        @param path of the image to analyze
        */
        Recognizer(string image_path);

        /**
        Wrapper for selection of circle detection method
        @param choice of circle detection method (1=Hough, 2=RANSAC)
        @param path in which to save subimages
        */
        // preprocess images: find circles and save subimages of single coins to be classified
        void preprocess(int choice, string path);

        /**
        Circle detection using OpenCV HoughCircle (with parameters on image proprieties)
        @param path in which to save subimages
        */
        void hough_preprocess(string path);

        /**
        RANSAC circle detection based off of "An Efﬁcient Randomized Algorithm for Detecting Circles" by Chen and Chung
        @param path in which to save subimages
        @param upper threshold of Canny Edge detection algorithm
        @param threshold on ratio (points near the circumference / ideal points on circumference)
        @param number of iterations of RANSAC
        */
        void ransac_preproc(string path, double canny_threshold, double circle_threshold, int numIterations);

        /**
        Saves found circles as subimages of the original and draws circumenferences on a copy of original, then display it
        @param path in which to save subimages
        */
        void save_and_draw(string path);

        /**
        Calls Python script to perform classification of coins, retrieves CSV created by this last one and parse it
        @param basedir of classification path
        @return vector of strings parsed from the CSV (vector of labels predicted)
        */
        vector<string> predict(string path);

        /**
        Support function to check if a string ends with another string
        @param string to analyze
        @param ending string
        @return true if the first string ends with the second
        */
        static bool endsWith(const string& str, const string& suffix)
        {
            return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
        }
private:

        // Path to original image
        string image_path;

        // Original image to analyze
        Mat image;

        // Output image in which to draw detected circle, to display result of detection
        Mat image_out;

        // Vector to store the details of coins circles
        vector<Vec3f> coin;

        // Vector to store subimages of dected coins (circles)
        vector<Mat> coins_img;

        // Vector of string label of Euro coins found in the image
        vector<string> pred;

        // maximum number of coins detectable
        const int MAX_COINS = 15;

};
