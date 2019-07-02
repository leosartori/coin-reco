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
        Encodes a single digit of a POSTNET "A" bar code.

        @param digit the single digit to encode.
        @return a bar code of the digit using "|" as the long
        bar and "," as the half bar.
        */
        // preprocess images: find circles and save subimages of single coins to be classified
        void preprocess(int choice, string path);

        /**
        Encodes a single digit of a POSTNET "A" bar code.

        @param digit the single digit to encode.
        @return a bar code of the digit using "|" as the long
        bar and "," as the half bar.
        */
        void hough_preprocess(string path);

        /**
        Encodes a single digit of a POSTNET "A" bar code.

        @param digit the single digit to encode.
        @return a bar code of the digit using "|" as the long
        bar and "," as the half bar.
        */
        void ransac_preproc(string path, double canny_threshold, double circle_threshold, int numIterations);

        /**
        Encodes a single digit of a POSTNET "A" bar code.

        @param digit the single digit to encode.
        @return a bar code of the digit using "|" as the long
        bar and "," as the half bar.
        */
        void save_and_draw(string path);

        /**

        @param digit the single digit to encode.
        @return a bar code of the digit using "|" as the long
        bar and "," as the half bar.
        */
        vector<string> predict(string path);

        /**
        Support function to check if a string ends with another string
        @param string to analyze
        @param ending string
        @return true if the first string ends with the second
        */
        static bool endsWith(const std::string& str, const std::string& suffix)
        {
            return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
        }
private:

        // path to image
        string image_path;

        //image
        Mat image;

        //output image
        Mat image_out;

        // vector to store the details of coins circles
        vector<Vec3f> coin;

        //images plates
        vector<Mat> coins_img;

        //euro found strings
        vector<string> pred;

};
