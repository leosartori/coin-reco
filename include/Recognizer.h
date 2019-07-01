// #ifndef PROJECT_SARTORI_RECOGNIZER_H
// #define PROJECT_SARTORI_RECOGNIZER_H

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

        //constructor
        Recognizer(string image_path);

        void preprocess();

        vector<string> output();


        // Support function
        static bool endsWith(const std::string& str, const std::string& suffix)
        {
            return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
        }
private:

        // path to image
        string image_path;

        //image
        Mat image;

        //images plates
        vector<Mat> coins_img;

        //plate string
        vector<string> pred;

        //image with coin predictions
        Mat finalImage;

};


// #endif //PROJECT_SARTORI_RECOGNIZER_H
