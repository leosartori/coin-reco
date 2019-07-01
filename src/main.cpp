#include <memory>
#include <iostream>
#include <stdio.h>
// The header files for performing input and output.

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// highgui - an interface to video and image capturing.
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
// imgproc - An image processing module that for linear and non-linear  image filtering, geometrical image transformations, color space conversion and so on.
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

#include </usr/include/jsoncpp/json/value.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <Recognizer.h>

using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;
// For input output operations.

int main()
{
    //Choose the image
    string image;
    cout << "Enter image name: Choose one \n"
            "[1]""[2]""[3]""[4]""[5]""[6]""[7]""[8]""[9]\n";
    cin >> image;

    //Build the coins recognizer
    Recognizer rec = Recognizer("images/samples/" + image + ".jpg");

    //Get as output the coins found
    vector<string> pred = rec.output();

    return 0;    // Return from main function.
}