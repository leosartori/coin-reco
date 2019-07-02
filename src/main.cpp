/**
    Coin Detector and Recognizer - Exam of Computer Vision 2018/2019 UNIPD

    @author Leonardo Sartori (leonardo.sartori.1@studenti.unipp.it)
    @version 1.0
*/

#include <memory>
#include <iostream>
#include <stdio.h>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

#include </usr/include/jsoncpp/json/value.h>
#include <fstream>
#include <Recognizer.h>

using namespace cv;
// Namespace where all the C++ OpenCV functionality resides.

using namespace std;
// For input output operations.

int main() {
    //Choose the image
    string image;
    cout << "Enter image name: Choose one \n"
            "[1] [2] [3] [4] [5] [6] [7] [8] [9]\n";
    cin >> image;

    //Build the coins recognizer
    Recognizer rec = Recognizer("images/samples/" + image + ".jpg");

    //Perform preprocessing: extract coins as circles and save images
    int method;
    cout << "Preprocessing type: Choose one \n"
            "[1] = HoughCircle; [2] = RANSAC (still in beta version...)\n";
    cin >> method;
    rec.preprocess(method, "images/detect/");

    //Get predictions on extracted subimages
    vector<string> pred = rec.predict("images/detect/");

    cout << "------------ DETECTOR RESUME ---------------" << endl;
    //Use map to catalogue type of coins found
    std::map<std::string, int> counter;
    for (auto p : pred) {
        if (Recognizer::endsWith(p,"\n"))
            p = p.substr(0,p.size()-2);
        counter[p]++;
    }

    float coins_sum = 0;
    float value;

    //Print map values
    for (map<std::string, int>::const_iterator it = counter.begin(); it != counter.end(); it++){
        cout << it->first << ": " << it->second << endl;
        if (it->first == "2e")
            value = 2 * it->second;
        if (it->first == "1e")
            value = it->second;
        if (it->first == "50c")
            value = 0.5 * it->second;
        if (it->first == "20c")
            value = 0.2 * it->second;
        if (it->first == "10c")
            value = 0.1 * it->second;
        if (it->first == "5c")
            value = 0.05 * it->second;
        if (it->first == "2c")
            value = 0.02 * it->second;
        if (it->first == "1c")
            value = 0.01 * it->second;

        // sum total of found euros
        coins_sum += value;
    }

    // Print total of euro found
    cout << "Total euro: " << coins_sum << endl;
    return 0;    // Return from main function.
}