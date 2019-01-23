#include <iostream>
#include <chrono>
#include <cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

#include "tictoc.hpp"

using namespace cv;
using namespace std;
using namespace cv::line_descriptor;
struct sort_descriptor_by_queryIdx
{
    inline bool operator()(const vector<DMatch>& a, const vector<DMatch>& b){
        return ( a[0].queryIdx < b[0].queryIdx );
    }
};
struct sort_lines_by_response
{
    inline bool operator()(const KeyLine& a, const KeyLine& b){
        return ( a.response > b.response );
    }
};
void ExtractLineSegment(const Mat &img, const Mat &image2, vector<KeyLine> &keylines,vector<KeyLine> &keylines2);
void ExtractORB(const Mat &img, const Mat &image2, vector<KeyPoint> &KeyPoints,vector<KeyPoint> &KeyPoints2);
int main(int argc, char**argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./Line path_to_image1 path_to_image2" << endl;
        return 1;
    }
    string imagePath1=string(argv[1]);
    string imagePath2=string(argv[2]);
    cout<<"import two images"<<endl;
    Mat image1=imread(imagePath1);
    Mat image2=imread(imagePath2);

//    imshow("ima1",image1);
//    imshow("ima2",image2);
//    waitKey(0);
    if(!image1.data || !image2.data)
    {
        cout<<"the path is wrong"<<endl;
    }

    cv::resize(image1,image1,Size(640,480));
    cv::resize(image2,image2,Size(640,480));


    vector<KeyLine> keylines,keylines2;


    TicToc time;
    ExtractLineSegment(image1,image2,keylines,keylines2);
    std::cout << "line cost time :" << time.toc() << std::endl;

    vector<KeyPoint> KeyPoints;
    vector<KeyPoint> KeyPoints2;
    time.tic();
    ExtractORB(image1, image2, KeyPoints, KeyPoints2);
    std::cout << "ORB cost time :" << time.toc() << std::endl;
    return 0;

}
void ExtractLineSegment(const Mat &img, const Mat &image2, vector<KeyLine> &keylines,vector<KeyLine> &keylines2)
{
    Mat mLdesc,mLdesc2;

    vector<vector<DMatch>> lmatches;

    Ptr<BinaryDescriptor> lbd = BinaryDescriptor::createBinaryDescriptor();
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();

//    cout<<"extract lsd line segments"<<endl;
    lsd->detect(img, keylines, 5,1);
    lsd->detect(image2,keylines2,5,1);
//    int lsdNFeatures = 1000;
//    cout<<"filter lines"<<endl;
//    if(keylines.size()>lsdNFeatures)
//    {
//        sort(keylines.begin(), keylines.end(), sort_lines_by_response());
//        keylines.resize(lsdNFeatures);
//        for( int i=0; i<lsdNFeatures; i++)
//            keylines[i].class_id = i;
//    }
//    if(keylines2.size()>lsdNFeatures)
//    {
//        sort(keylines2.begin(), keylines2.end(), sort_lines_by_response());
//        keylines2.resize(lsdNFeatures);
//        for(int i=0; i<lsdNFeatures; i++)
//            keylines2[i].class_id = i;
//    }
//    cout<<"lbd describle"<<endl;
    lbd->compute(img, keylines, mLdesc);
    lbd->compute(image2,keylines2,mLdesc2);//计算特征线段的描述子
    BFMatcher* bfm = new BFMatcher(NORM_HAMMING, false);
    bfm->knnMatch(mLdesc, mLdesc2, lmatches, 2);
    vector<DMatch> matches;

    for(auto &i:lmatches){
        double distanceRatio = i[0].distance/i[1].distance;
        if(distanceRatio < 0.5 && i[0].distance < 50)
            matches.push_back(i[0]);
    }

    cv::Mat outImg;
    std::vector<char> mask( lmatches.size(), 1 );
    drawLineMatches( img, keylines, image2, keylines2, matches, outImg, Scalar::all( -1 ), Scalar::all( -1 ), mask,
                     DrawLinesMatchesFlags::DEFAULT );

    std::cout << "line match size :" << matches.size() << std::endl;
//    imshow( "Matches", outImg );
    cv::imwrite("Matched.jpg",outImg);
}

void ExtractORB(const Mat &img, const Mat &image2, vector<KeyPoint> &KeyPoints,vector<KeyPoint> &KeyPoints2)
{
    Mat desc,desc2;

    vector<vector<DMatch>> matches;

    Ptr<FeatureDetector> orb = ORB::create(2000, 1.2, 8);
    Ptr<DescriptorExtractor> orb_des = ORB::create(2000, 1.2, 8);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");


    orb->detect(img, KeyPoints);
    orb->detect(image2, KeyPoints2);


    orb_des->compute(img, KeyPoints, desc);
    orb_des->compute(image2, KeyPoints2, desc2);

    matcher->knnMatch(desc, desc2, matches,2);

    vector<DMatch> matches_perfect;
    for(auto &m:matches){
        double p = m[0].distance/m[1].distance;
        if(p < 0.5 && m[0].distance<50){
            matches_perfect.push_back(m[0]);
        }
    }

    cv::Mat outImg;
    drawMatches(img, KeyPoints, image2, KeyPoints2, matches_perfect, outImg);

    std::cout << "orb match size :" << matches.size() << std::endl;
//    imshow( "Matches", outImg );
    cv::imwrite("Matched_ORB.jpg",outImg);
}