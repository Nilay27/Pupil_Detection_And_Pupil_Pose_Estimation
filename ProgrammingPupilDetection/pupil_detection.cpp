#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// Load image
	Mat src = imread("TestData/Real/01.jpg");
	if (src.empty())
		return -1;

	// Invert the source image and convert to grayscale
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);
	GaussianBlur( gray, gray, Size( 5, 5 ), 0, 0 );


	// Convert to binary image by thresholding it
	Mat binary;
	threshold(gray, binary, 50, 255, CV_THRESH_BINARY_INV  );
	//erosion is applied to remove the unneccesay smaller contours that and also to ensure that there is not connected contour with the pupil
	erode(binary, binary, Mat(), Point(-1, -1), 5, 1, 1);
	//dilation is applied to regain the approximate shape of the target contour
	dilate(binary, binary, Mat(), Point(-1, -1), 5, 1, 1);

	///imshow("binary",binary);
	// Find all contours

	vector<vector<Point> > contours;
	findContours(binary, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	// Fill holes in each contour
	drawContours(binary, contours, -1, CV_RGB(255,255,0), -1);
	
	vector<vector<Point> > approx;
	approx.resize(contours.size());
	
	vector<RotatedRect> minEllipse( contours.size() );
	Mat drawing(src.rows,src.cols,CV_8UC1,Scalar(0));
	
	for( int i = 0; i < contours.size(); i++ )
     { 
       	if( contours[i].size() > 100 )
         { 	
         	minEllipse[i] = fitEllipse( Mat(contours[i]) ); 
         }
     }
     
     vector<Moments> mu(contours.size() );

     vector<Moments> mc(contours.size() );
     for( int i = 0; i< contours.size(); i++ )
     {
     	Mat drawing1(src.rows,src.cols,CV_8UC1,Scalar(0));
     	 
     	if(contours[i].size()>100)
     	{
	       ellipse( drawing1, minEllipse[i],  Scalar( 255, 0, 0 ), -1 );
	      
	      	mu[i] = moments( drawing1, false );
	      	float x,y;
	      	x=mu[i].m10/mu[i].m00;
	      	y=mu[i].m01/mu[i].m00;
	      	
	      	if(x>150 and x<550 and y>50 and y<400)
	      	{
	      		ellipse( drawing, minEllipse[i],  Scalar( 255, 0, 0 ), -1 );
	      
	      	}
        
	       //cout<<mc[i]<<endl;
   		}
       //cout<<minEllipse[i]<<endl;
       // rotated rectangle
       
     }
     imshow( "Contours", drawing );

	///imshow("image", binary);
	imshow("original",src);
	imwrite("contour.png",drawing);
	waitKey(0);

	return 0;
}