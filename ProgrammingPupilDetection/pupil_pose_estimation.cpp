#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <bits/stdc++.h>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	// Load image
	Mat src = imread("TestData/Synthetic/03.png");
	ifstream infile_matrix("TestData/Synthetic/CameraMatrix.txt");
     ifstream infile_sphere("TestData/Synthetic/03_3DSphereParam.txt");
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
	Mat drawing(src.rows,src.cols,CV_8UC3,Scalar(255,255,255));
	
	for( int i = 0; i < contours.size(); i++ )
     { 
       	if( contours[i].size() > 100 )
         { 	
         	minEllipse[i] = fitEllipse( Mat(contours[i]) ); 
         }
     }
     
     vector<Moments> mu(contours.size() );

     vector<Moments> mc(contours.size() );
    
     vector< vector<float> > camera_matrix(3, vector <float> (3,0));
     vector< float> sphere(3);
     
     float x1,y1,z1;
     int line=0;
     // reading values from CameraMatrix.txt
     while(infile_matrix >> x1 >> y1 >> z1)
     {
     	camera_matrix[line][0]=x1;
     	camera_matrix[line][1]=y1;
     	camera_matrix[line][2]=z1;
    
     	line++;
     }

     float x2,y2,z2,r;
     float radius;
//reading values of center of circle into another vector named sphere
     while(infile_sphere )
     {

     	if(infile_sphere >> x2 >> y2 >> z2 >> r)
     	{
	     	sphere[0]=x2;
	     	sphere[1]=y2;
	     	sphere[2]=z2;
	     	
	     	radius=r;
	     	//cout<<sphere[0]<<" "<<sphere[1]<<" "<<sphere[2]<<" "<<radius<<endl;
     	}

     }
     
     
  /***************** Drawing the 2-D projection of the sphere as a Circle onto the image plane***************/   
     float fx=0,fy=0,z=0;
    //matrix multiplication of intrinsic matrix with sphere vector to calculate scaled values of center of sphere i.e f*x, f*y and z
    	for(int j=0;j<3;j++)
        {
     		fx= fx+ camera_matrix[0][j]*sphere[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		fy= fy+ camera_matrix[1][j]*sphere[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		z= z+ camera_matrix[2][j]*sphere[j];
     	}
     //	cout<<fx<<" "<<fy<<" "<<z<<endl;
// calculating 	pixel values of center of projection of 3-D sphere

 	float u,v;
 	u=fx/z;
 	v=fy/z;

 // to get one more point on the circle, we are calculating the pixel value of one more point on the projected circle which has coordinates x+radius,y,z in camera frame  
  	sphere[0]=sphere[0]+radius;
 	float fx1=0,fy1=0,z3=0;
 	for(int j=0;j<3;j++)
        {
     		fx1= fx1+ camera_matrix[0][j]*sphere[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		fy1= fy1+ camera_matrix[1][j]*sphere[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		z3= z3+ camera_matrix[2][j]*sphere[j];
     	}

 /*********** finding the radius of the projection of the 3-d sphere ************/
    float u1,v1,radius1;
 	u1=fx1/z3;
 	v1=fy1/z3;
 	radius1=u1-u;

 /*********** finding the center of the ellipse and the points containing major axis of the ellipse**********/
 	float center_x=0,center_y=0;
 	Point2f pts[4];
 	Point2f pts_ellipse1;
 	Point2f pts_ellipse2;
 	float dist;
 	Point2f ellipse_point1, ellipse_point2,ellipse_point3,ellipse_point4 ;
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
	      		ellipse( drawing, minEllipse[i],  Scalar( 0,0,255 ), 1.5 );
	      		//drawing the projected circle
	      		circle(drawing,Point(u,v),radius1,Scalar(255,0,0),1.5);
	      		// center of the ellipse
	      		center_x=minEllipse[i].center.x;
	      		center_y=minEllipse[i].center.y;
	      		minEllipse[i].points(pts);
	
	      		// end points of the major and minor axis 
	      		ellipse_point1=(pts[0]+pts[1])/2;
	      		ellipse_point2=(pts[1]+pts[2])/2;
	      		ellipse_point3=(pts[2]+pts[3])/2;
	      		ellipse_point4=(pts[3]+pts[0])/2;
	      		// finding out the major axis and its corresponding points
	      		if(sqrt(pow((ellipse_point1.x-ellipse_point3.x),2)+pow((ellipse_point1.y-ellipse_point3.y),2))>sqrt(pow((ellipse_point2.x-ellipse_point4.x),2)+pow((ellipse_point2.y-ellipse_point4.y),2)))
	      		{
					pts_ellipse1=ellipse_point1;
					pts_ellipse2=ellipse_point3;
				}
				else
				{
					pts_ellipse1=ellipse_point2;
					pts_ellipse2=ellipse_point4;
				}

	   //   		cout<<pts_ellipse1<<" "<<pts_ellipse2<<endl;
	      		
	      
	      	}
   		}
    }
   
      float determinant = 0;

   /****************** finding inverse of the camera matrix ***********************/

    //finding determinant of the camera matrix
    for(int i = 0; i < 3; i++)
        determinant = determinant + (camera_matrix[0][i] * (camera_matrix[1][(i+1)%3] * camera_matrix[2][(i+2)%3] - camera_matrix[1][(i+2)%3] * camera_matrix[2][(i+1)%3]));
    
  
    // inverse of the matrix
    vector< vector<float> > inverse_camera_matrix(3, vector <float> (3,0));

    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
           inverse_camera_matrix[i][j]=((camera_matrix[(j+1)%3][(i+1)%3] * camera_matrix[(j+2)%3][(i+2)%3]) - (camera_matrix[(j+1)%3][(i+2)%3] * camera_matrix[(j+2)%3][(i+1)%3]))/ determinant;
        }
        
    }
 	
    /*************** FINDING THE 3D COORDINATES OF CENTER OF THE ELLIPSE IN THE PLAIN CONTAINING THE CENTER OF SPHERE************/ 
     float a=0,b=0,c=0;
     vector <float> reverse(3);
     reverse[0]=center_x*z;
     reverse[1]=center_y*z;
     reverse[2]=z;
     for(int j=0;j<3;j++)
        {
     		a= a+ inverse_camera_matrix[0][j]*reverse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		b= b+ inverse_camera_matrix[1][j]*reverse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		c= c+ inverse_camera_matrix[2][j]*reverse[j];
     	}

    // 	cout<<a<<" "<<b<<" "<<c<<endl;


/************** FINDING OUT RADIUS OF 3-D Circle *********/
//finding 3-D coordinate of first point on the diameter of circle 

	vector <float> inverse_ellipse(3);
     inverse_ellipse[0]=pts_ellipse1.x*z;
     inverse_ellipse[1]=pts_ellipse1.y*z;
     inverse_ellipse[2]=z;
     float a1=0,b1=0,c1=0;
     for(int j=0;j<3;j++)
        {
     		a1= a1+ inverse_camera_matrix[0][j]*inverse_ellipse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		b1= b1+ inverse_camera_matrix[1][j]*inverse_ellipse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		c1= c1+ inverse_camera_matrix[2][j]*inverse_ellipse[j];
     	}
     
//finding 3-D coordinate of second point on the diameter of circle
     inverse_ellipse[0]=pts_ellipse2.x*z;
     inverse_ellipse[1]=pts_ellipse2.y*z;
     inverse_ellipse[2]=z;
     float a2=0,b2=0,c2=0;
     for(int j=0;j<3;j++)
        {
     		a2= a2+ inverse_camera_matrix[0][j]*inverse_ellipse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		b2= b2+ inverse_camera_matrix[1][j]*inverse_ellipse[j];
     	}
     	for(int j=0;j<3;j++)
        {
     		c2= c2+ inverse_camera_matrix[2][j]*inverse_ellipse[j];
     	}
     	

 // radius of the circle
     	float circle_radius=0;
     	sphere[0]=sphere[0]-radius;
     //	cout<<radius<<endl;	

    // the lengths of the major axis of ellipse in the hemispherical
    // plane of sphere containing the center of the ellipse will always be equal to the diameter of the 3-D circle
     	circle_radius=(sqrt(pow((a2-a1),2)+pow((b2-b1),2)))/2;
     	
     	cout<<"Radius of the 3-D Circle (r): "<<circle_radius<<endl;

// using pythagorous theorem to find the actual center of the 3-D circle
     
// as we know the x,y coordinate of the 3-D circle ,radius of the ellipse, radius of the sphere and center of the sphere 
// therefore using Pythagorous theorem and Eucledian distance, we can find out original z-coordinate of the 3-D circle 
     	float z_coordinate_ellipse=0;
     	z_coordinate_ellipse=sqrt(pow(radius,2)-pow(circle_radius,2)-pow((a-sphere[0]),2)-pow((b-sphere[1]),2));

     	if(sphere[2]>0)
     		z_coordinate_ellipse=z_coordinate_ellipse+sphere[2];
     	else
     		z_coordinate_ellipse=z_coordinate_ellipse-sphere[2];

     	//cout<<z_coordinate_ellipse<<endl;
cout<<"Center of the 3-D circle (c): ("<<a<<" , "<<b<<" , "<<z_coordinate_ellipse<<" ) "<<endl;
// Finding the normal of the circle using Center of 3-D Sphere and Center of 3-D circle

	float normal_x,normal_y,normal_z;

	float denominator=0;

	denominator=sqrt(pow((a-sphere[0]),2)+pow((b-sphere[1]),2)+pow((z_coordinate_ellipse-sphere[2]),2));

	normal_x=(a-sphere[0])/denominator;
	normal_y=(b-sphere[1])/denominator;
	normal_z=(z_coordinate_ellipse-sphere[2])/denominator;
	cout<< "Normal of the 3-D circle (n): ("<<normal_x<<" i, "<<normal_y<<" j, "<<normal_z<<" k) "<<endl;
     imshow( "projected image", drawing );

	///imshow("image", binary);
	imshow("original",src);
	waitKey(0);

	return 0;
}