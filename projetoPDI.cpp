// Flags to compile in opencv using g++ in linux terminal
/*
	You can make an .sh to help -> to use .sh ->
	script in sh: g++ $1.cpp -o $1 `pkg-config --cflags --libs opencv` && ./$1

	http://rnd.azoft.com/instant-license-plate-recognition-in-ios-apps/
*/

// Libs to use opencv (img process and gui - User interface)
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"

// Lib with a pack of libs (io read ...)
#include <bits/stdc++.h>

// define namespace either i'm using cv:: and std::
using namespace std;
using namespace cv;

#define READ_VIDEO 1

// Number of images to be read
#define IMG_QTD 15

// Convert int to String - in C++11 => std::to_string()
std::string intToString(int intNumber)
{
	string stringToConvert;
	std::stringstream ss;
	ss << intNumber;
	ss >> stringToConvert;
	return stringToConvert;
}

// Applying Top Hat 
void applyingTopHat(cv::Mat &img_gray, cv::Mat &img_aux)
{
	int operation = MORPH_TOPHAT;
	int morph_elem = 1; // Rectangle

	Mat element = getStructuringElement( morph_elem, Size( 3, 9 ), Point( -1, -1 ) );

	dilate( img_gray, img_aux, element );
	//cv::imshow("2.1 - Dilate ",img);
	erode( img_aux, img_aux, element );
	//cv::imshow("2.2 - Erode ",img);
	subtract(img_aux,img_gray,img_aux);
	//cv::imshow("2.3 - Subtract",img);
}

// Applying Sobel Operator in dx
void applyingSobelOperator(cv::Mat &img_aux, cv::Mat &ssobelx)
{
	cv::Mat sobelx;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;

	Sobel( img_aux, sobelx, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	convertScaleAbs( sobelx, ssobelx );
	//cv::imshow("teste",ssobelx);
}

// Applying Closing Operator
void applyingClosingOperator(cv::Mat &gaussianBlurMat, cv::Mat &matCloseImage)
{
	int morph_elem = 1; // Rectangle
	Mat element = getStructuringElement( morph_elem, Size( 3, 9 ), Point( -1, -1 ) );
	morphologyEx( gaussianBlurMat, matCloseImage, 3, element );
	//cv::imshow("Closing Operator", matCloseImage);
}

Mat histeq(Mat &in)
{
    Mat out(in.size(), in.type());
    if(in.channels()==3){
        Mat hsv;
        vector<Mat> hsvSplit;
        cvtColor(in, hsv, CV_BGR2HSV);
        split(hsv, hsvSplit);
        equalizeHist(hsvSplit[2], hsvSplit[2]);
        merge(hsvSplit, hsv);
        cvtColor(hsv, out, CV_HSV2BGR);
    }else if(in.channels()==1){
        equalizeHist(in, out);
    }

    return out;
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

// In future we can use argc and argv with a pre-build maker to get input data for imgsQTD
int main( int argc, char **argv )
{

   // Mat defines to open img and process
   		cv::Mat img,img_gray,img_aux, ssobelx, gaussianBlurMat,matCloseImage;

   // READ_VIDEO == 1 - Read Video else read Database of cars with plates or plates
    
   // Loop to read imgs and process...
   for(int i = 1; i <= IMG_QTD;i++)
  {

   		std::string imgString = intToString(i);
		imgString+=".jpg";
		img = cv::imread("input/" + imgString/*"teste.jpg"*/);
		
		histeq(img);

		//////////////// 1 Step ///////////////////
		/* 1 - Convert camera image to grayscale */
		cv::cvtColor(img, img_gray, CV_BGR2GRAY);
		cv::imshow("1 - Image Grayscale",img_gray);

		//////////////// 2 Step ///////////////////
		/* 2 - Applying the morphological operation Top Hat over the resulting image */
		/* The result applying isolated (dilate+erode+subtract) was better*/
		applyingTopHat(img_gray,img_aux);
		cv::imshow("2 - Top Hat",img_aux);

		//////////////// 3 Step ///////////////////
		/* 3 - Applying Sobel operator */
		applyingSobelOperator(img_aux,ssobelx);
		imshow("3 - Sobel in X", ssobelx);  

		//////////////// 4 Step ///////////////////
		/* 4 - Applying Gaussian blur */
		GaussianBlur( ssobelx, gaussianBlurMat, Size( 5, 5 ), 0, 0 );
		imshow("4 - Applying Gaussian blur", gaussianBlurMat);  
			
		//////////////// 5 Step ///////////////////
     	/* 5 - Closing */
		//applyingClosingOperator(gaussianBlurMat,matCloseImage);
		applyingClosingOperator(gaussianBlurMat, matCloseImage);
		cv::imshow("5 - Closing",matCloseImage);
 
		
 		// 6 - Calculate Histogram
	    // Initialize parameters
	    int histSize = 256;    // bin size
	    float range[] = { 0, 255 };
	    const float *ranges[] = { range };
 
    	MatND hist;
    	calcHist( &matCloseImage, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );
     
    	// Show the calculated histogram in command window
    	double total;
    	total = matCloseImage.rows * matCloseImage.cols;
    	for( int h = 0; h < histSize; h++ )
		{
			float binVal = hist.at<float>(h);
			cout<<" "<<binVal;
		}
 	
 	   	// Plot the histogram
 	   	int hist_w = 512; int hist_h = 400;
 	   	int bin_w = cvRound( (double) hist_w/histSize );
 	
 	   	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0,0,0) );
 	   	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
 	   	 
 	   	for( int i = 1; i < histSize; i++ )
 	   	{
 	   	  line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ) ,
 	   	                   Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
 	   	                   Scalar( 255, 0, 0), 2, 8, 0  );
 	   	}
 	
    	namedWindow( "Result", 1 );    imshow( "Result", histImage );
 
 		int cont = 0;

 		//string ty =  type2str( matCloseImage.type() );
		//printf("Matrix: %s %dx%d \n", ty.c_str(), matCloseImage.cols, matCloseImage.rows );

 		// 7 - Threshold image
    	Mat image;
		for(int i = 0; i < matCloseImage.rows; i++)
		{
		   for(int j=0; j < matCloseImage.cols; j++)
		   {
		      //apply condition here
		      if((int) matCloseImage.at<uchar>(i,j) >= 70 && (int) matCloseImage.at<uchar>(i,j) <= 150)
		      {

		        matCloseImage.at<uchar>(i,j) = (int) 255 ;
		        
		        
		        if(i > 0 && matCloseImage.at<uchar>(i-1,j) != 255)
		        	matCloseImage.at<uchar>(i-1,j) = (int) 255 ;
		        if(i < matCloseImage.rows-1 && matCloseImage.at<uchar>(i+1,j) != 255)
		        	matCloseImage.at<uchar>(i+1,j) = (int) 255 ;
		        if(j > 0 && matCloseImage.at<uchar>(i,j-1) != 255)
		        	matCloseImage.at<uchar>(i,j-1) = (int) 255 ;
		        if(j < matCloseImage.rows-1 && matCloseImage.at<uchar>(i,j+1) != 255)
		        	matCloseImage.at<uchar>(i,j+1) = (int) 255 ;
		        cont++;
		      }
		      else
		      {
		      	matCloseImage.at<uchar>(i,j) = (int) 0 ;
		      }
		   }
		}

    	//M.at<double>(i,j) += 1.f;

    	cout << "Contador: " <<  cont << endl;



  		// 8 - Erode and Dilate image
		
		int morph_elem = MORPH_ELLIPSE; // Rectangle

		Mat element = getStructuringElement( MORPH_RECT, Size( 3, 3), Point( 0, 0 ) );
		Mat element3 = getStructuringElement( MORPH_CROSS, Size( 5, 5), Point( 0, 0 ) );
		Mat element2 = getStructuringElement( MORPH_ELLIPSE, Size( 9, 9 ), Point( 0, 0 ) );
		erode( matCloseImage, matCloseImage, element );
		dilate( matCloseImage, matCloseImage, element2 );
		erode( matCloseImage, matCloseImage, element3 );
		dilate( matCloseImage, matCloseImage, element2 );
		

	 
	    // Split the image into different channels
	    vector<Mat> rgbChannels(3);
	    split(img, rgbChannels);

		  // Show individual channels
	    Mat g, fin_img;
	    g = Mat::zeros(Size(img.cols, img.rows), CV_8UC1);
	     
	    // Showing Red Channel
	    // G and B channels are kept as zero matrix for visual perception
	    
	    vector<Mat> channels;
	    channels.push_back(rgbChannels[0]);
	    channels.push_back(rgbChannels[1]);
	    channels.push_back(rgbChannels[2]+matCloseImage);
	 
	    /// Merge the three channels
	    merge(channels, fin_img);
	    namedWindow("R",1);imshow("R", fin_img);
		
        


		if(cv::waitKey(0))
		{
		//	cv::destroyWindow(imgString);
		}

   }

   return 0;
}  

