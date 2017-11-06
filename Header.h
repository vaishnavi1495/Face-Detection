#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2\contrib\contrib.hpp"
#include "opencv2\opencv.hpp"
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void dbread(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) {
		string error = "no valid input file";
		CV_Error(CV_StsBadArg, error);
	}

	string line, path, label;
	while (getline(file, line))
	{
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, label);
		
		
		if (!path.empty() && !label.empty()) {
			images.push_back(imread(path));
			labels.push_back(atoi(label.c_str()));
		}
	}
}

void eigenFaceTrainer() {
	vector<Mat> images;
	vector<Mat> output;
	vector<int> labels;
	Mat face2;
	Mat img2;
	try {
		dbread("train.csv", images, labels);
		for (int i = 0; i < images.size(); i++)
		{
			cvtColor(images.at(i), img2, CV_BGR2GRAY);
			cv::resize(img2, face2, Size(400, 600));
			output.push_back(face2);
		}
		//cout << "size of the images is " << images.size() << endl;
		//cout << "size of the label is " << labels.size() << endl;
		cout << "Training begins...." << endl;
	}
	catch (cv::Exception& e) {
		cerr << " Error opening the file " << e.msg << endl;
		exit(1);
	}

	//create algorithm eigenface recognizer
	Ptr<FaceRecognizer>  model = createEigenFaceRecognizer();
	//train data
	model->train(output, labels);

	model->save("C:/Users/bhosa/Documents/Visual Studio 2015/Projects/Assign4/Assign4/eigenfacefinal33.xml");

	cout << "Training finished...." << endl;
	
	waitKey(10000);
}

//void reconstruction(Ptr<FaceRecognizer> model, Mat image)
//{
	//Mat w = model->getMat("eigenvectors");
	//////get the sample mean from the training data
	//Mat mean = model->getMat("mean");
	//Mat projection = subspaceProject(w, mean, image.reshape(1, 1));
	//Mat reconstruction = subspaceReconstruct(w, mean, projection);
	//	// Normalize the result:
	//Mat dst;
	//cv::normalize(reconstruction.reshape(1, image.rows), dst, 0, 255, NORM_MINMAX, CV_8UC1);
	
	//	// Display or save:
//	imshow("Reconstruct", dst);
//}

int  FaceRecognition() 
{

	cout << "start recognizing..." << endl;

	//load pre-trained data sets
	Ptr<FaceRecognizer>  model = createEigenFaceRecognizer();
	
	model ->load("C:/Users/bhosa/Documents/Visual Studio 2015/Projects/Assign4/Assign4/eigenfacefinal33.xml");
	cout << "xml file loaded";
	Mat testSample = imread("C:\\Users\\bhosa\\OneDrive\\Pictures\\Vash\\14.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	string classifier = "C:\\Users\\bhosa\\Documents\\Visual Studio 2015\\Projects\\Assign4\\Assign4\\haarcascade_frontalface.xml";

	CascadeClassifier face_cascade;
	string window = "Capture - face detection";

	if (!face_cascade.load(classifier)) {
		cout << " Error loading file" << endl;
		return -1;
	}

	VideoCapture cap(1);
	
	if (!cap.isOpened())
	{
		cout << "exit" << endl;
		return -1;
	}

	namedWindow(window, 1);
	long count = 0;

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;
		Mat face2;
		Mat roi;
		cap >> frame;
		count = count + 1;//count frames;
		int ct = 0;
		if (!frame.empty()) {

			
			//clone from original frame
			original = frame.clone();
			
			//convert image to gray scale and equalize
			cvtColor(original, graySacleFrame, CV_BGR2GRAY);

			//detect face in gray image
			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			//number of faces detected
			cout << faces.size() << " faces detected" << endl;
			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			//person name
			string Pname = "";
			string Pname2 = "";
			for (int i = 0; i < faces.size(); i++)
			{
				//region of interest
				Rect face_i = faces[i];

				//crop the roi from grya image
				Mat face = graySacleFrame(face_i);

				//resizing the cropped image to suit to database image sizes
				Mat face_resized;
				cv::resize(face, face_resized, Size(400,600));
				imshow("face resized", face_resized);
				
				//recognizing what faces detected
				int label = -1; 
				double confidence = 0.0;
				model->predict(face_resized, label, confidence);
				//reconstruction(model, face_resized);
				//cout << " confidencde " << confidence << endl;

				rectangle(original, face_i, CV_RGB(255, 0, 255), 1);
				string text = "Detected";

			    if (label == 1)// && (confidence > 40000 || confidence < 60000)) 
				{
					Pname = "vaishnavi";

					int x = std::max(face_i.tl().x - 30, 0);
					int y = std::max(face_i.tl().y - 30, 0);

					//name the person who is in the image
					putText(original, Pname, Point(x, y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
		

					for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++)
					{
						roi = original(*r);
						medianBlur(roi, roi, 45);
					}

				}
				else if (label != 1)
			    {
					Pname = "unknown";
				}

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				//name the person who is in the image
				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			}

			putText(original, "Frames: " + frameset, Point(30, 60), CV_FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			//display to the winodw
			cv::imshow(window, original);

		}
		if (waitKey(30) >= 0) break;
	}
}