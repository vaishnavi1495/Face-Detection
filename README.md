# Face-Detection

## Idea
This is a video analytics assignment to detect the face of a person.  

## Overview
1)	#include "opencv2\contrib\contrib.hpp” header is included to use facerecognizer
2)	Training is required for face recognition. 150 images are used for training. In text file the path and labels are stored.
3)	Images are converted into gray scale and resized then stored in vector.
4)	Eigenface Recognizer model is trained and xml file is stored. Now the training is done
5)	In recognition process, xml trained data file is loaded. 
6)	VideoCapture is used to catch live video though default camera.  
7)	Image frames are cloned and converted to gray scale. detectMultiScale() used to detect faces. 
8)	Region of interest is got and resized. 
9)	Recognizing what images are detected using predict() method which returns label.
10)	Label is compared to recognize own face. If it’s there then blur that face by median blur. 
11)	Confidence is printed and live video still goes on till Esc key pressed.


## Output
![Face Detected](/facedetect.jpg)


