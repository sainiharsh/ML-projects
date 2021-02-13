# Face Recognition And Detection

Face Recognition using OpenCV in Python

### Prerequisites

Numpy</br>
OpenCV

#### Note: Please install opencv-contrib-python package instead of opencv-contrib as it contains the main modules and also contrib modules.

### Installing

Install Numpy via Python :
 `pip install Numpy` </br>
 Install openCv via Python :
  `pip install opencv-python`
  
 ### Import Modules
 `import cv2` </br>
 `import numpy as np`
 
 ### openCV used code
 `cv2.VideoCapture(id)` : For default webcam `id=0` but for multiple webcams specified ids are provided.</br>
 `cv2.cascadeClassifier('File Required')`: Object file </br>
 `cv2.VideoCapture.read()`  : Return boolean and capture frame.</br>
  

**Closing video stream** </br>

 `cv2.VideoCapture.release()` </br>
 `cv2.destroyAllWindows()`
 
 ### Files Required
* [haarcascade_frontalface_alt.xml](https://github.com/sainiharsh/Machine-Learning-Projects/blob/master/Face%20Recognition/haarcascade_frontalface_alt.xml) : **Face Detection**
* [haarcascade_eye.xml](https://github.com/sainiharsh/Machine-Learning-Projects/blob/master/Face%20Recognition/haarcascade_eye.xml) : **Eye Detection** 
* [haarcascade_smile.xml](https://github.com/sainiharsh/Machine-Learning-Projects/blob/master/Face%20Recognition/haarcascade_smile.xml) : **Smile Detection**

 
 ### [Face_Data_Collect](https://github.com/sainiharsh/Machine-Learning-Projects/blob/master/Face%20Recognition/face_data.py)
 1. Read a video stream from webcam or camera , capture images.
 2. Detect faces and show bounding box.
 3. Flatten the largest face image and save in an numpy array.
 4. Repeat the above for multiple people to generate training data.
 
 ### [Face Recognize](https://github.com/sainiharsh/Machine-Learning-Projects/blob/master/Face%20Recognition/face_Recognition.py)
 1. Load the training data (Numpy array of all the person)
 2. Read a video stream using openCV.
 3. Extract faces out of it.
 4. Use of KNN algorithm to find the prediction of face.
 5. Map the predicted id to name of the user.
 6. Display the predictions on the screen-bounding box and name.
 
 
