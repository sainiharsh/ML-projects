import cv2
import numpy as np
import os

############### KNN Code #################
def distance(x1,x2):   # Eucledian 
    return np.sqrt(sum(x1-x2)**2)

def knn(train,test,k=5):
    dist = []

    for i in range(train.shape[0]):
        # Get the vector and label  
        ix = train[i,:-1]   #stored in numpy array
        iy = train[i, -1]   #assign value for each person
        # compute the distance from test point 
        d = distance(test,ix)
        dist.append([d,iy])
    # Sort based on distance and get top k    
    dk = sorted(dist,key=lambda x:x[0])[:k]
    # Retrieve only labels
    labels = np.array(dk)[:,-1]

    # Get Frequencies of each labels
    output = np.unique(labels,return_counts = True)
    # Find max frequency and Corresponding labels
    index = np.argmax(output[1])
    return output[0][index]

##############################################################

# Init Camera 
cap = cv2.VideoCapture(0)

# Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

skip = 0
dataset_path = './face_data/'
face_data = []
labels = []

class_id = 0   # Labels for the given file
names = {}     # Mapping btw id - name

# Data Preparation
for fx in os.listdir(dataset_path):  #this function is used to check file in directory 
    if fx.endswith('.npy'):
        # Create a mapping btw class_id and names
        names[class_id] = fx[:-4]
        print("Loaded "+fx)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        #Create Labels for the class
        target = class_id*np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target) 

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

# Testing
while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
 

    for face in faces:
        x,y,w,h = face

        # Get the face ROI
        offset = 10
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        
        # Predicted Label (out)
        out = knn(trainset,face_section.flatten())


        # Display on th screen the name and rectangle around it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,251,0),2,cv2.LINE_AA)
        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roismile_gray = gray_frame[y:y+h, x:x+w]
        roismile_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            center = (ex+ew//2,ey+eh//2)
            radius = (ew+eh)//4
            cv2.circle(roi_color,center,radius,(0,255,0),2)
            cv2.circle(roi_gray,center,radius,(0,255,0),2)

        smiles = smile_cascade.detectMultiScale(roismile_gray,1.8,20)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roismile_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)  
            cv2.rectangle(roismile_gray,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)  
            



    cv2.imshow("Faces",frame)
    cv2.imshow("Gray Frame",gray_frame)
    # cv2.imshow("face section",face_section)

    key = cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    






