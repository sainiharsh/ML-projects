# read a video stream from Camera(frame by frame)
import cv2

cap = cv2.VideoCapture(0)   # id = 0 for default webcam, if u use multiple webcam give ther ids  

while True:
    ret,frame = cap.read()   # return two values boolen one and frame which is capture
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    if ret == False:
        continue

    # otherwise     
    cv2.imshow("Video Frame",frame)
    cv2.imshow("Gray Frame",gray_frame)

    # wait for user input -q you will stop the loop
    # cv2.waitKey(1) = 32 bits & 0xFF is 11111111(8 bits) ==> 8 bits which is comapred with ascii values
    key_pressed = cv2.waitKey(1) & 0xFF 
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    