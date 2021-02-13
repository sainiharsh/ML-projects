import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        roismile_gray = gray[y:y+h, x:x+w]
        roismile_color = img[y:y+h, x:x+w]
        
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
            
    
    
    
    cv2.imshow("video Frame",img)
    cv2.imshow("Gray frame",gray)
    # Wait for user input - q then you will stop the loop
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    