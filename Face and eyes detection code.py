import numpy as np
import cv2

face_path = r'C:\Users\komsi\Desktop\Smart Methods\Third Path\Face and eyes detection Program\haarcascade_frontalface_default.xml' #location of haarcascade files
face_cascade = cv2.CascadeClassifier(face_path)
eye_path = r'C:\Users\komsi\Desktop\Smart Methods\Third Path\Face and eyes detection Program\haarcascade_eye.xml'
eye_cascade = cv2.CascadeClassifier(eye_path)
img_path = r'C:\Users\komsi\Desktop\Smart Methods\Third Path\Face and eyes detection Program\test.jpg' #location of the pic that we want to detect the face and eyes on it
img = cv2.imread(img_path) #reading the image


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # converting the image to the gray scale (black and white)
faces = face_cascade.detectMultiScale(gray, 1.05, 5) # searching for the faces in the gray scale image
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) # drawing rectangle on the face
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray) # detect eyes in the detected face , otherwise it might detect eyes without any faces (this is why it's inside the for loop of faces)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # drawing rectangle on detected eyes

cv2.imshow('Face and eyes detection Program' , img) 


cv2.waitKey(0)
cv2.destroyAllWindows()
