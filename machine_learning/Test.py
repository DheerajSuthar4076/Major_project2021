import numpy as np
import cv2
import dlib

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
cap = cv2.VideoCapture(0)


while(True):
    #capture Frame-by-Frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors=5)
    for(x,y,w,h)in faces:
        print("face",x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)
        color = (255, 0, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)

    for(x,y,w,h)in eyes:
        print("eyes",x,y,w,h)
        roi_gray1 = gray[y:y+h, x:x+w]
        img_item1 = "my-image1.png"
        cv2.imwrite(img_item1, roi_gray1)
        color = (0, 255, 0) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)


    #display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# when everything done, release the VideoCapture
cap.release()
cv2.destroyAllWindows()
