#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:27:06 2019

@author: indra
"""

#opencv is called cv2 
import cv2

#loading the cascades
eyecascade=cv2.CascadeClassifier('haarcascade_eye.xml')
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#defining the detection function; gray: b/w image(detection happens only on this one and then the coordinates are used to draw the rectangles on the color image as well) ;frame :color image
def face_detect(gray,frame):
    
    #get the coordinates of the rectangle that has the face
     face_rect_coord=facecascade.detectMultiScale(gray,1.3,5)
    
    #iterate through the coordinates to draw the rectangle x,y=upperleft corner w=width ,h =height
     for (x,y,w,h) in face_rect_coord:
    
         #cv2.rectangle(original image,upperleft corner coord, lowerright corner coord,color of the rectangle , thickness of the rectangle)
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
         
         #detect the eyes inside the detected face-rectangle the roi are the face rectangle region
         roi_gray=gray[y:y+h,x:x+w]
         roi_color=frame[y:y+h,x:x+w]
         eye_rect_coord=eyecascade.detectMultiScale(roi_gray,1.1,3)
         
         #iterate through the coordinates of eye recatangles
         for (ex,ey,ew,eh) in eye_rect_coord:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
     return frame
 
#create a video stream from the webcame to get the images where detection takes place for every frame
# 1= external webcam ;    0= internal webcam 
video_capture=cv2.VideoCapture(0)

#continously access the video from webcam until q is pressed to exit
while True:
    
#to objects are returned by the following function where the second object is 
#the color photo or video frame
    _,frame=video_capture.read()


#converting the color frame to a gray frame using inbuilt cv2 functions
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #passing to the above created detect function
    canvas=face_detect(gray,frame)
    cv2.imshow('Video',canvas)

    #if q is pressed , process stops
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
