# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os 

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
id = 0
names = ['None', 'Ajithkumar', 'Ajiii', 'Atharsh', 'Abishek', 'Ajish'] 


def detect_face(img):
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
     faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        
       )
     if(len(faces)==0):
         return None,None
     
     (x, y, w, h) = faces[0]
     return gray[y:y+w, x:x+h], faces[0]
    
    
def predict(test_img):
    
    img = test_img.copy()   
    face, rect = detect_face(img)  
    label, confidence = recognizer.predict(face)  
    label_val= confidence
    label_text = names[label]    
    return label_text,label_val




test_img1 = cv2.imread("test-data/test1.jpeg")
predicted_img1,val = predict(test_img1)
print(predicted_img1)
print(val)





