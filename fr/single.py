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

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
	
	
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


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
    label_text = names[label]
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img




test_img1 = cv2.imread("test-data/test2.jpeg")
predicted_img1 = predict(test_img1)

cv2.imshow(names[1], cv2.resize(predicted_img1, (400, 500)))
cv2.waitKey(10000)
cv2.destroyAllWindows()






