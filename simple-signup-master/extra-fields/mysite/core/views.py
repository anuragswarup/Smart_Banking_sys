import time
import cv2
import os
import numpy as np
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, authenticate
from django.shortcuts import render, redirect
from mysite.core.forms import SignUpForm
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/ajithkumar/Desktop/simple-signup-master/extra-fields/mysite/core/trainer/trainer.yml')
cascadePath = "/home/ajithkumar/Desktop/simple-signup-master/extra-fields/mysite/core/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
id = 0
names = ['None', 'Ajii', 'Abishek', 'Anurag', 'Abishek', 'Ajish'] 
def signup(request):
	if request.method == 'POST':
		form = SignUpForm(request.POST)
		if form.is_valid():
			form.save()
			username = form.cleaned_data.get('username')
			raw_password = form.cleaned_data.get('password1')
			user = authenticate(username=username, password=raw_password)
			login(request, user)
			return redirect('home')
	else:
		form = SignUpForm()
	return render(request, 'signup.html', {'form': form})
def detect_face(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2, minNeighbors = 5)
	if(len(faces)==0):
		return None,None
	(x, y, w, h) = faces[0]
	return gray[y:y+w, x:x+h], faces[0]   
def predict(test_img1):
	img = test_img1.copy()
	face, rect = detect_face(img)
	label, confidence = recognizer.predict(face)
	label_val= confidence
	label_text = names[label]
	return label_text,label_val
@login_required
def home(request):
	val=30
	camera = cv2.VideoCapture(0)
	for i in xrange(val):
 		return_value,temp =camera.read() 
	return_value, image = camera.read()
	cv2.imwrite("test.jpg", image)
	del(camera)
	test_img1 = cv2.imread("/home/ajithkumar/Desktop/simple-signup-master/extra-fields/test.jpg")
	predicted_img1,val = predict(test_img1)
	return render(request, 'home.html',{'name':predicted_img1})
def god(request):
	return render(request,'god.html')
