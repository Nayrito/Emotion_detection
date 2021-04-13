# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:04:48 2021

@author: nairo
"""
import cv2
import os
import numpy as np
import keyboard

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

face_cascade = cv2.CascadeClassifier('C:/Users/nairo/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml"')


while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(350,350),interpolation= cv2.INTER_CUBIC)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('rostro',frame)   
    
    if keyboard.is_pressed('p'):
         
         print('se presionó [p]arar!')
         break


cap.release()
cv2.destroyAllWindows()
