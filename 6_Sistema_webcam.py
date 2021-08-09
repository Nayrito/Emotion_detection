# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 19:04:48 2021

@author: nairo
"""
import cv2
import os
import numpy as np
import keyboard
import time

#%% ALMACENAMOS BASE DE DATOS CON NUESTRO ROSTRO

# emocion = 'feliz'
emocion = 'triste'
# emocion = 'enojo'
# emocion = 'asco'
# emocion = 'sorpresa'
# emocion = 'neutral'
path = "D:/Documentos/semestre actual/Emotion_detection/Dataset_propio" 
emocion_path = path + '/' + emocion
if not os.path.exists(emocion_path):
    os.makedirs(emocion_path)

face_cascade = cv2.CascadeClassifier('C:/Users/nairo/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') #path Nayro 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
k = 0 # contador para el número de imágenes a almacenar

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
        cv2.imwrite(emocion_path + '/rotro_{}.jpg'.format(k),rostro)
        k = k + 1
    
    cv2.imshow('rostro',frame)
    
    if k ==100:
        break

cap.release()
cv2.destroyAllWindows()




#%% ENTRENAMOS EL ALGORITMO LBPH 
    
path = "D:/Documentos/semestre actual/Emotion_detection/Dataset_propio" # actualizar carpeta para cada emocion
lista = os.listdir(path)
labels = []
data = []
label = 0
for directorio in lista:
    emocionpath = path + '/' + directorio
    for fileName in os.listdir(emocionpath):
        labels.append(label)
        data.append(cv2.imread(emocionpath+'/'+fileName,0))
    label = label + 1

emotion_recognizer = cv2.face.LBPHFaceRecognizer_create()
# Entrenando el reconocedor de rostros
print("Entrenando LBPH ...")
inicio = time.time()
emotion_recognizer.train(data, np.array(labels))
tiempoEntrenamiento = time.time()-inicio
print("Tiempo de entrenamiento : ", tiempoEntrenamiento)
# Almacenando el modelo obtenido
emotion_recognizer.write("modeloLBPH.xml")


#%% RECONCER EMOSIONES 

emotion_recognizer.read('modeloLBPH.xml')
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) # inicio de captura 
face_cascade = cv2.CascadeClassifier('C:/Users/nairo/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml') #path Nayro 

while True:
    ret,frame = cap.read()
    if ret == False: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(350,350),interpolation= cv2.INTER_CUBIC)
        result = emotion_recognizer.predict(rostro)
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        if result[1] < 60:
            cv2.putText(frame,'{}'.format(lista[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
               
        else:
            cv2.putText(frame,'No identificado',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
        
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    
    cv2.imshow('rostro',frame)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
