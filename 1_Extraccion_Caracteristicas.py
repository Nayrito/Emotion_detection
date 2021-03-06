# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 11:10:02 2021

@author: nairo
"""
import cv2 
import numpy as np
import os 
import time
#import imutils 

path = "C:/Users/nairo/OneDrive/Documentos/semestre actual/Emotion_detection/Dataset_inteligencia" # actualizar carpeta para cada emocion
lista = os.listdir(path) #convierte el contenido del pat en una lista 


labels = []
data = []
label = 0

for directorio in lista:
    emocionpath = path +'/'+ directorio # accedemos a la carpeta de cada emocion
    for imagen in os.listdir(emocionpath):
        data.append(cv2.imread(emocionpath +'/'+imagen,0)) # agregar imagenes
        labels.append(label) # agregar etiquetas 
    
    label= label+1


#%% local binary pattern LBPH##
labels =np.asarray(labels)
lbph_recognizer = cv2.face.LBPHFaceRecognizer_create() # creamos el objeto LBPH
lbph_recognizer.train(data,labels) # entrenamos el algoritmo con las imagenes del dataset
hist = lbph_recognizer.getHistograms() # histogramas (características)

hist = np.asarray(hist)
hist = hist[:,0,:]  # se ajustan las dimensiones de las caracteristicas 

#%%
## EigenFaces ##
labels =np.asarray(labels)
EigenFaces_recognizer = cv2.face.EigenFaceRecognizer_create() # creamos el objeto LBPH
EigenFaces_recognizer.train(data,labels) # entrenamos el algoritmo con las imagenes del dataset
#vect = EigenFaces_recognizer.getEigenVectors() # histogramas (características)
vect = EigenFaces_recognizer.getEigenVectors()

vect = np.asarray(vect)
vect = vect.reshape(331,122500)
np.savetxt('caracteristicas_EF.csv',vect,delimiter=',',fmt="%f")

#%%
np.savetxt('labels.csv',labels,delimiter=' ',fmt='%s')
np.savetxt('caracteristicas.csv',hist,delimiter=',',fmt="%f")




