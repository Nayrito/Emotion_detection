# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:34:49 2021

@author: nairo
"""
#%% 1. importar librerias y cargar datos

import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit # validador cruzado de permutación aleatoria 
from sklearn.preprocessing import StandardScaler # normalizar datos
from sklearn.svm import SVC # clasificador SVM
from sklearn.metrics import accuracy_score # rendimiento para evaluar los splits 
from sklearn.metrics import confusion_matrix



data = pd.read_csv("caracteristicas-lda.csv").to_numpy() # cargamos caracteristicas lda
labels = pd.read_csv("labels.csv").to_numpy() # cargamos etiquetas 
labels = labels.reshape(329)

#%% 2. Generación splits (solo una vez)

test_percent=30 
ss=ShuffleSplit(n_splits=5, test_size=test_percent/100.0)
split = 1
for train_index,test_index in ss.split(data): # train index indica los indices de entrenamiento, test index los de prueba
    X_train=data[train_index,:] # Obtener los datos de entrenamiento
    y_train=labels[train_index]
    zParam=StandardScaler().fit(X_train) # Obtener los parametros de estandarizacion
    X_train=zParam.transform(X_train) # Estandarizar los datos de entrenamiento
    X_test=zParam.transform(data[test_index,:]) # Obtener y estandarizar los datos de prueba
    y_true=labels[test_index]
    if split==1:
        np.savetxt('train1.txt',X_train,fmt='%5.15f')
        np.savetxt('labelsTrain1.txt',y_train,fmt='%5.15f')
        np.savetxt('test1.txt',X_test,fmt='%5.15f')
        np.savetxt('labelsTest1.txt',y_true,fmt='%5.15f')
        
    if split==2:
        np.savetxt('train2.txt',X_train,fmt='%5.15f')
        np.savetxt('labelsTrain2.txt',y_train,fmt='%5.15f')
        np.savetxt('test2.txt',X_test,fmt='%5.15f')
        np.savetxt('labelsTest2.txt',y_true,fmt='%5.15f')

    if split==3:
        np.savetxt('train3.txt',X_train,fmt='%5.15f')
        np.savetxt('labelsTrain3.txt',y_train,fmt='%5.15f')
        np.savetxt('test3.txt',X_test,fmt='%5.15f')
        np.savetxt('labelsTest3.txt',y_true,fmt='%5.15f')

    if split==4:
        np.savetxt('train4.txt',X_train,fmt='%5.15f')
        np.savetxt('labelsTrain4.txt',y_train,fmt='%5.15f')
        np.savetxt('test4.txt',X_test,fmt='%5.15f')
        np.savetxt('labelsTest4.txt',y_true,fmt='%5.15f')
        
    if split==5:
        np.savetxt('train5.txt',X_train,fmt='%5.15f')
        np.savetxt('labelsTrain5.txt',y_train,fmt='%5.15f')
        np.savetxt('test5.txt',X_test,fmt='%5.15f')
        np.savetxt('labelsTest5.txt',y_true,fmt='%5.15f')
        
    split = split+1
    
#%% 3. Cargamos splits (siempre)

train1 = np.loadtxt('train1.txt')
train2 = np.loadtxt('train2.txt')
train3 = np.loadtxt('train3.txt')
train4 = np.loadtxt('train4.txt')
train5 = np.loadtxt('train5.txt')
test1 = np.loadtxt('test1.txt')
test2 = np.loadtxt('test2.txt')
test3 = np.loadtxt('test3.txt')
test4 = np.loadtxt('test4.txt')
test5 = np.loadtxt('test5.txt')
labelsTrain1 = np.loadtxt('labelsTrain1.txt')
labelsTrain2 = np.loadtxt('labelsTrain2.txt')
labelsTrain3 = np.loadtxt('labelsTrain3.txt')
labelsTrain4 = np.loadtxt('labelsTrain4.txt')
labelsTrain5 = np.loadtxt('labelsTrain5.txt')
labelsTest1 = np.loadtxt('labelsTest1.txt')
labelsTest2 = np.loadtxt('labelsTest2.txt')
labelsTest3 = np.loadtxt('labelsTest3.txt')
labelsTest4 = np.loadtxt('labelsTest4.txt')
labelsTest5 = np.loadtxt('labelsTest5.txt')

#%% 4. sistema de clasificacion y validación 

gama=[0.01,0.05,0.1,0.5,1]
kernel='rbf' # 'linear'
C=1
svm =SVC(kernel=kernel,C=C,gamma=1)  # definimos clasificador  // mejor gama 0.5
svm.fit(train3,labelsTrain3)  # entrenar svm 

labels_pred = svm.predict(test3)
accuracy = accuracy_score(labelsTest3,labels_pred) # evaluamos rendimiento en cada split 
print (accuracy)













