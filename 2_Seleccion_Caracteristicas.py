# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt



data = pd.read_csv("caracteristicas.csv").to_numpy()
labels = pd.read_csv("labels.csv").to_numpy()
labels = labels.reshape(330)

data_norm = (data - np.mean(data))/np.std(data) #normalizamos los datos 
lw=2
target_names = ['asco','enfado','feliz', 'neutral','sorpresa','triste']
colors =['orange','green','blue','red','black','pink']


#%% Datos crudos
target_names = ['neutral','sorpresa']
colors=['blue','red']
for color, i, target_name in zip(colors, [3,4], target_names):
    plt.scatter(data_norm[labels == i, 0], data_norm[labels == i, 1], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Datos crudos normalizados')
plt.figure()

#%% PCA Razón de varianza acumulada 

pca = PCA().fit(data_norm)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.title('Razón de varianza acumulada')
plt.xlabel('número de componentes')
plt.ylabel('varianza explicada acumulada');
plt.grid()

#%% PCA (análisis de componentes principales)
target_names = ['neutral','sorpresa']
colors=['blue','red']
pcaObj = PCA(n_components=5)
pca = pcaObj.fit_transform(data_norm)
for color, i, target_name in zip(colors, [3,4], target_names):
    plt.scatter(pca[labels == i, 0], pca[labels == i, 1], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Datos PCA nc=5')
plt.figure()
# plt.boxplot(pca)
# plt.title('PCA nc=5')
# plt.figure()

#%% LDA Razón de varianza acumulada
lda = LinearDiscriminantAnalysis().fit(data,labels)
plt.plot(np.cumsum(lda.explained_variance_ratio_))
plt.title('Razón de varianza acumulada')
plt.xlabel('número de componentes')
plt.ylabel('varianza explicada acumulada');
plt.grid()

#%% LDA análisis de discrimación lineal 
target_names = ['neutral','sorpresa']
colors=['blue','red']
ldaObj = LinearDiscriminantAnalysis(n_components=4)
lda = ldaObj.fit(data_norm,labels).transform(data_norm)

for color, i, target_name in zip(colors, [3,4], target_names):
    plt.scatter(lda[labels == i, 0], lda[labels == i, 3], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Datos lda nc=4')
plt.figure()
# plt.boxplot(lda)
# plt.title('lda nc=5')
# plt.figure()
np.savetxt('caracteristicas-lda.csv',lda,delimiter=',',fmt="%f")
#%% análisis global lda 
target_names = ['asco','enfado','feliz', 'neutral','sorpresa','triste']
datos = [lda[:,0],lda[:,1]]
colors =['orange','green','blue','red','black','pink']


for color,i, target_name in zip(colors,[0,1,2,3,4,5], target_names):
    plt.scatter( lda[labels == i,0],lda[labels == i,2],color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Datos lda nc=4')
plt.figure()



