# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import FastICA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


data = pd.read_csv("caracteristicas_EF.csv").to_numpy()
labels = pd.read_csv("labels.csv").to_numpy()
labels = labels.reshape(330)

data_norm = (data - np.mean(data))/np.std(data) #normalizamos los datos 

lw=2
target_names = ['asco','enfado','feliz', 'neutral','sorpresa','triste']
colors =['orange','green','blue','red','black','pink']

#%% Datos crudos

for color, i, target_name in zip(colors, [2,5], target_names):
    plt.scatter(data_norm[labels == i, 0], data_norm[labels == i, 1], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('datos crudos normalizados')
plt.figure()

#%% PCA (análisis de componentes principales)

pcaObj = PCA(n_components=5)
pca = pcaObj.fit_transform(data_norm)
target_names = ['feliz','triste']
for color, i, target_name in zip(colors, [2,5], target_names):
    plt.scatter(pca[labels == i, 0], pca[labels == i, 1], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('datos PCA nc=5')
plt.figure()

#%% LDA análisis de discrimación lineal 

ldaObj = LinearDiscriminantAnalysis(n_components=3)
lda = ldaObj.fit(data_norm,labels).transform(data_norm)
target_names = ['feliz','triste']

for color, i, target_name in zip(colors, [2,5], target_names):
    plt.scatter(lda[labels == i, 0], lda[labels == i, 1], color=color, alpha=.8,lw=lw,label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('datos lda nc=3')
plt.figure()
