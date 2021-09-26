# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:57:55 2021

@author: nairo
"""

import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay # rendimiento para evaluar los splits 



#%% 2. Cargamos splits (siempre)



train1 = np.loadtxt('train1.txt')
train2 = np.loadtxt('train2.txt')
train3 = np.loadtxt('train3.txt')
train4 = np.loadtxt('train4.txt')
train5 = np.loadtxt('train5.txt')
trains = [train1,train2,train3,train4,train5]

test1 = np.loadtxt('test1.txt')
test2 = np.loadtxt('test2.txt')
test3 = np.loadtxt('test3.txt')
test4 = np.loadtxt('test4.txt')
test5 = np.loadtxt('test5.txt')
tests = [test1,test2,test3,test4,test5]

labelsTrain1 = np.loadtxt('labelsTrain1.txt')
labelsTrain2 = np.loadtxt('labelsTrain2.txt')
labelsTrain3 = np.loadtxt('labelsTrain3.txt')
labelsTrain4 = np.loadtxt('labelsTrain4.txt')
labelsTrain5 = np.loadtxt('labelsTrain5.txt')
labelsTrains =[labelsTrain1,labelsTrain2,labelsTrain3,labelsTrain4,labelsTrain5]

labelsTest1 = np.loadtxt('labelsTest1.txt')
labelsTest2 = np.loadtxt('labelsTest2.txt')
labelsTest3 = np.loadtxt('labelsTest3.txt')
labelsTest4 = np.loadtxt('labelsTest4.txt')
labelsTest5 = np.loadtxt('labelsTest5.txt')
labelsTests =[labelsTest1,labelsTest2,labelsTest3,labelsTest4,labelsTest5]


#%%

#-----------------------------------------------   Modelos de clasificación  -------------------------------------------------------

#fil_train = round(fil*0.7) # número de filas de datos de entrenamientoran_point = np.random.randint(fil_train)
activ_fun = 'relu' # función de activación de capas ocultas , relu, tanh, identity , logistic
n_iter = 3000
modelo_1 = MLPClassifier(
                hidden_layer_sizes=(5),  # número y tamaño de las capas ocultas  ( 1 capa oculta , 5 neuronas )
                #learning_rate_init=0.01,
                #solver = 'lbfgs',    # Algoritmo de optimización utilizado para aprender pesos y bias de la red
                max_iter = n_iter,     # número de iteraciones para entrenamiento.
                activation=activ_fun # función de activación sigmoidal
            )

modelo_2 = MLPClassifier(
                hidden_layer_sizes=(30),
                #solver = 'lbfgs',
                max_iter = n_iter,
                activation = activ_fun
            )


modelos = []
modelos.append(modelo_1)  # almacenamos modelos 
modelos.append(modelo_2)



#%%



#------------------------------------------- Ajuste y  Entrenamiento -------------------------------------------------------------------
redes= []
i = 1

print("Ajustando...")
for modelo in modelos :
    redes.append(modelo.fit(train5,labelsTrain5))

for red in redes:
    print("score entrenamiento {} = {}".format(i,red.score(train5,labelsTrain5))) # calculamos error de entrenamiento
    i+=1

#---------------- ---------------------------------- Fase de Prueba -----------------------------------------------------------------
print("\n------------------------------------------------\n")
i = 1
for red in redes:
    print("score prueba {} = {}".format(i,  red.score(test5,labelsTest5))) # calculamos error de entrenamiento 
    i += 1

Accuracy=[]
for train,labelsTrain,test,labelsTest in zip(trains, labelsTrains, tests, labelsTests):
        modelo_2.fit(train,labelsTrain)
        labels_pred = modelo_2.predict(test)
        accuracy = accuracy_score(labelsTest,labels_pred)
        Accuracy.append(accuracy*100)

print(np.mean(Accuracy))
print(np.std(Accuracy))

cm = confusion_matrix(labelsTest5,labels_pred)
cmp = ConfusionMatrixDisplay(cm, display_labels=['asco','enojo','feliz','neutral','    sorpresa','     triste'])

fig, ax = plt.subplots(figsize=(8,8))
ax.set_title("Matriz de Confusión ",fontsize=20)
cmp.plot(ax =ax)
#ax1.set_title('Matriz')
#ax1.plot()








