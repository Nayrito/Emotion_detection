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
import time


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
Accuracy=[]
activ_fun = 'relu' # función de activación de capas ocultas , relu, tanh, identity , logistic
n_iter = 3000
neuronas=[5,10,15,20,30]
MLPClassifier(hidden_layer_sizes=(30),max_iter = n_iter,activation = activ_fun)
for n in neuronas:
    mlp =MLPClassifier(hidden_layer_sizes=(n),max_iter = n_iter,activation = activ_fun)
    
    for train,labelsTrain,test,labelsTest in zip(trains, labelsTrains, tests, labelsTests):
        mlp.fit(train,labelsTrain)
        labels_pred = mlp.predict(test)
        accuracy = accuracy_score(labelsTest,labels_pred)
        Accuracy.append(accuracy*100)
#%%
z = np.asarray(Accuracy)
z = np.reshape(z,(5,5))
#z = z.T
x = [1,2,3,4,5]
y= [5,10,15,20,30]
fig = plt.figure()
ax3d = plt.axes(projection="3d")
X,Y = np.meshgrid(x,y)
ax3d.plot_surface(X, Y, z,cmap='plasma',linewidth=0, )
ax3d.view_init(20, 40)  #25,20
ax3d.set_title('Accuracy')
ax3d.set_xlabel('fold')
ax3d.set_ylabel('neuronas')
ax3d.set_zlabel('%')
plt.show()


#%%

#---------------- ---------------------------------- Fase de Prueba -----------------------------------------------------------------
print("\n------------------------------------------------\n")


Accuracy=[]
Accuracy_train=[]
train_times = []
for train,labelsTrain,test,labelsTest in zip(trains, labelsTrains, tests, labelsTests):
    #Entrenamiento
    inicio = time.time()
    modelo_2.fit(train,labelsTrain)
    train_times.append(time.time()-inicio)
    Accuracy_train.append(modelo_2.score(train,labelsTrain))
    #Prueba
    labels_pred = modelo_2.predict(test)
    accuracy = accuracy_score(labelsTest,labels_pred)
    Accuracy.append(accuracy*100)

print ("mean train accuracy: {}".format(np.mean(Accuracy_train)))
print ("mean train time: {}".format(np.mean(train_times)))
print("mean accuracy: {}".format(np.mean(Accuracy)))
print("std accuracy: {}".format(np.std(Accuracy)))

cm = confusion_matrix(labelsTest5,labels_pred)
cmp = ConfusionMatrixDisplay(cm, display_labels=['asco','enojo','feliz','neutral','    sorpresa','     triste'])

fig, ax = plt.subplots(figsize=(8,8))
ax.set_title("Matriz de Confusión MLP",fontsize=20)
cmp.plot(ax =ax)









