import arff, os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, StratifiedShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree, metrics
#from sklearn.model_selection import cross_val_score
dataset = arff.load(open('/home/marcos/Documentos/wine.arff'))
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
instancias=[]
sminstancias=[]
lables=[]
scores=[]
#print(dataset['data'])


"""
cria base de dados 
instancias sao dados sem lable

"""
for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
    lables.append(dataset['data'][i][13])
lables = [int(w) for w in lables]#transforma em int
for i in dataset['data']:
    instancias.append(i[:-1])#tira a classe das instancias



X_train, X_test, y_train, y_test = train_test_split(instancias, lables, test_size=0.3, random_state=None)#divide a base entre treino e teste

cv = StratifiedShuffleSplit (n_splits=10, test_size=0.3)#parametro para kfold 10 folds 70, 30

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
                         'C': [5e-1, 5, 50, 500, 5000]},
                        {'kernel': ['linear'], 'C': [1e-1, 1, 10, 100, 1000]}]
"""arvore de decisao"""
t=tree.DecisionTreeClassifier()
t.fit(X_train,y_train)#treina a arvore
predict=cross_val_predict(t,X_test,y_test,cv=16)
m=metrics.accuracy_score(y_test, predict)
print("Tree\n")
print(confusion_matrix(y_test,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""naive basy gaussiano"""
nb=GaussianNB()
nb.fit(X_train,y_train)
predict=cross_val_predict(nb,X_test,y_test,cv=16)
m=metrics.accuracy_score(y_test, predict)
print("NB\n")
print(confusion_matrix(y_test,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""knn"""
nn=KNeighborsClassifier(n_neighbors=29,algorithm='kd_tree', weights='distance')
nn.fit(X_train,y_train)
predict=cross_val_predict(nn,X_test,y_test,cv=16)
m=metrics.accuracy_score(y_test, predict)
print("Knn\n")
print(confusion_matrix(y_test,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""MLP"""
mlp=MLPClassifier(solver='lbfgs', alpha=.1, hidden_layer_sizes=(5, 3), random_state=1)
mlp.fit(X_train,y_train)
predict=cross_val_predict(mlp,X_test,y_test,cv=16)
m=metrics.accuracy_score(y_test, predict)
print("MLP\n")
print(confusion_matrix(y_test,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""Svm"""
clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, n_jobs=4)
clf_svm.fit(X_train, y_train)
predict=cross_val_predict(t,X_test,y_test,cv=16)
m=metrics.accuracy_score(y_test, predict)
print("SVM\n")
print(confusion_matrix(y_test,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")
print(scores)


