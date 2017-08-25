import arff, os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, StratifiedShuffleSplit, StratifiedKFold
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
labels=[]
scores=[]
predicted=[]
#print(dataset['data'])


"""
cria base de dados 
instancias sao dados sem lable

"""
for i in range(len(dataset['data'])):  # percorre a base treino e separa os labels das classes
    labels.append(dataset['data'][i][13])
labels = [int(w) for w in labels]#transforma em int
for i in dataset['data']:
    instancias.append(i[:-1])#tira a classe das instancias



X_train, X_test, y_train, y_test = train_test_split(instancias, labels, test_size=0.3, random_state=None)#divide a base entre treino e teste

cv = ShuffleSplit (n_splits=10, test_size=0.3)#parametro para kfold 10 folds 70, 30
kf=StratifiedKFold(n_splits=10, shuffle=False, random_state=None)


"""arvore de decisao"""
tuned_parameters = [{'splitter': ['best', 'random'], 'max_depth': [3,6,9,12],
                    'max_leaf_nodes': [3,6,9,12], 'max_features': ['auto','sqrt','log2'],
                    'criterion': ['gini','entropy']}]
t = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), tuned_parameters, cv=5, n_jobs=4)
#t=tree.DecisionTreeClassifier()
#t.fit(X_train,y_train)#treina a arvore
predict=cross_val_predict(t,instancias,labels,cv=kf)
m=metrics.accuracy_score(labels, predict)
print("Tree\n")
print(confusion_matrix(labels,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""naive basy gaussiano"""
nb=GaussianNB()
#nb.fit(X_train,y_train)
predict=cross_val_predict(nb,instancias,labels,cv=kf)
m=metrics.accuracy_score(labels, predict)
print("NB\n")
print(confusion_matrix(labels,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")

"""knn"""
tuned = [{'n_neighbors': [i for i in range(1,30)]}]
neigh = KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree')
predict=cross_val_predict(neigh,instancias,labels,cv=kf)
m=metrics.accuracy_score(labels, predict)
print("Knn\n")
print(confusion_matrix(labels,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")
    #neigh.fit(instancias[train_index], labels[train_index])
#ypred = neigh.predict(instancias[test_index],labels[test_index])
    #kappa_score = cohen_kappa_score(labels[test_index], ypred)
    #confusion_matrix = confusion_matrix(labels[test_index], ypred)
#print cross_val_score(neigh, instancias, labels, cv=kf, n_jobs=1)
#nn=KNeighborsClassifier(n_neighbors=1,algorithm='kd_tree', )
#nn.fit(X_train,y_train)
#print(neigh.score(instancias,labels))
#for i in range(len(instancias)):
    #nn.predict(instancias)

#nn.fit(instancias,labels)
# predict=cross_val_predict(nn,instancias,labels,cv=10)
# m=metrics.accuracy_score(labels, predict)
# print("Knn\n")
# print(confusion_matrix(labels,predict))
# scores.append(m)#media dos scores do crosvalie
# print ("\n")

"""MLP"""
mlp=MLPClassifier(solver='lbfgs', alpha=.1, hidden_layer_sizes=(5, 5), random_state=1)
#mlp.fit(X_train,y_train)
predict=cross_val_predict(mlp,instancias,labels,cv=kf)
m=metrics.accuracy_score(labels, predict)
print("MLP\n")
print(confusion_matrix(labels,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2, 1, 4e-1, 2e-1],
                    'C': [5e-1, 5, 10, 50, 500]},
                     {'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 10, 100, 1000]}]
"""Svm"""
clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, n_jobs=4)
#clf_svm.fit(X_train, y_train)
predict=cross_val_predict(clf_svm,instancias,labels,cv=kf)
#print(predict)
m=metrics.accuracy_score(labels, predict)
print("SVM\n")
print(confusion_matrix(labels,predict))
scores.append(m)#media dos scores do crosvalie
print ("\n")
print(scores)


