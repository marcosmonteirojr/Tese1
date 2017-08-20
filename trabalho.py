import arff, os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree
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
lables = [int(w) for w in lables]
for i in dataset['data']:
    instancias.append(i[:-1])


print(lables)

#print(lables)
print(instancias)

#instancias=np.array(instancias)
#lables=np.array(lables)?
#print (instancias)
#for i in range(20):
X_train, X_test, y_train, y_test = train_test_split(instancias, lables, test_size=0.3, random_state=45, shuffle=True)
print (X_test)

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
                         'C': [5e-1, 5, 50, 500, 5000]},
                        {'kernel': ['linear'], 'C': [1e-1, 1, 10, 100, 1000]}]

t=tree.DecisionTreeClassifier()
model=t.fit(X_train,y_train)
tt=t.predict(X_test)
#print(confusion_matrix(tt,y_test))
scores.append(t.score(X_test,y_test))

nb=GaussianNB()
nb.fit(X_train,y_train)
scores_cross=cross_val_score(nb,instancias,lables,cv=10)
scores.append(nb.score(X_test,y_test))
print (scores_cross.mean())

#print(confusion_matrix(nbb,y_test))

nn=KNeighborsClassifier(n_neighbors=20,algorithm='kd_tree', weights='distance')
nn.fit(X_train,y_train)
scores.append(nn.score(X_test,y_test))
scores_cross=cross_val_score(nn,instancias,lables,cv=10)
print (scores_cross.mean())

mlp=MLPClassifier(solver='lbfgs', alpha=.1, hidden_layer_sizes=(5, 3), random_state=1)
mlp.fit(X_train,y_train)
scores.append(mlp.score(X_test,y_test))

#print (confusion_matrix(mlp.predict(X_test),y_test))

clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, n_jobs=4)
clf_svm.fit(X_train, y_train)

scores.append((clf_svm.score(X_test, y_test)))

print(scores)


