import arff, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import tree
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
instancias=[]
lables=[]
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
#instancias=np.array(instancias)
#lables=np.array(lables)?
#print (instancias)
#for i in range(20):
X_train, X_test, y_train, y_test = train_test_split(instancias, lables, test_size=0.3, random_state=45, shuffle=True)



t=tree.DecisionTreeClassifier()
model=t.fit(X_train,y_train)
t.predict(X_test,y_test)



nb=GaussianNB()
nb.fit(X_train,y_train)

nn=NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
nn.fit(X_train,y_train)

mlp=MLPClassifier(solver='lbfgs', alpha=1e-1,hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train,y_train)

print (mlp.score(X_test,y_test))



