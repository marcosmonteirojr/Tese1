import arff, os
import numpy as np
from sklearn.model_selection import  StratifiedKFold
from sklearn import preprocessing
from sklearn import tree
dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
smlables=[]
lables=[]
for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
    smlables.append(dataset['data'])
    lables.append(dataset['data'][i][13])
smlables=np.array(smlables)
lables=np.array(lables)
#print (lables)
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(smlables, lables)
print(skf)
for train_index, test_index in skf.split(smlables, lables):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = smlables[train_index], smlables[test_index]
    y_train, y_test = lables[train_index], lables[test_index   ]

print(X_test)
