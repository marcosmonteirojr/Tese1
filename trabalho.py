import arff, os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
lables=[]
for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
    lables.append(dataset['data'][i][13])
print (lables)