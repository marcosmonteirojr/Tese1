import arff
import numpy as np
from sklearn.linear_model import perceptron
lables=[]
lables2=[]
dados = list()
dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)





for i in range (len(dataset['data'])):
    lables.append(dataset['data'][i][13])
for i in dataset['data']:
    dados.append(i[:-1])
net.fit(dados,lables)
print(net.score(dados,lables))
for i in range (len(dataset['data'])):
    dados2=np.array([dados[i]])
    #print(net.predict(dados2))

