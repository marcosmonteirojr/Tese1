import arff
import numpy as np
from sklearn.linear_model import perceptron
lables=[]
dados = list()
dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True, eta0=0.002)

#dados=(dataset['data'][0][:-1])


dados['data'] = list()
for i in range (len(dataset['data'])):
    dados = dados['data'].append(dataset['data'][i][:-1])
    lables.append(dataset['data'][i][13])
    print(dados['data'])
net.fit(dados,lables)

#print(lables)
