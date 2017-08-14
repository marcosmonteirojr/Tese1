import arff, os
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import perceptron
lables=[]
dados = list()
treino = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
teste = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))
nome_base="Wine"
#net = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)

if (os.path.exists("clf_pec"+nome_base+".pkl") == False):
    perc = perceptron.Perceptron(n_iter=100, verbose=0, random_state=None, fit_intercept=True)
    for i in range(len(treino['data'])):
        lables.append(treino['data'][i][13])
    for i in treino['data']:
        dados.append(i[:-1])
    perc.fit(dados,lables)
    print(perc.score(dados,lables))
    joblib.dump(perc, "clf_pec"+nome_base+".pkl")
    print("entrei if")
else:
    perc = joblib.load("clf_pec"+nome_base+".pkl")
    for i in teste["data"]:
        dados.append(i[:-1])
    for i in range (len(teste['data'])):
         dados2=np.array([dados[i]])
         print(perc.predict(dados2))
