import arff, numpy, random, os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
global resultados, media, maior
resultados=[]
def cria_dataset(dataset):
    lables =[]
    instancias=[]
    for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
        lables.append(dataset['data'][i][16])
        lables = [str(w) for w in lables]#transforma as classes em string
    for i in dataset['data']:
        instancias.append(i[:-1])#salva so as instancias(sem classes)
    X_train, X_x, y_train, y_y = train_test_split(instancias, lables, test_size=0.5, random_state=True, stratify=lables)#divide a base de treino
    X_test, X_val, y_test, y_val= train_test_split(X_x, y_y, test_size=0.5, random_state=None, stratify=y_y)#divide a base entre teste e validacao
    return X_train, X_test, X_val, y_train, y_test, y_val # X_ -> bases sem lables, y_-> lables

def cria_classificadores(X_train, y_train, repeticoes):
    #bags=list()
    bagging=[]
    tree = DecisionTreeClassifier()
    for i in range(repeticoes):
        r = random.seed()
        X_bag, X_test, y_bag, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=r, stratify=y_train)  # divide o treino para criar os bags
        tree.fit(X_bag,y_bag)
        joblib.dump(tree, "clf/TreeClas"+str(i)+".pkl")#salva o classificador

def acuracia(repeticao, X_test, y_test):#mede a acuracia do classificador, retorna a media, o maior, os resultados e o nome do maior
    resultados = []
    anterior=0
    for i in range(repeticao):
        classificador = joblib.load('clf/TreeClas' + str(i) + '.pkl')
        atual=classificador.score(X_test,y_test)
        resultados.append(atual)
        if (atual>anterior):
            anterior=atual
            nome='TreeClas'+str(i)
    maior=max(resultados)
    media=numpy.mean(resultados)
    return maior, media, resultados, nome

def main():
    dataset = arff.load(open('letter.arff'))
    X_train, X_test, X_val, y_train, y_test, y_val =  cria_dataset(dataset)
    if (os.path.exists("clf/TreeClas99.pkl") == False):
        cria_classificadores(X_train,y_train,100)
    else:
        maior, media, resultados, nome=acuracia(100,X_test,y_test)
        print(nome)
        print (maior)
        print (resultados)
if __name__ == '__main__':
        main()