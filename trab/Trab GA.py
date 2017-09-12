import arff, numpy, random, os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

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

    tree = DecisionTreeClassifier()
    for i in range(repeticoes):
        r = random.seed()
        X_bag, X_test, y_bag, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=r, stratify=y_train)  # divide o treino para criar os bags
        tree.fit(X_bag,y_bag)
        joblib.dump(tree, "clf/TreeClas"+str(i)+".pkl")#salva o classificador

def acuracia(classificador, X_test, y_test):#mede a acuracia do classificador
    print(classificador.score(X_test,y_test))


def main():
    dataset = arff.load(open('letter.arff'))
    X_train, X_test, X_val, y_train, y_test, y_val =  cria_dataset(dataset)
    if (os.path.exists("clf/TreeClas99.pkl") == False):
        cria_classificadores(X_train,y_train,100)
    else:
        for i in range(100):
            classificador=joblib.load('clf/TreeClas'+str(i)+'.pkl')
            acuracia(classificador,X_test,y_test )
if __name__ == '__main__':
        main()