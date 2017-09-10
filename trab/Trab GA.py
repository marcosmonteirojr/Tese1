import arff
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit, cross_val_predict, StratifiedShuffleSplit, StratifiedKFold


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

def main():
    dataset = arff.load(open('letter.arff'))
    X_train, X_test, X_val, y_train, y_test, y_val =  cria_dataset(dataset)
    print (X_train)
    print (y_train)
if __name__ == '__main__':
        main()