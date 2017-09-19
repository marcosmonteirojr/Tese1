import arff, os
import numpy as np

from sklearn.model_selection import train_test_split,  ShuffleSplit, cross_val_predict,  StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import tree, metrics, svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


dataset = arff.load(open('/home/marcos/Documentos/wine.arff'))
dataset2 = arff.load(open('/home/marcos/Downloads/diabetes.arff'))

scores=[]
predicted=[]


def print_parameters(clf):
    print("Best parameters with score {:.5f}% set found on development set:".format(clf.best_score_))
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
"""
cria base de dados 
instancias sao dados sem lable

"""
def wine(dataset):
    global instancias, lables, X_test, X_train, y_test, y_train, kf
    instancias = []
    lables = []
    for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
        lables.append(dataset['data'][i][13])
    lables = [int(w) for w in lables]#transforma em int
    for i in dataset['data']:
        instancias.append(i[:-1])#tira a classe das instancias
    X_train, X_test, y_train, y_test = train_test_split(instancias, lables, test_size=0.3, random_state=None)#divide a base entre treino e teste
    kf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None, stratify=lables)
    return X_train, X_test, y_train, y_test, kf

def diabetes(dataset):
    global instancias, lables, X_test, X_train, y_test, y_train, kf
    instancias = []
    lables = []
    for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
        lables.append(dataset['data'][i][8])
    lables = [str(w) for w in lables]  # transforma em int
    print(lables)
    for i in dataset['data']:
        instancias.append(i[:-1])  # tira a classe das instancias
    X_train, X_test, y_train, y_test = train_test_split(instancias, lables, test_size=0.3,
                                                        random_state=None, stratify=lables)  # divide a base entre treino e teste
    kf = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
    return X_train, X_test, y_train, y_test, kf

def Classificadores():
    diabetes(dataset2)

    """arvore de decisao"""
    tuned_parameters = [{'splitter': ['best', 'random'], 'max_depth': [3,6,9,12],
                        'max_leaf_nodes': [3,6,9,12], 'max_features': ['auto','sqrt','log2'],
                        'criterion': ['gini','entropy']}]
    t = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), tuned_parameters, cv=kf, n_jobs=4)
    predict=cross_val_predict(t,X_test,y_test,cv=kf)
    m=metrics.accuracy_score(y_test, predict)
    print("Tree\n")
    print(confusion_matrix(y_test,predict))
    scores.append(m)#media dos scores do crosvalie
    #print_parameters(t)
    print ("\n")
    #exit(0)

    """naive gaussiano"""
    nb=GaussianNB()
    nb.fit(X_train,y_train)
    predict=cross_val_predict(nb,instancias,lables,cv=kf)
    m=metrics.accuracy_score(lables, predict)
    print("NB\n")
    print(confusion_matrix(lables,predict))
    scores.append(m)#media dos scores do crosvalie
    print ("\n")

    """knn"""
    tuned_parameters = [{'n_neighbors': [i for i in range(1,20)],'algorithm':['ball_tree','kd_tree'], 'leaf_size':[10,20,30] }]
    neigh = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=kf, n_jobs=4)
    #neigh.fit(X_train,y_train)
    predict=cross_val_predict(neigh,X_test,y_test,cv=kf)
    m=metrics.accuracy_score(y_test, predict)
    print("Knn\n")
    print(confusion_matrix(y_test,predict))
    scores.append(m)#media dos scores do crosvalie
    #print_parameters(neigh)
    print ("\n")

    """MLP"""
    tuned_parameters = [{'hidden_layer_sizes': [(4)], 'activation': ['tanh'],
                       'solver': ['adam'], 'alpha': [1e-4, 1e-3],
                       'learning_rate': ['adaptive'],
                       'max_iter': [100],
                       'tol': [1e-4],
                       'momentum': [0.5]}]
    mlp = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy', n_jobs=4)
    mlp.fit(X_train,y_train)
    predict=cross_val_predict(mlp,X_test,y_test,cv=kf)
    m=metrics.accuracy_score(y_test,predict)
    print("MLP\n")
    print(confusion_matrix(y_test,predict))
    scores.append(m)#media dos scores do crosvalie
    print ("\n")
    print_parameters(mlp)
   #  """Svm"""

    #  tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2, 1, 4e-1, 2e-1],
    #                       'C': [5e-1, 5, 10, 50, 500]},
    #                       {'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 10, 100, 1000]}]
    #  clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters,cv=kf, n_jobs=8)
    #
    # # clf.fit(X_train, y_train)
    #  predict=cross_val_predict(clf_svm,X_test,y_test,cv=kf)
    #  m=metrics.accuracy_score(y_test,predict)
    #  print("SVM\n")
    #  print(confusion_matrix(y_test,predict))
    #  scores.append(m)#media dos scores do crosvalie
    #  #print_parameters(clf_svm)
    #  print("\n")

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [4e-1, 2e-1],
                          'C': [5e-1, 5]},
                          {'kernel': ['linear'], 'C': [1e-2, 1e-1]}]
    clf_svm = GridSearchCV(SVC(probability=True), tuned_parameters,cv=kf, n_jobs=8)

    # clf.fit(X_train, y_train)
    predict=cross_val_predict(clf_svm,X_test,y_test,cv=kf)
    m=metrics.accuracy_score(y_test,predict)
    print("SVM\n")
    print(confusion_matrix(y_test,predict))
    scores.append(m)#media dos scores do crosvalie
     #print_parameters(clf_svm)
    print("\n")
    #print_parameters(clf_svm)

    """Bagging"""
    c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best',)
    x=0
    for i in [0.5, 0.6, 0.8, 1]:
        for j in [5, 10, 25, 50, 75, 100]:

            bagging_bag = BaggingClassifier(c45, n_estimators=j, max_samples=i)
            predict = cross_val_predict(bagging_bag, X_test, y_test, cv=kf)
            y = metrics.accuracy_score(y_test, predict)
            if(x<y):
                x=y
                maxx=i
                estimator=j
    bagging_bag = BaggingClassifier(c45, n_estimators=estimator, max_samples=maxx)

    predict = cross_val_predict(bagging_bag, X_test, y_test, cv=kf)
    m = metrics.accuracy_score(y_test, predict)
    print("Bagging\n")
    print(confusion_matrix(y_test,predict))
    print("Baggign, Melhor parametro: n_estimator={}, Max_samples={}, score={}".format(estimator, maxx,m))
    scores.append(m)#media dos scores do crosvalie
    print ("\n")

    '''Rss'''
    #
    c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best')
    x=0
    for i in [0.5, 0.7, 0.9]:
        for j in [5, 10, 25, 50, 75, 100]:

            rss = BaggingClassifier(c45, n_estimators=j, max_features=i,max_samples=1)
            predict = cross_val_predict(rss, X_test, y_test, cv=kf)
            y = metrics.accuracy_score(y_test, predict)
            if (x < y):
                x = y
                maxx = i
                estimator = j
    rss = BaggingClassifier(c45, n_estimators=estimator, max_samples=1,max_features=maxx)
    predict = cross_val_predict(rss, X_test, y_test, cv=kf)
    m = metrics.accuracy_score(y_test, predict)
    print("RSS\n")
    print(confusion_matrix(y_test, predict))
    print("Rss, Melhor parametro: n_estimator={}, Max_features={}, score={}".format(estimator, maxx,m))
    scores.append(m)  # media dos scores do crosvalie
    print ("\n")
    #
    """RF"""
    #
    x=0
    for i in [5, 10, 25, 50, 75, 100]:
        Rf = RandomForestClassifier(n_estimators=i)
        predict = cross_val_predict(Rf, X_test, y_test, cv=kf)
        y = metrics.accuracy_score(y_test, predict)
        if (x < y):
            x = y
            estimator = i
    Rf = RandomForestClassifier(n_estimators=estimator)

    predict = cross_val_predict(Rf, X_test, y_test, cv=kf)
    m = metrics.accuracy_score(y_test, predict)
    print("RF\n")
    print(confusion_matrix(y_test, predict))
    print("RF, Melhor parametro: n_estimator={}, score={}".format(estimator, m))
    scores.append(m)  # media dos scores do crosvalie
    print ("\n")

    #
    """Boosting"""
    #
    for i in [25, 50, 75, 100]:
        boosting = AdaBoostClassifier(n_estimators=i)
        predict = cross_val_predict(boosting, X_test, y_test, cv=kf)
        y = metrics.accuracy_score(y_test, predict)
        if (x < y):
            x = y
            estimator = i
    boosting = RandomForestClassifier(n_estimators=estimator)
    predict = cross_val_predict(boosting, X_test, y_test, cv=kf)
    m = metrics.accuracy_score(y_test, predict)
    print("boosting\n")
    print(confusion_matrix(y_test, predict))
    print("Boosting, Melhor parametro: n_estimator={}, score={}".format(estimator, m))
    scores.append(m)  # media dos scores do crosvalie
    print ("\n")
    print ("\n")



Classificadores()
print(scores)