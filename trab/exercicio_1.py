#!/usr/bin/python
#
from __future__ import print_function
import arff
import sys
import graphviz 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
#
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
#
#
# --------------------- Print best parameters --------------------
#
#
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
#
#
# --------------- Execute repeatitions of a hold out ---------------
#
#
def test_classifier(clf, repeats):
    global_predict = list()
    for i in repeats:
        clf.fit(i[0], i[1])
        #print(clf.score(i[2],i[3]))
        predict = list()
        for j in range(len(i[2])):
            predict.append(clf.predict_proba(np.array([i[2][j]])))
        global_predict.append(predict)
    correct = 0
    total = 1
    #
    confusions = list()
    scores = list()
    #
    for i in range(len(global_predict)):
        t = np.array(repeats[i][3]).max()
        confusion = [[0 for x in range(t+1)] for y in range(t+1)]
        for j in range(len(global_predict[i])):
            if(np.argmax(np.array(global_predict[i][j][0])) == repeats[i][3][j]):
                correct += 1
            confusion[np.argmax(np.array(global_predict[i][j][0]))][repeats[i][3][j]] += 1
#            if(global_predict[i][j][0][0] > global_predict[i][j][0][1]):
#                if(repeats[i][3][j] == 0):
#                    confusion[0][0] += 1
#                    correct += 1
#                else:
#                    confusion[0][1] += 1
#            if(global_predict[i][j][0][0] < global_predict[i][j][0][1]):
#                if(repeats[i][3][j] == 1):
#                    confusion[1][1] += 1
#                    correct += 1
#                else:
#                    confusion [1][0] += 1
            total += 1
        confusions.append(confusion)
        confusion = [[0 for x in range(t+1)] for y in range(t+1)]
        scores.append(float(correct)/total)
        correct = 0
        total = 0
    return(np.array(scores), confusions)
#
#
# -------------- Print confusion matrix -----------
#
#
def print_confusion_matrix(confusions, which_matrix):
    if( len(confusions) < which_matrix ):
        return 0
    for i in range(len(confusions[which_matrix])):
        print("p\\a;", end="")
        for j in range(len(confusions[which_matrix][i])):
            print("{};".format(j), end="")
        break
    print()
    for i in range(len(confusions[which_matrix])):
        print("{};".format(i), end="")
        for j in range(len(confusions[which_matrix][i])):
            print("{};".format(confusions[which_matrix][i][j]), end="")
        print()
    return 1
#
#
#
# -------------- Start of the program ------------
#
#
if(len(sys.argv) != 2):
    print("Use: exercicio_1.py [base]")
    exit(0)
#
#
# ------------ Read the dataset ------------
#
#
base_file = open(sys.argv,"r")
dataset = arff.load(open(base_file))
#
#
# ------------ Load liver disorder dataset -------------
#
#
def load_liver_disorder(base):
    X = list()
    Y = list()
    T = list()
    #
    for i in base['data']:
        x_temp = list()
        x_temp.append(float(i[0]))
        x_temp.append(float(i[1]))
        x_temp.append(float(i[2]))
        x_temp.append(float(i[3]))
        x_temp.append(float(i[4]))
        X.append(x_temp)
        if(float(i[5]) > 3):
            Y.append(1)
        else:
            Y.append(0)
        T.append(int(i[6]))
    X_train = list()
    X_test = list()
    Y_train = list()
    Y_test = list()
    for i in range(len(T)):
        if(T[i] == 1):
            X_test.append(X[i])
            Y_test.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return X,Y
#
#
# --------------- Load wine dataset
#
#
def load_wine(dataset):

    for i in range(len(dataset['data'])):  # percorre a base treino e separa os lables das classes
        Y.append(dataset['data'][i][13])
    lables = [int(w) for w in Y]  # transforma em int
    for i in dataset['data']:
        X.append(i[:-1])  # tira a classe das instancias
#
X, Y = load_wine(dataset)
#X, Y = load_liver_disorder(base)
#
base_file.close()
#
#
# ------------- Hold-out 70%-30% ----------
#
#
repeats = list()
for i in range(10):
    X_tmp_train, X_tmp_test, Y_tmp_train, Y_tmp_test = train_test_split(X, Y, test_size=0.30)
    repeats.append([X_tmp_train,Y_tmp_train,X_tmp_test,Y_tmp_test])
#
#
# -------------- Monolitic ---------------
#
#
print("-----------------Tree------------------")
#
tuned_parameters = [{'splitter': ['best', 'random'], 'max_depth': [3,6,9,12],
                    'max_leaf_nodes': [3,6,9,12], 'max_features': ['auto','sqrt','log2'], 
                    'criterion': ['gini','entropy']}]
c45 = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy'), tuned_parameters, cv=5, scoring='precision_weighted', n_jobs=4)
scores, confusions = test_classifier(c45, repeats)
print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
print_confusion_matrix(confusions, scores.argmax())
print_parameters(c45)
#
print("-----------------Naive Bayes------------------")
#
naive = GaussianNB()
scores, confusions = test_classifier(naive, repeats)
print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
print_confusion_matrix(confusions, scores.argmax())
#
print("-----------------SVM------------------")
#
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 6e-1, 4e-1, 2e-1],
#                    'C': [5e-1, 5, 50, 500, 5000]},
#                    {'kernel': ['linear'], 'C': [1e-1, 1, 10, 100, 1000]}]
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [2, 1, 4e-1, 2e-1],
                    'C': [5e-1, 5, 10, 50, 500]},
                     {'kernel': ['linear'], 'C': [1e-2, 1e-1, 1, 10, 100, 1000]}]
smo = GridSearchCV(svm.SVC(probability=True), tuned_parameters, cv=5, scoring='precision_weighted', n_jobs=4)
scores, confusions = test_classifier(smo, repeats)
print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
print_confusion_matrix(confusions, scores.argmax())
print_parameters(smo)
#
print("-----------------KNN------------------")
#
tuned_parameters = [{'n_neighbors': [i for i in range(1,30)]}]
neigh = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='precision_weighted', n_jobs=4)
scores, confusions = test_classifier(neigh, repeats)
print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
print_confusion_matrix(confusions, scores.argmax())
print_parameters(neigh)
#
#
# --------------- Bagging -----------------
#
#
print("-----------------Bagging+CART (Classification and Regression Tree)------------------")
#
c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best')
for i in [0.5, 0.6, 0.8, 1]:
    for j in [5, 10, 25, 50, 75, 100]:
        print("Bagging: max_samples={} n_estimators={}".format(i,j))
        bagging_bag = BaggingClassifier(c45, n_estimators=j, max_samples=i)
        bagging_bag.predict_proba(.)
        scores, confusions = test_classifier(bagging_bag, repeats)
        print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
        print_confusion_matrix(confusions, scores.argmax())
#
print("-----------------RSS+CART------------------")
#
c45 = tree.DecisionTreeClassifier(criterion='entropy',max_features='sqrt', max_leaf_nodes=12, max_depth=3, splitter='best')
for i in [0.5, 0.7, 0.9]:
    for j in [5, 10, 25, 50, 75, 100]:
        print("RSS: max_features={} n_estimators={}".format(i,j))
        bagging_rss = BaggingClassifier(c45, n_estimators= j, max_samples=1, max_features=i)
        scores, confusions = test_classifier(bagging_rss, repeats)
        print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
        print_confusion_matrix(confusions, scores.argmax())
#
print("-----------------RandomForest------------------")
#
for i in [5, 10, 25, 50, 75, 100]:
    print("RandomForest: {}".format(i))
    bagging_rf = RandomForestClassifier(n_estimators=i)
    scores, confusions = test_classifier(bagging_rf, repeats)
    print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
    print_confusion_matrix(confusions, scores.argmax())
#
print("-----------------AdaBoost------------------")
#
for i in [25, 50, 75, 100]:
    print("Adaboost: {}".format(i))
    boosting = AdaBoostClassifier(n_estimators=i)
    scores, confusions = test_classifier(boosting, repeats)
    print("Media: {:.5f}, Desvio: {:.5f}, Melhor: {:.5f}, Pior: {:.5f}".format(scores.mean(), scores.std(), scores.max(), scores.min()))
    print_confusion_matrix(confusions, scores.argmax())
#
exit(0)
#
#dot_data = tree.export_graphviz(c45, out_file=None,
#                feature_names=['mean corpuscular volume', 'alkaline phosphotase', 'alamine aminotransferase', 'aspartate aminotransferase', 'gamma-glutamyl transpeptidase'],
#                class_names=['not drink','drink'],  
#                filled=True, rounded=True,  
#                special_characters=True)
#
#graph = graphviz.Source(dot_data) 
#graph.render("exercicio_liver_disorder_pdf") 
#clf.predict_proba([[2., 2.]])
#