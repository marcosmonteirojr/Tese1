import arff, numpy, random, os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
global resultados, media, maior
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from sklearn.ensemble import VotingClassifier
classifiers = list()
current_ind = 1
resultados=[]

def cria_dataset(dataset): #cria treino teste e validacao e grava em arquivo
    print('Criando dataset\n')
    lables =[]
    instancias=[]
    for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
        lables.append(dataset['data'][i][16])
        lables = [str(w) for w in lables]#transforma as classes em string
    for i in dataset['data']:
        instancias.append(i[:-1])#salva so as instancias(sem classes)
    X_train, X_x, y_train, y_y = train_test_split(instancias, lables, test_size=0.5, random_state=True, stratify=lables)#divide a base de treino
    X_test, X_val, y_test, y_val= train_test_split(X_x, y_y, test_size=0.5, random_state=None, stratify=y_y)#divide a base entre teste e validacao

    dados = dict()
    X=list()
    for i in range(len(y_train)):
        linha_arff=list()
        for j in X_train[i]:
            linha_arff.append(j)
        linha_arff.append(y_train[i])
        X.append(linha_arff)
    dados['data'] = X
    print ('Criando treino\n')
    cria_arff(dataset,dados,"treino",'dataset')


    dados = dict()
    X = list()
    for i in range(len(y_test)):
        linha_arff = list()
        for j in X_test[i]:
            linha_arff.append(j)
        linha_arff.append(y_test[i])
        X.append(linha_arff)
    dados['data'] = X
    print ('Criando teste\n')
    cria_arff(dataset, dados, "teste",'dataset')


    for i in range(len(y_val)):
        linha_arff = list()
        for j in X_val[i]:
            linha_arff.append(j)
        linha_arff.append(y_val[i])
        X.append(linha_arff)
    dados['data'] = X
    cria_arff(dataset, dados, "validacao",'dataset')
    print ('Criando validacao\n')
    return X_train, X_test, X_val, y_train, y_test, y_val # X_ -> bases sem lables, y_-> lables


def cria_arff(info, data, nome, pasta):
    obj = {
        'description': info['description'],
        'relation': info['relation'],
        'attributes': info['attributes'],
        'data': data['data'],

    }
    arq1 = arff.dumps(obj)
    arq = open(pasta + '/' + nome + '.arff', 'w')
    arq.write(arq1)
    arq.close()


def cria_classificadores(X_train, y_train, repeticoes, dataset):#cria classificadores e os bags
   
    tree = DecisionTreeClassifier()
    for x in range(repeticoes):
        r = random.seed()
        X_bag, X_test, y_bag, y_test = train_test_split(X_train, y_train, test_size=0.5, random_state=r, stratify=y_train)  # divide o treino para criar os bags
        dados = dict()
        X = list()
        for i in range(len(y_bag)):
            linha_arff = list()
            for j in X_bag[i]:
                linha_arff.append(j)
            linha_arff.append(y_bag[i])
            X.append(linha_arff)
        dados['data'] = X
        cria_arff(dataset, dados, "bags"+str(x), 'bag')
        print('Criando bags-bag'+str(x))
        tree.fit(X_bag,y_bag)
        joblib.dump(tree, "clf/TreeClas"+str(x)+".pkl")#salva o classificador
        print('Criando classificador-Tree' + str(x))

def abre_arff(dataset):
    lables = []
    instancias = []
    for i in range(len(dataset['data'])):  # percorre a base e separa os labels das classes
        lables.append(dataset['data'][i][16])
        lables = [str(w) for w in lables]  # transforma as classes em string
    for i in dataset['data']:
        instancias.append(i[:-1])  # salva so as instancias(sem classes)
    return instancias, lables

def carrega_classificadores():
    global nome
    nome=[]
    for i in range(100):
        nome.append('TreeClas'+str(i))
        classifiers.append(joblib.load('clf/TreeClas'+str(i)+'.pkl'))
    print ('carregando classificadores')

    return classifiers, nome


# def acuracia(repeticao, X_test, y_test):#mede a acuracia do classificador, retorna a media, o maior, os resultados e o nome do maior
#     resultados = []
#     anterior=0
#     for i in range(repeticao):
#         classificador = joblib.load('clf/TreeClas' + str(i) + '.pkl')
#         atual=classificador.score(X_test,y_test)
#         resultados.append(atual)
#         if (atual>anterior):
#             anterior=atual
#             nome='TreeClas'+str(i)
#     maior=max(resultados)
#     media=numpy.mean(resultados)
#     return maior, media, resultados, nome

def CombineBySum(results):
    if (len(results) > 0):
        if (len(results[0]) > 0):
            vote_list = [0 for i in range(len(results[0]))]
            for i in results:
                for j in range(len(i)):
                    vote_list[j] += i[j]
            return np.argmax(np.array(vote_list))
        return -1
    return -1

def evalEnsemble(individual):
    print (individual)
    global current_ind
    c=[]
    n=[]
    for j in range(len(individual)):
        if (individual[j] == 1):
            nome_val=nome[j]
            pred_val = classifiers[j]
            c.append(pred_val)
            n.append(nome_val)

    eclf = VotingClassifier(estimators=zip(n,c),voting='soft')
    c.append(eclf)
    n.append('Ensemble')
  #  print(zip(c,n))
    for clf, label in zip(c,n):
        scores = cross_val_score(clf, X_val, Y_val, cv=2, scoring='accuracy')
        print("Accuracy: %f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print(scores.mean())

    return scores.mean(),

def cxEnsemble(ind1, ind2):
    # tmp1 = ind1


    # tmp2 = ind2
    midsize = individual_size / 2
    ind1 = ind1[0:midsize - 2] + ind2[midsize:individual_size - 1]
    ind2 = ind2[0:midsize - 2] + ind1[midsize:individual_size - 1]
    return creator.Individual(ind1), creator.Individual(ind2)


def mutEnsemble(individual):
    idx_rand = random.randint(0, len(individual) - 1)
    if (individual[idx_rand] == 1):
        individual[idx_rand] = 0
    else:
        individual[idx_rand] = 1
    return individual,


dataset = arff.load(open('letter.arff'))
X_train, X_test, X_val, Y_train, Y_test, Y_val =  cria_dataset(dataset)
cria_classificadores(X_train,Y_train,100,dataset)
carrega_classificadores()

individual_size = 10
random.seed(64)
nr_generation = 5
qt_selection = 6  # (elitismo)
nr_children_generation = 10
proba_crossover = 0.7
proba_mutation = 0.2

random.seed(64)
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

toolbox.register("attr_item", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_item, individual_size)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalEnsemble)
toolbox.register("mate", cxEnsemble)
toolbox.register("mutate", mutEnsemble)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=qt_selection)

hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean, axis=0)
stats.register("std", np.std, axis=0)
stats.register("min", np.min, axis=0)
stats.register("max", np.max, axis=0)
#
algorithms.eaMuPlusLambda(pop, toolbox, qt_selection, nr_children_generation, proba_crossover, proba_mutation,
                         nr_generation,)


print(hof)
#

#print("Accuracy: {}".format(CheckAccuracy(X_test, Y_test, hof[0])))

#else:
 #   maior, media, resultados, nome=acuracia(100,X_test,y_test)
#print(random.randint(0, 1))

  #  print(nome)
  #  print (maior)
   # print (resultados)
