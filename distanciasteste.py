import arff, os
from scipy.spatial import distance
nome_base="Wine"

dataset = arff.load(open('/home/marcos/Documents/Tese/Distancias/TesteWine1.arff'))
dataset2 = arff.load(open('/home/marcos/Documents/Tese/Distancias/ValidaWine1.arff'))

def euclidean4(vector1, vector2):
    """

    @:param vector1: dados a serem medidos
    @:param vector2: dados a serem medidos
    @:return: distancia
    """
    dist = distance.euclidean(vector1, vector2)
    return dist
