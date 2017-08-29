import arff, os
from scipy.spatial import distance
#nome_base="Wine2"

# dataset = arff.load(open('/home/marcos/Documents/Tese/Bases/Teste/2/TesteWine2.arff'))
# dataset2 = arff.load(open('/home/marcos/Documents/Tese/Bases/Validacao/2/ValidaWine2.arff'))
# dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"
# enderecoin = " -i /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/"+nome_base+"/"
# os.system("mkdir /home/marcos/Documents/Tese/Complexidade/"+nome_base)
# enderecoout = " -o /home/marcos/Documents/Tese/ComplexidadeDist/"+nome_base+"/complexidade"+nome_base


def euclidean4(vector1, vector2):
    """

    @:param vector1: dados a serem medidos
    @:param vector2: dados a serem medidos
    @:return: distancia
    """
    dist = distance.euclidean(vector1, vector2)
    return dist

def cria_arff(info, data, nome):
    """
    cria um arquivo arff no E:
    @:param info: descricao da base/ relacao
    @:param data: dados da base
    @:param nome: do arquivo a ser gerado
    :return:
    """
    obj = {
        'description': info['description'],
        'relation': info['relation'],
        'attributes': info['attributes'],
        'data': data['data'],

    }
    arq1=arff.dumps(obj)
    arq=open('/home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/'+nome_base+'/'+nome+'.arff','w')
    arq.write(arq1)
    arq.close()

def distancia_maxima(data, data2, pos, dist):
    """compara data[pos] com todos data2 1 a um retorna a distancia escolhida
        @:param data: base extraida do arrf
        @:param data2: base a ser medida (arff)
        @:param pos: posicao a ser comparada
        @:param dist: posicao do vetor que fica a distancia

    """
    distancias = []#retornoda distancia
    vetor = (data['data'][pos])
    vetor = (vetor[:-1])
    print(vetor)
    for j in data2['data']:  # percorre a base valida
        vetor2 = (j[:-1])  # elimina a ultima coluna
        c = euclidean4(vetor, vetor2)  # calcula as distancias
       # print(c)
        distancias.append(c)  # salva em um array as distancias
        distancias.sort() #ordena
    #print(distancias)
    return distancias[dist] #retorna o valor da distancia na posisao desejada

def main():


    for i in range(1,21):
        b=i
        a=str(i)
        i=b
        global nome_base,k,nome_b#nome_base serve para abrir os arquivos, nome_b para dar nome ao novos arquivos
        nome_base = 'Wine'+a
        nome_b = 'Wine'
        k = 0
       # print(i)
        dataset = arff.load(open('/home/marcos/Documents/Tese/Bases/Teste/'+a+'/TesteWine'+a+'.arff'))
        dataset2 = arff.load(open('/home/marcos/Documents/Tese/Bases/Validacao/'+a+'/ValidaWine'+a+'.arff'))
        dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"
        enderecoin = " -i /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/" + nome_base + "/"
        os.system("mkdir /home/marcos/Documents/Tese/ComplexidadeDist/" + nome_base)
        os.system("mkdir /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/" + nome_base)
        enderecoout = " -o /home/marcos/Documents/Tese/ComplexidadeDist/" + nome_base + "/complexidade" + nome_b

        #valor = (distancia_maxima(dataset, dataset2, 0, 30))
        dados=dict()
        dados['data'] = list()
        for i in range(len(dataset['data'])):#range tamnho da base
            dados = dict()
            dados['data'] = list()
            valor=(distancia_maxima(dataset, dataset2, i, 30))
            for j in dataset2['data']:
                vetor2 = (j[:-1])
                c=euclidean4(dataset['data'][i][:-1], vetor2)
                if c<=valor:
                    dados['data'].append(j)

            distancias='Distancias'+nome_b+str(k)
            cria_arff(dataset, dados, distancias)
            os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -d -F 1 -N 2")
            k=k+1

if __name__ == '__main__':
    main()