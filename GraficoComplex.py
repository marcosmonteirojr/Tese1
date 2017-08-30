import numpy as np
import matplotlib.pyplot as plt

nome_base="Wine"


def gera_grafico(arq1, arq2 ,title):
    data1 = []
    data2 = []
    data3 = []
    data4 = []

    for k in arq1:
        b = k.replace(', ', ' ')
        b=b.split(' ')
        if b[0] == 'F1':
            print(b)
        else:
            #print(b[1])
            data1.append(b[0])
            data2.append(b[1])
            data1= [float(w) for w in data1]
            data2= [float(w) for w in data2]
        #print (data1)
        #data1 = (data1[1:])
    for k in arq2:
        b = k.replace(', ', ' ')
        b = b.split(' ')
        if b[0] == 'F1':
            print(b)
        else:
            #print(b[1])
            data3.append(b[0])
            data4.append(b[1])
            data3 = [float(w) for w in data3]
            data4 = [float(w) for w in data4]
    print(((data3)))
    fig = plt.figure()
    plt.title(title)
    bag=plt.scatter(data1, data2, c='red',marker='v', ) # green bolinha
    dist=plt.scatter(data3, data4, c='blue')
    plt.xlabel("N2")
    plt.ylabel("F1")
    plt.legend([bag, dist], ["Bag", "Dist"])
    #py.plot_mpl(fig, filename="mpl-complex-scatter")


    fig.savefig('/home/marcos/Documents/Tese/Graficos/'+title, dpi=fig.dpi)

    #plt.show()


def main():
    nome_base='Wine'

    for i in range(1,21):
    #arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
        arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
        arq1 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_base + '/' + nome_base + '_medias.txt', 'r')

        gera_grafico(arq1=arq1, arq2=arq2, title=nome_base+str(i))





if __name__ == '__main__':
        main()