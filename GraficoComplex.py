import numpy as np
import matplotlib.pyplot as plt
import re
nome_base="Wine"
data1=[]
data2=[]
#for i in range(1,21):

def gera_grafico(arq1):
    for k in arq1:
        b = k.replace(', ', ' ')
        b=b.split(' ')
        if b[0] == 'F1':
            print(b)
        else:
            print(b[1])
            data1.append(b[0])
            data2.append(b[1])
        #print (data1)
        #data1 = (data1[1:])
    print((len(data2)))
    print(data1)
    plt.scatter(data2, data1,edgecolors='green') # green bolinha
    plt.scatter(data2, data1)
    plt.xlabel("N2")
    plt.ylabel("F1")
    plt.show()
# plt.plot( x, data1, 'k:', color='orange') # linha pontilha orange
#
# plt.plot( x, data2, 'r^') # red triangulo
# plt.plot( x, data2, 'k--', color='blue')  # linha tracejada azul

#plt.axis([-10, 60, 0, 11])


#data2 = (data2[1:])
#print((data2))
#print(len(data2))
#data2 = [float(w) for w in data2]
#data1 = [float(j) for j in data1]
#print((data2))


#     c-=1cond


#x = np.array(range(len(data1)))

def main():
    nome_base='Wine'
   # arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    arq1 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_base+ '/'+nome_base+'_medias.txt', 'r')
    gera_grafico(arq1=arq1)





if __name__ == '__main__':
        main()