import numpy as np
import matplotlib.pyplot as plt
import re
nome_base="Wine"
arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_medias.txt', 'r')

#print(arq2.read())

data1=[]
data2=[]

for k in arq1:

    b = k.split(',')
    print(b)
    data1.append(b[0])
    data2.append(b[1])
# print (data1)
# data1 = (data1[1:])
# print(data1)
# data2 = (data2[1:])
data2 = [float(w) for w in data2]
data1 = [float(j) for j in data1]



#     c-=1cond


#x = np.array(range(len(data1)))

plt.scatter(data2, data1, s=50 ) # green bolinha
plt.xlabel("N2")
plt.ylabel("F1")
# plt.plot( x, data1, 'k:', color='orange') # linha pontilha orange
#
# plt.plot( x, data2, 'r^') # red triangulo
# plt.plot( x, data2, 'k--', color='blue')  # linha tracejada azul

#plt.axis([-10, 60, 0, 11])

plt.show()