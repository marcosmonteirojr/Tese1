nome_base="Wine"
arq2 = open('/home/marcos/Documents/Tese/Complexidade/'+nome_base+'/'+nome_base+'_resumo.txt', 'w')
def main():
    for j in range(44):
        arq1 = open('/home/marcos/Documents/Tese/Complexidade/'+nome_base+'/complexidade'+nome_base+str(j)+'.txt', 'r')
        print(arq1.readline())
        print(arq1.readline())
        print(arq1.readline())
        print(arq1.readline())
        print(arq1.readline())
        if(j==0):
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
        print(arq1.readline())
        arq2.write(arq1.readline())
        arq1.close()
    arq2.close()
    arq3 = open('/home/marcos/Documents/Tese/Complexidade/'+nome_base+'/'+nome_base+'_resumo.txt', 'r')
    arq4 = open('/home/marcos/Documents/Tese/Complexidade/'+nome_base+'/'+nome_base+'_resumo2.txt', 'w')
    a=arq3.read()
    b=len(a)
    for i in range(b):
        a=a.replace('       ', ' ')

    arq4.write(a)
    arq3.close()
    arq4.close()
if __name__ == '__main__':
    main()