import subprocess
import csv
nome_base="Wine"#nome da base
proc = subprocess.Popen(["ls /home/marcos/Documents/Tese/ComplexidadeDist/"+nome_base+"/*.txt | wc -l"], stdout=subprocess.PIPE, shell=True)
(out, err)=proc.communicate()
out=int(out)#retorna o numeros de arquivos na pasta (lembrar que poode ter os resumos, dae da erro)

media=[]#medias de F1
median2=[]#medias de F2

arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo.txt', 'w')#arquivo que salva os resumos



def main():
    rownum = 0
    f1 = 0
    n2 = 0
    cont = 0
    nclas = 3  # numero de classes
    con2 = 0
    d1 = 3  # numero de classes para dividir
    d2 = 2  # numero de classes para dividir caso uma seja inf
    for j in range(out):
        arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/complexidade'+nome_base+str(j)+'.txt', 'r')# abre os aquivos um por um da comlexidade
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        if(j==0):#filtra o heder e cria um cmo ' F1 e N2"
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())

        arq1.readline()
        arq2.write(arq1.readline())#salva valores de F1 e N2, cuidar com a qntidade de classes
        arq2.write(arq1.readline())
        arq2.write(arq1.readline())
        arq2.write("\n")
        arq1.close()
    arq2.close()
    arq3 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo.txt', 'r')#abre o resumo que foi gerado acima
    arq4 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo2.csv', 'w')#salva um novo arquivo formatado em csv
    #print(len(arq3.readlines()))

    for i in (arq3):
            i = i.replace('                               ', ',')  # 9
            i = i.replace('          ', ',')  #10
            i = i.replace('         ', ',')#9
            i = i.replace('        ', ',')  #8
            i = i.replace('       ', ',')  # 7
            i = i.replace('      ', ',')#6
            i = i.replace('     ', ',')#5
            i = i.replace('   ',',')#3


            print (i)
            arq4.write(i)
    arq3.close()
    arq4.close()
    arq5=open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_medias.txt','w')#salva um arquivo com as médias, varia de acordo com o numero de classes
    with open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo2.csv') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
        # Save header row.

            if rownum == 0:
               None
            else:
                colnum =0
                #percorre F1 e calcula a média, percorre N2 e calcula a média levando em conta se possui infinito ou nao. lembrar de trocar o cont de acordo com o numero de classes
                for col in  row:

                    if(cont<=nclas and colnum==1):
                        f1 += (float(col))
                        cont += 1
                       # print (col,f1)
                        #print(rownum+1)
                        #print ((("\n")))
                        #print (media)
                        if cont ==3:
                            f1 += (float(col))
                            f1=f1/d1
                            cont = 0
                           # print ("\n media")
                            #print (f1)
                            #print ("\n cont")
                            media.append(f1)
                            f1=0
                            #print (cont)
                    if (con2 <= nclas and colnum == 2):
                        if(col=='inf'):
                            flag=1
                            con2 += 1
                        else:
                            n2 += (float(col))
                            con2 += 1
                           # print (col, n2)
                            if con2 == 3:
                                n2 += (float(col))
                                if (flag == 1):
                                    n2 = n2 / d2
                                    con2 = 0
                                else:
                                    n2 = n2/d1
                                #print ("\n media")
                               # print (n2)
                               # print ("\n con2")
                                median2.append(n2)
                                n2 = 0
                                #print (con2)
                    colnum += 1

                #print(media)

            rownum += 1


    #print ((median2))
    test=map(list,zip(media,median2))#compacta e transfora em uma matriz o resultado das médias
    #print (test)
    arq5.write('F1     N2\n')
    te = [str(i) for i in test]#converte para string
    for i in te:
        print (i[1:-1])
        #test = [float(i) for w in data2]
        arq5.write(i[1:-1]+"\n")#salva em um arquivo
   # print(test)
    arq5.close()

if __name__ == '__main__':
    main()