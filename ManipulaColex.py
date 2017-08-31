import subprocess
import csv

nome_bs = "Wine1"#nome da base
#proc = subprocess.Popen(["ls /home/marcos/Documents/Tese/ComplexidadeDist/"+nome_bs+"1/*.txt | wc -l"], stdout=subprocess.PIPE, shell=True)
proc = subprocess.Popen(["ls /home/marcos/Documents/Tese/ComplexidadeBags/"+nome_bs+"/*.txt | wc -l"], stdout=subprocess.PIPE, shell=True)
(out, err)=proc.communicate()
out=int(out)#retorna o numeros de arquivos na pasta (lembrar que poode ter os resumos, dae da erro)
out=out-2

def cria_resumo(nome_b,i,out,nome_base,tipo):

    """


    :param nome_b: string nome nao variavel para
    :param i: incremento para percorrer as pastas
    :param out: numero de arquivos, extraido do supprocess
    :param nome_base:
    :param tipo: se e do Andre ou minha, se 0 minha se 1 andre
    :return: arquivos de resumo na pasta destino
    """
    #print(nome_b)
    if tipo == 0:
        arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/' + nome_base + '_resumo.txt', 'w')  # arquivo que salva os resumos
    if tipo == 1:
        arq2 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_base+str(i)+'/' + nome_base + '_resumo.txt',
                'w')  # arquivo que salva os resumos
    for j in range(out):
        if tipo == 0:
            arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_b + '/complexidade' + nome_base + str(j) + '.txt', 'r')  # abre os aquivos um por um da comlexidade
        if tipo == 1:
            arq1 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_b +'/complexidadeBag' + nome_base + str(j) + '.txt','r')  # abre os aquivos um por um da comlexidade
        # print(nome_b)
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        if (j==0):  # filtra o heder e cria um cmo ' F1 e N2"
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
        # if (j == 1 and tipo==1):  # filtra o heder e cria um cmo ' F1 e N2"
        #     arq2.write(arq1.readline())
        #     arq2.write(arq1.readline())
        #     arq2.write(arq1.readline())
        #     arq2.write(arq1.readline())

        #arq1.readline()
        arq1.readline()
        arq2.write(arq1.readline())  # salva valores de F1 e N2, cuidar com a qntidade de classes
        arq2.write(arq1.readline())
        arq2.write(arq1.readline())
        arq2.write("\n")
        arq1.close()
    arq2.close()


def cria_csv(nome_b,nome_base,tipo):
    if(tipo==0):
        arq3 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_b + '/' + nome_base + '_resumo.txt',
                'r')  # abre o resumo que foi gerado acima
        arq4 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_b + '/' + nome_base + '_resumo2.csv',
                'w')  # salva um novo arquivo formatado em csv
    # print(len(arq3.readlines()))
    if(tipo==1):
        arq3 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_b + '/' + nome_base + '_resumo.txt',
                    'r')  # abre o resumo que foi gerado acima
        arq4 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_b + '/' + nome_base + '_resumo2.csv',
                    'w')
    if(tipo==2):
        arq3=open('/home/marcos/Documents/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_resumo.txt',
                    'r')
        arq4 = open('/home/marcos/Documents/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_resumo2.csv',
                    'w')

    for i in (arq3):
        i = i.replace('                               ', ',')  # 9
        i = i.replace('          ', ',')  # 10
        i = i.replace('         ', ',')  # 9
        i = i.replace('        ', ',')  # 8
        i = i.replace('       ', ',')  # 7
        i = i.replace('      ', ',')  # 6
        i = i.replace('     ', ',')  # 5
        i = i.replace('   ', ',')  # 3

        # print (i)
        arq4.write(i)
    arq3.close()
    arq4.close()


def calcula_media(nome_base, nome_b, cont, con2, nclas, d1,tipo):
    rownum = 0
    media = []  # medias de F1
    median2 = []  # medias de F2
    f1=n2=0
    if tipo==0:
        None
        # arq5 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_b + '/' + nome_base + '_medias.txt',
        #         'w')  # salva um arquivo com as medias, varia de acordo com o numero de classes
        # with open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_b + '/' + nome_base + '_resumo2.csv') as csvfile:
        #     reader = csv.reader(csvfile)
    if tipo==1:
        arq5 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_b + '/' + nome_base + '_medias.txt',
                    'w')  # salva um arquivo com as medias, varia de acordo com o numero de classes
        with open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_b + '/' + nome_base + '_resumo2.csv') as csvfile:
            reader = csv.reader(csvfile)
    # if tipo == 2:
    #     arq = open('/home/marcos/Documents/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_medias.txt',
    #                 'w')  # salva um arquivo com as medias, varia de acordo com o numero de classes
    #     with open('/home/marcos/Documents/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_resumo2.csv') as csvfile:
    #         reade = csv.reader(csvfile)

            for row in reader:
                #cont = 0
                flag = 0
                #con2 = 0
                if rownum == 0:
                    None
                else:
                    colnum = 0
                    # percorre F1 e calcula a media, percorre N2 e calcula a media levando em conta se possui infinito ou nao. lembrar de trocar o cont de acordo com o numero de classes
                    for col in row:
                        if (cont <= nclas and colnum == 1):
                            f1 += (float(col))
                            cont += 1
                            #print(col)

                            if cont == 3:
                               # f1 += (float(col))
                                f1 = f1 / d1
                                print(col)
                               # print("\n")
                               # print(f1)
                               # print("\n")
                                media.append(f1)
                                cont=0
                                f1 = 0
                        if (con2 <= nclas and colnum == 2):
                            #print(col)
                            # if (col == 'inf'):
                            #     flag = 1
                            #     con2 += 1

                           # else:
                            n2 += (float(col))
                            con2 += 1
                                #print(con2)
                            if con2 == 3:
                                #n2 += (float(col))
                                    # if (flag == 1):
                                    #     n2 = n2 / d2
                                    #     median2.append(n2)
                                    #     n2 = 0
                                    #     con2 = 0
                                    #else:
                                       # print(col)
                               # print("\n")
                                #print(n2)
                               # print("\n")
                                n2 = n2 / d1
                                median2.append(n2)
                                #print(median2)
                                n2 = 0
                                con2=0
                        colnum += 1
                rownum += 1

          #  print ((len(median2)))
            test = map(list, zip(media, median2))  # compacta e transfora em uma matriz o resultado das medias
           # print (test)
            arq5.write('F1     N2\n')
            te = [str(i) for i in test]  # converte para string
            for i in te:

                print( i.split(',', 1))
            for i in te:
                #i=i.split(' ')
                #print (i[1:-1])
            # test = [float(i) for w in data2]
                arq5.write(i[1:-1] + "\n")  # salva em um arquivo
            # print(test)
        arq5.close()
        csvfile.close()


def main():
    for i in range(1,15):
        nome_b='Wine'+str(i)
        nome_base='Wine'
        cria_resumo(nome_b, i, out, nome_base,tipo=1)
        cria_csv(nome_b,nome_base,tipo=1)
        calcula_media(nome_base,nome_b, cont=0,con2=0,nclas=3,d1=3,tipo=1)
    # global nome_b,nome_base, media,median2#medias de F2
    # media = []  # medias de F1
    # median2 = []  # medias de F2
    # for i in range(1,21):
    #     media = []  # medias de F1
    #     median2 = []  # medias de F2
    #
    #     nome_b=nome_base+str(i)
    #     #print(nome_b)
    #     rownum = 0
    #     f1 = 0
    #     n2 = 0
    #     cont = 0
    #     nclas = 3  # numero de classes
    #     con2 = 0
    #     d1 = 3  # numero de classes para dividir
    #     d2 = 2  # numero de classes para dividir caso uma seja inf




if __name__ == '__main__':
    main()