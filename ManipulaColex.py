import subprocess
nome_base="Wine"
proc =subprocess.Popen(["ls /home/marcos/Documents/Tese/ComplexidadeDist/"+nome_base+"/*.txt | wc -l"], stdout=subprocess.PIPE, shell=True)
(out, err)=proc.communicate()
out=int(out)
out=out-2
arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo.txt', 'w')



def main():
    for j in range(out):
        arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/complexidade'+nome_base+str(j)+'.txt', 'r')
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        (arq1.readline())
        if(j==0):
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())
            arq2.write(arq1.readline())

        arq1.readline()
        arq2.write(arq1.readline())
        arq2.write(arq1.readline())
        arq2.write(arq1.readline())
        arq2.write("\n")
        arq1.close()
    arq2.close()
    arq3 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo.txt', 'r')
    arq4 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/'+nome_base+'_resumo2.txt', 'w')
    #print(len(arq3.readlines()))

    for i in (arq3):
        if 'inf' in i:
            i=i.replace('      ', ' ')
            i=i.replace('         ', ' ')
            i = i.replace('   ', ' ')
        else:
            i = i.replace('       ', ' ')
            i = i.replace('        ', ' ')
        print (i)
    #arq4.write(a)
    arq3.close()
    arq4.close()
if __name__ == '__main__':
    main()