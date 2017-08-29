import arff, os
nome_base = 'Wine'
os.system("mkdir /home/marcos/Documents/Tese/ComplexidadeBags/" + nome_base)
enderecoin = " -i /home/marcos/Documents/Tese/Baggs/" + nome_base + "/"

    #os.system("mkdir /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/" + nome_base)
enderecoout = " -o /home/marcos/Documents/Tese/ComplexidadeBags/" + nome_base + "/complexidadeBag" + nome_base

dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"
for i in range(0,100):


    distancias='IndividuoWine'+str(i)
    os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(i)+" -d -F 1 -N 2")