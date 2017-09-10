import arff, os
nome_base = 'Wine'
# os.system("mkdir /home/marcos/Documents/Tese/ComplexidadeBags/" + nome_base)
# enderecoin = " -i /home/marcos/Documents/Tese/Baggs/" + nome_base + "/"
#
#     #os.system("mkdir /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/" + nome_base)
# enderecoout = " -o /home/marcos/Documents/Tese/ComplexidadeBags/" + nome_base + "/complexidadeBag" + nome_base

dcol = "/home/marcos/Documents/Tese/dcol/DCoL-v1.1/Source/dcol"
for j in range(15,21):
    os.system("mkdir /home/marcos/Documents/Tese/ComplexidadeBags/"+nome_base+"/" + nome_base+str(j))
    enderecoin = " -i /home/marcos/Documents/Tese/Baggs/" "Bag"+ nome_base + "/"+nome_base+str(j)
    # os.system("mkdir /home/marcos/Documents/Tese/Distancias/ResultadosDistanciasValidaTeste/" + nome_base)
    enderecoout = " -o /home/marcos/Documents/Tese/ComplexidadeBags/"+nome_base +"/"+ nome_base+str(j) + "/complexidade" + nome_base
    for i in range(1,101):
        k=i-1
        distancias='/IndividuoWine'+str(i)
        os.system(dcol+enderecoin+distancias+".arff"+enderecoout+str(k)+" -d -F 1 -N 2")