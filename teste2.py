import numpy

from matplotlib import pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
nome_base="Wine"


def gera_grafico():
    pdf_pages = PdfPages('histograms.pdf')

    # Generate the pages
    nb_plots = 14

    nb_plots_per_page = 5
    nb_pages = int(numpy.ceil(nb_plots / float(nb_plots_per_page)))
    grid_size = (nb_plots_per_page, 2)

    data1 = []
    data2 = []
    data5 = []
    data6 = []
    nome_base = 'Wine'
    j = 0
    r=0
    for i in range(1, 15):
        data3 = []
        data4 = []
        data1 = []
        data2 = []
        data5 = []
        data6 = []
        # arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
        arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/'+nome_base+'/' + nome_base + str(i) + '/'+nome_base+'_medias.txt', 'r')
        arq1 = open('/home/marcos/Documents/Tese/ComplexidadeBags/'+nome_base+'/' + nome_base + str(i)+'/' + nome_base + '_medias.txt', 'r')
        arq3 = open('/home/marcos/Documents/Tese/ComplexidadeAG/'+nome_base+'/' + nome_base + '1/' + nome_base + '_medias.txt', 'r')
        for k in arq1:
            b = k.replace(', ', ' ')
            b=b.split(' ')
            if b[0] == 'F1':
                None
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
                None
            else:
                #print(b[1])
                data3.append(b[0])
                data4.append(b[1])
                data3 = [float(w) for w in data3]
                data4 = [float(w) for w in data4]
        for k in arq3:
            b = k.replace(', ', ' ')
            b = b.split(' ')
            if b[0] == 'F1':
                None
            else:
                #print(b[1])
                data5.append(b[0])
                data6.append(b[1])
                data5 = [float(w) for w in data5]
                data6 = [float(w) for w in data6]
        #print(((data5)))

        if (r == 2):
            r = 0
        if j % nb_plots_per_page == 0:
            fig = plot.figure(figsize=(8, 15), dpi=50)
        print(j)
        # Plot stuffs !
        plot.subplot2grid(grid_size, (j % nb_plots_per_page, r))
        plot.title(nome_base+str(i))
        bag=plot.scatter(x=data1,y=data2,alpha=0.6)
        dist=plot.scatter(x=data3, y=data4,alpha=0.6)
        ag=plot.scatter(x=data5, y=data6, alpha=0.6)
        plot.xlabel("N2")
        plot.ylabel("F1")
        plot.legend([bag, dist, ag], ["Bag", "Dist", "AG"])
        # Close the page if needed
        if (j + 1) % nb_plots_per_page == 0 or (j + 1) == nb_plots:
            plot.tight_layout()
            pdf_pages.savefig(fig)
        j=j+1
        r=r+1

    pdf_pages.close()

def main():
    nome_base='Wine'
    #
    # for i in range(1,21):
    # #arq1 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    #     arq2 = open('/home/marcos/Documents/Tese/ComplexidadeDist/' + nome_base + str(i) + '/Wine_medias.txt', 'r')
    #     arq1 = open('/home/marcos/Documents/Tese/ComplexidadeBags/' + nome_base + '/' + nome_base + '_medias.txt', 'r')
    #     arq3 = open('/home/marcos/Documents/Tese/ComplexidadeAG/' + nome_base + '/' + nome_base + '_medias.txt', 'r')

    gera_grafico()





if __name__ == '__main__':
        main()