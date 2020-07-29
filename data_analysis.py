from pca_corre import PCA_analysis
from k_means import K_analysis
from Hierarcheal import Hier_analysis
from Umap import UMAP_analysis
from Parse import parsefile
from pickle_data import Pickle_Data
from bargraph import Hier_BarGraph
import sys

if __name__ == '__main__':

    file = sys.argv[1]
    print("Parsing started")
    data, title, folder = parsefile(file)
    Pickle_Data(data,folder)
    print("Parsing done!!!. Strating the analysis")
    PCA_analysis(data, title, folder)
    print("PCA analysis is done")
    Hier_analysis(data, title, folder)
    print("Hierarcheal clustering is done")
    Hier_BarGraph(data, title, folder)
    print("Bargraph analysis is done")
    K_analysis(data, title, folder)
    print(" K means clustering is done\n")
    #UMAP_analysis(data, title, folder)
    #print("UMAP is done")
    print("All the analysis is done. Good bye :)")
