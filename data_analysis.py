from Parse import parse_input
from pickle_data import Pickle_Data
from pca_corre import PCA_analysis
from k_means import K_analysis
from Hierarcheal import Hier_analysis
from bargraph import Hier_BarGraph, K_BarGraph
from bool_data import Hier_Bool, K_Bool
from stat_tests import Hier_significance, K_significance
from Umap import UMAP_analysis
from scatter import Scatter2D

import sys

if __name__ == '__main__':

    file = sys.argv[1]
    print("Parsing started")
    data, title, folder, runs = parse_input(file)
    Pickle_Data(data, title, folder)
    print("Parsing done!!!. Strating the analysis")
    for run in runs:
        exec(run+"(data, title, folder)")
        print(run+" is done")
    '''PCA_analysis(data, title, folder)
    print("PCA analysis is done")
    Hier_analysis(data, title, folder)
    print("Hierarcheal clustering is done")
    Hier_BarGraph(data, title, folder)
    K_BarGraph(data, title, folder)
    print("Bargraph analysis is done")
    Hier_Bool(data, title, folder)
    K_Bool(data, title, folder)
    print("Boolean data comparision done")
    Hier_significance(data, title, folder)
    K_significance(data, title, folder)
    print("Significance tests are done")
    K_analysis(data, title, folder)
    print("K means clustering is done")
    #UMAP_analysis(data, title, folder)
    #print("UMAP is done")'''
    print("\nAll the analysis is done. Good bye :)")
