from Parse import *
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
    IN, runs, Dim, Color = parse_input(file)
    params = dict(Dims=Dim,Colors=Color)
    for In in IN:
        print("Parsing started")
        data, title, folder = parse_infile(In)
        # Pickle_Data(data, title, folder) Run it only if you need it
        print("Parsing done!!!. Starting the analysis for {}".format(In[:-3]))
        for run in runs:
            exec(run+"(data, title, folder, **params)")
            print(run+" is done")
        print("Analysis is done for {}".format(In[:-3]))
    print("\nAll the analysis is done. Good bye :)")
