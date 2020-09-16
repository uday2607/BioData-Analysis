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
    # Pickle_Data(data, title, folder) Run it only if you need it
    print("Parsing done!!!. Strating the analysis")
    for run in runs:
        exec(run+"(data, title, folder)")
        print(run+" is done")
    print("\nAll the analysis is done. Good bye :)")
