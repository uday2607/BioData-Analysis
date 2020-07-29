import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

def Hier_BarGraph(Data, title, folder):

    if not os.path.exists(Path(folder,'Bargraph')):
        os.mkdir(Path(folder,'Bargraph'))

    Nodes = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1']
    colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    data_n = Data.loc[Nodes]
    data_n = data_n.astype(float)
    scaled_data_n = preprocessing.scale(data_n.T)

    columns = np.array(data_n.columns)

    hc_labels_n = []

    for h in range(2,7):

        hc_n = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y_n = hc_n.fit_predict(scaled_data_n)
        hc_labels_n.append(np.array(hc_n.labels_))

    for h in range(2,7):

        y = []
        y_error = []
        x = np.arange(len(Nodes))*0.1

        for i in range(h):
            y.append(np.array(data[columns[hc_labels_n[h] == i]].mean(axis = 0)))
            yer.append(np.array(data[columns[hc_labels_n[h] == i]].std(axis = 0)))
            plt.bar(x, y, yerr = y_error, color = colours[:len(Nodes)], labels = Nodes, width = 0.25)

            x = x + 0.5

        plt.xlabel('Clusters')
        plt.ylabel("Expression value")
        plt.suptitle(title + ": Expression levels of {} nodes for hier : {}".format(len(Nodes), h))
        plt.legend()
        plt.savefig(Path(folder,'Bargraph',"Expression levels of {} nodes for hier : {}".format(len(Nodes), h)))
