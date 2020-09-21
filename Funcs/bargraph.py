import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import os
from pathlib import Path

def Hier_BarGraph(Data, title, folder, **kwargs):

    Nodes = kwargs['Dims']
    if Nodes == ['']:
        NODES = []
        for node in open('sclcnetwork.ids').readlines():
            NODES.append(str(node.split('\t')[0].strip()))

        Nodes = NODES

    colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)
    if not os.path.exists(Path(folder,"bargraph")):
        os.mkdir(Path(folder,"bargraph"))
    data_n = Data.loc[Nodes]
    data_n = data_n.astype(float)
    scaled_data_n = preprocessing.scale(data_n.T)

    columns = np.array(data_n.columns)

    hc_labels_n = []

    patches = []
    for i in range(len(Nodes)):
        patches.append(mpatches.Patch(color=colours[i], label=Nodes[i]))

    for h in range(2,7):

        hc_n = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y_n = hc_n.fit_predict(scaled_data_n)
        hc_labels_n.append(np.array(hc_n.labels_))

    fig, ax = plt.subplots()

    for h in range(2,7):

        ticks = []
        tick_labels = []
        x = np.arange(len(Nodes))*0.1

        for i in range(h):
            ticks.append(x[int(len(Nodes)/2)])
            tick_labels.append(np.sum(hc_labels_n[h-2] == i))
            y = np.array(np.mean(scaled_data_n[hc_labels_n[h-2] == i], axis = 0))
            y_error = np.array(np.std(scaled_data_n[hc_labels_n[h-2] == i], axis = 0))
            ax.bar(x, y, yerr = y_error, color = colours[:len(Nodes)], width = 0.1, label = str(i))

            x = x + 0.6

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Cluster cardinality')
        ax.set_ylabel("Expression value")
        ax.set_title(title + ": Exp_of_{}_nodes_hier={}".format(Nodes, h))
        ax.legend(handles = patches)
        plt.savefig(Path(folder,"bargraph",title+"_Exp_of_{}_nodes_hier={}.png".format(len(Nodes), h)), format='png')
        plt.show()

def K_BarGraph(Data, title, folder, **kwargs):

    Nodes = kwargs['Dims']
    if Nodes == ['']:
        NODES = []
        for node in open('sclcnetwork.ids').readlines():
            NODES.append(str(node.split('\t')[0].strip()))

        Nodes = NODES
    if not os.path.exists(Path(folder,"bargraph")):
        os.mkdir(Path(folder,"bargraph"))
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

    k_labels_n = []

    patches = []
    for i in range(len(Nodes)):
        patches.append(mpatches.Patch(color=colours[i], label=Nodes[i]))

    for k in range(2,7):

        kmeans_n = KMeans(init="random",n_clusters=k,n_init=20,max_iter=300)
        x_n = kmeans_n.fit(scaled_data_n)
        k_labels_n.append(kmeans_n.labels_)

    fig, ax = plt.subplots()

    for k in range(2,7):

        ticks = []
        tick_labels = []
        x = np.arange(len(Nodes))*0.1

        for i in range(k):

            ticks.append(x[int(len(Nodes)/2)])
            tick_labels.append(np.sum(k_labels_n[k-2] == i))
            y = np.array(np.mean(scaled_data_n[k_labels_n[k-2] == i], axis = 0))
            y_error = np.array(np.std(scaled_data_n[k_labels_n[k-2] == i], axis = 0))
            ax.bar(x, y, yerr = y_error, color = colours[:len(Nodes)], width = 0.1, label = str(i))

            x = x + 0.6

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Cluster cardinality')
        ax.set_ylabel("Expression value")
        ax.set_title(title + ": Exp_of_{}_nodes_K={}".format(Nodes, k))
        ax.legend(handles = patches)
        plt.savefig(Path(folder,"bargraph",title+"_Exp_of_{}_nodes_K={}.png".format(len(Nodes), k)), format='png')
        plt.show()
