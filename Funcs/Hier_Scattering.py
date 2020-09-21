import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram,linkage,fcluster,cophenet
import scipy.spatial.distance as ssd
from sklearn.cluster import AgglomerativeClustering,MeanShift
from sklearn.cluster import MeanShift
import sklearn.metrics as sm
import os
from pathlib import Path
import seaborn as sns

def Hier_Scat(Data, title, folder, **kwargs):

    ## Perform Agglomerative for N clusters ##
    N = 4
    Nodes = kwargs['Dims']
    if Nodes == ['']:
        NODES = []
        for node in open('sclcnetwork.ids').readlines():
            NODES.append(str(node.split('\t')[0].strip()))

        Nodes = NODES

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    data = Data.loc[Nodes]
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    hc = AgglomerativeClustering(n_clusters = N,affinity='euclidean',linkage='ward')
    y = hc.fit_predict(scaled_data)
    
    label_color = []
    for lab in hc.labels_:
        for i in range(0,N):
            if lab == i:
                label_color.append(colors[i])
    All_Nodes = []
    
    for node in open('sclcnetwork.ids').readlines():
        All_Nodes.append(str(node.split('\t')[0].strip()))
    All_Nodes = sorted(All_Nodes)
        
    Remove = []
    for node in All_Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        All_Nodes.remove(node)

    if len(Remove) > 0:
        with open(Path(folder,'Nodes_not_found.txt'),'w') as f:
            f.write("The nodes of SCLC which are not found in dataset are \n\n")
            for node in Remove:
                f.write(node+'\n')

    data = Data.loc[All_Nodes].copy()
    data = data.astype('float64')
    scaled_data = pd.DataFrame(data = preprocessing.scale(data.T).T, index = All_Nodes)

    ## append list of nodes that needs to be scattered plotted

    plot_nodes = []
    plot_nodes.append(["ASCL1",'NEUROD1']) ######################################
    sns.set(font_scale=8)
    with sns.axes_style('white'):
        with plt.rc_context({'lines.linewidth': 9}):
            fig, ax = plt.subplots(figsize=(50,50))
            for pnodes in plot_nodes:
                X = np.array(scaled_data.loc[pnodes[0]])
                Y = np.array(scaled_data.loc[pnodes[1]])
                ax.scatter(X, Y, s = 650, c = label_color)
                ax.set_xlabel(pnodes[0])
                ax.set_ylabel(pnodes[1])
                plt.axvline(x=0, color='r', linestyle='--')
                plt.axhline(y=0, color='r', linestyle='--')
                # plt.suptitle(title+": "+"HC Scat for Choosen ({}) Nodes for {} clusters".format(Nodes,str(N)))
                plt.savefig(Path(folder,
                                 title+"_"+"HCScAT_({})_Nodes_{}clus.png".format(len(Nodes),str(N))), format='png')
                plt.show()
