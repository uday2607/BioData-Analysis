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

def Hier_analysis(Data, title, folder):

    out = "Hier"
    if not os.path.exists(Path(folder,out)):
        os.mkdir(Path(folder,out))

    NODES = []
    for node in open('sclcnetwork.ids').readlines():
        NODES.append(str(node.split('\t')[0].strip()))

    ## Dendogram for 33 nodes ##

    Remove = []
    for node in NODES:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        NODES.remove(node)

    data = Data.loc[NODES]
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    dend1 = dendrogram(linkage(scaled_data,method='ward'),leaf_rotation=90,leaf_font_size=8)
    plt.suptitle(title+": "+"Dendogram for SCLC {} nodes".format(len(NODES)))
    plt.savefig(Path(folder,out,title+": "+"Dendogram for SCLC {} nodes".format(len(NODES))))
    plt.clf()

    ## Dendogram for 5 nodes ##

    Nodes = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1']

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    data = Data.loc[Nodes]
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    dend1 = dendrogram(linkage(scaled_data,method='ward'),leaf_rotation=90,leaf_font_size=8)
    plt.suptitle(title+": "+"Dendogram for Choosen ({}) Nodes".format(len(Nodes)))
    plt.savefig(Path(folder,out,title+": "+"Dendogram for Choosen ({}) Nodes".format(len(Nodes))))
    plt.clf()
