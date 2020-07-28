import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import umap
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering,MeanShift
import os
from pathlib import Path

def UMAP_analysis(Data, title, folder):

    trial = 3

    out = "UMAP"
    if not os.path.exists(Path(folder,out)):
        os.mkdir(Path(folder,out))

    NODES = []
    for node in open('sclcnetwork.ids').readlines():
        NODES.append(str(node.split('\t')[0].strip()))
    NODES = sorted(NODES)

    Remove = []
    for node in NODES:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        NODES.remove(node)

    Nodes = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1']
    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    ## Labels of 33 and 5 nodes ##

    data_n = Data.loc[Nodes]
    data_n = data_n.astype(float)
    scaled_data_n = preprocessing.scale(data_n.T)

    data_N = Data.loc[NODES]
    data_N = data_N.astype(float)
    scaled_data_N = preprocessing.scale(data_N.T)

    hc_labels_n = []
    hc_labels_N = []
    k_labels_n = []
    k_labels_N = []

    for h in range(2,10):
        hc_n = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        hc_N = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        kmeans_n = KMeans(init="random",n_clusters=h,n_init=20,max_iter=300)
        kmeans_N = KMeans(init="random",n_clusters=h,n_init=20,max_iter=300)

        y_n = hc_n.fit_predict(scaled_data_n)
        y_N = hc_N.fit_predict(scaled_data_N)
        x_n = kmeans_n.fit(scaled_data_n)
        x_N = kmeans_N.fit(scaled_data_N)

        hc_labels_n.append(hc_n.labels_)
        hc_labels_N.append(hc_N.labels_)
        k_labels_n.append(kmeans_n.labels_)
        k_labels_N.append(kmeans_N.labels_)

    ## Umap for 33 nodes with n and N labels ##

    # n labels #
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for h in range(2,10):
        label_color = []
        for lab in hc_labels_n[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):

            if not os.path.exists(Path(folder,out,str(n))):
                os.mkdir(Path(folder,out,str(n)))

            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_N)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name + " for SCLC {} nodes".format(len(NODES)))
            plt.clf()

        label_color = []
        for lab in k_labels_n[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_N)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name+ " for SCLC {} nodes".format(len(NODES)))
            plt.clf()

    # N labels #
    for h in range(2,10):
        label_color = []
        for lab in hc_labels_N[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):

            if not os.path.exists(Path(folder,out,str(n))):
                os.mkdir(Path(folder,out,str(n)))

            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_N)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name + " for SCLC {} nodes".format(len(NODES)))
            plt.clf()

        label_color = []
        for lab in k_labels_N[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_N)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name+ " for SCLC {} nodes".format(len(NODES)))
            plt.clf()

    ## Umap for 5 nodes ##

    # n labels #
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for h in range(2,10):
        label_color = []
        for lab in hc_labels_n[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):

            if not os.path.exists(Path(folder,out,str(n))):
                os.mkdir(Path(folder,out,str(n)))

            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_n)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name + " for Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()

        label_color = []
        for lab in k_labels_n[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_n)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n) + " (labels of Choosen ({}) nodes)".format(len(Nodes))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name+ " Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()

    # N labels #
    for h in range(2,10):
        label_color = []
        for lab in hc_labels_N[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):

            if not os.path.exists(Path(folder,out,str(n))):
                os.mkdir(Path(folder,out,str(n)))

            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_n)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name + " Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()

        label_color = []
        for lab in k_labels_N[h-2]:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,18, 3):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data_n)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n) + " (labels of {} SCLC nodes)".format(len(NODES))
            fig.savefig(folder+"/"+out+'/'+str(n)+'/'+title+": "+fig_name+ " Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()
