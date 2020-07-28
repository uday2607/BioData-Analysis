import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import umap
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering,MeanShift

def UMAP_analysis(Data, title, folder):

    trial = 3

    ## Umap for 33 nodes ##

    Nodes = []
    for node in open('sclcnetwork.ids').readlines():
        Nodes.append(str(node.split('\t')[0].strip()))
    Nodes = sorted(Nodes)

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
    for h in range(2,9):
        hc = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y = hc.fit_predict(scaled_data)
        label_color = []
        for lab in hc.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name + " for SCLC {} nodes".format(len(Nodes)))
            plt.clf()

        kmeans = KMeans(init="random",n_clusters=h,n_init=20,max_iter=300)
        alpha = kmeans.fit(scaled_data)
        label_color = []
        for lab in kmeans.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name+ " for SCLC {} nodes".format(len(Nodes)))
            plt.clf()

    ## Umap for 5 nodes ##

    Nodes = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1']

    data = Data.loc[Nodes]
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for h in range(2,9):
        hc = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y = hc.fit_predict(scaled_data)
        label_color = []
        for lab in hc.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name + " for Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()

        kmeans = KMeans(init="random",n_clusters=h,n_init=20,max_iter=300)
        alpha = kmeans.fit(scaled_data)
        label_color = []
        for lab in kmeans.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name+ " for Choosen ({}) nodes".format(len(Nodes)))
            plt.clf()

    ## Umap for all the nodes ##

    data = Data
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for h in range(2,9):
        hc = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y = hc.fit_predict(scaled_data)
        label_color = []
        for lab in hc.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " Hier Clusters = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)
            fig_name = "Hier " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name + " for All nodes")
            plt.clf()

        kmeans = KMeans(init="random",n_clusters=h,n_init=20,max_iter=300)
        alpha = kmeans.fit(scaled_data)
        label_color = []
        for lab in kmeans.labels_:
            for i in range(0,h):
                if lab == i:
                    label_color.append(colors[i])
        for n in range(2,11):
            fig, axs = plt.subplots(trial, trial)
            fig_tit = "n neighbor:" + str(n) + " K = " + str(h)
            fig.suptitle(fig_tit)
            for i in range(0,trial):
                for j in range(0,trial):
                    reducer = umap.UMAP(n_neighbors = n, n_epochs = 1000)
                    embedding = reducer.fit_transform(scaled_data)
                    axs[i,j].scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=2)

            fig_name = "K means " + str(h) + " umap " + str(n)
            fig.savefig(folder+"/"+title+": "+fig_name+ " for All nodes")
            plt.clf()
