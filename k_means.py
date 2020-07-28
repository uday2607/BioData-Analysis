import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import os
from pathlib import Path

def K_analysis(Data,title,folder):

    out = "K means"
    if not os.path.exists(Path(folder,out)):
        os.mkdir(Path(folder,out))

    ## For 33 nodes of SCLC ##
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

    clus_num = len(list(data.columns))
    num_trial = 10

    repl_avg = []
    freq = {}
    distortions = []
    vars = []

    for j in range(2,clus_num):
        mul_trial = []
        sil_avg = []
        dists = []

        for i in range(num_trial):
            kmeans = KMeans(init="random",n_clusters=j,n_init=20,max_iter=300)
            alpha = kmeans.fit(scaled_data)
            identified_clusters = kmeans.fit_predict(scaled_data)
            sil_avg.append(silhouette_score(scaled_data,identified_clusters))
            dists.append(kmeans.inertia_)

        distortions.append(np.mean(dists))
        vars.append(np.var(dists))
        repl_avg.append(sil_avg)

    # Silhoutte score #

    plt.boxplot(repl_avg[0:8],labels=np.array(range(2,10)),showmeans=True,notch=1,sym='')
    plt.ylabel("Silhoutte score")
    plt.xlabel("K values")
    plt.suptitle(title+": "+"Silhoutte score for SCLC {} nodes".format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+"Silhoutte score for SCLC {} nodes".format(len(Nodes)))
    plt.clf()

    # Distortions of K means #

    K = list(range(2,clus_num))
    diff = -1*np.ediff1d(distortions)
    plt.plot(K,distortions,'bx-')
    plt.errorbar(K,distortions,yerr=vars)
    plt.ylabel("Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Elbow plot for SCLC {} nodes'.format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+'Elbow plot for SCLC {} nodes.png'.format(len(Nodes)))
    plt.clf()

    # Difference between Distortions #

    K = list(range(2,clus_num-1))
    plt.plot(K,diff,'rx-')
    plt.ylabel("Difference in Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Differences in Distortions for SCLC {} nodes'.format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+'Differences in Distortions for SCLC {} nodes'.format(len(Nodes)))
    plt.clf()

    ## For all the nodes ##
    data = Data
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    clus_num = len(list(data.columns))
    num_trial = 10

    repl_avg = []
    freq = {}
    distortions = []
    vars = []

    for j in range(2,clus_num):
        mul_trial = []
        sil_avg = []
        dists = []

        for i in range(num_trial):
            kmeans = KMeans(init="random",n_clusters=j,n_init=20,max_iter=300)
            alpha = kmeans.fit(scaled_data)
            identified_clusters = kmeans.fit_predict(scaled_data)
            sil_avg.append(silhouette_score(scaled_data,identified_clusters))
            dists.append(kmeans.inertia_)

        distortions.append(np.mean(dists))
        vars.append(np.var(dists))
        repl_avg.append(sil_avg)

    # Silhoutte score #

    plt.boxplot(repl_avg[0:8],labels=np.array(range(2,10)),showmeans=True,notch=1,sym='')
    plt.ylabel("Silhoutte score")
    plt.xlabel("K values")
    plt.suptitle(title+": "+"Silhoutte score for All nodes")
    plt.savefig(folder+"/"+out+"/"+title+": "+"Silhoutte score for All nodes")
    plt.clf()

    # Distortions of K means #

    K = list(range(2,clus_num))
    diff = -1*np.ediff1d(distortions)
    plt.plot(K,distortions,'bx-')
    plt.errorbar(K,distortions,yerr=vars)
    plt.ylabel("Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Elbow plot for All nodes')
    plt.savefig(folder+"/"+out+"/"+title+": "+'Elbow plot for All nodes.png')
    plt.clf()

    # Difference between Distortions #

    K = list(range(2,clus_num-1))
    plt.plot(K,diff,'rx-')
    plt.ylabel("Difference in Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Differences in Distortions for all nodes')
    plt.savefig(folder+"/"+out+"/"+title+": "+'Differences in Distortions for all nodes.png')
    plt.clf()

    ## For 5 nodes ##
    Nodes = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1']

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    if len(Remove) > 0:
        with open(folder+'/'+'Nodes_not_found.txt','a') as f:
            f.write("The nodes of Choosen nodes which are not found in dataset are \n\n")
            for node in Remove:
                f.write(node+'\n')

    data = Data.loc[Nodes]
    data = data.astype(float)
    scaled_data = preprocessing.scale(data.T)

    clus_num = len(list(data.columns))
    num_trial = 10

    repl_avg = []
    freq = {}
    distortions = []
    vars = []

    for j in range(2,clus_num):
        mul_trial = []
        sil_avg = []
        dists = []

        for i in range(num_trial):
            kmeans = KMeans(init="random",n_clusters=j,n_init=20,max_iter=300)
            alpha = kmeans.fit(scaled_data)
            identified_clusters = kmeans.fit_predict(scaled_data)
            sil_avg.append(silhouette_score(scaled_data,identified_clusters))
            dists.append(kmeans.inertia_)

        distortions.append(np.mean(dists))
        vars.append(np.var(dists))
        repl_avg.append(sil_avg)

    # Silhoutte score #

    plt.boxplot(repl_avg[0:8],labels=np.array(range(2,10)),showmeans=True,notch=1,sym='')
    plt.ylabel("Silhoutte score")
    plt.xlabel("K values")
    plt.suptitle(title+": "+"Silhoutte score for Choosen ({}) nodes".format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+"Silhoutte score for Choosen ({}) nodes".format(len(Nodes)))
    plt.clf()

    # Distortions of K means #

    K = list(range(2,clus_num))
    diff = -1*np.ediff1d(distortions)
    plt.plot(K,distortions,'bx-')
    plt.errorbar(K,distortions,yerr=vars)
    plt.ylabel("Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Elbow plot for Choosen ({}) nodes'.format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+'Elbow plot for Choosen ({}) nodes.png'.format(len(Nodes)))
    plt.clf()

    # Difference between Distortions #

    K = list(range(2,clus_num-1))
    plt.plot(K,diff,'rx-')
    plt.ylabel("Difference in Distortions")
    plt.xlabel("K values")
    plt.suptitle(title+": "+'Differences in Distortions for Choosen ({}) nodes'.format(len(Nodes)))
    plt.savefig(folder+"/"+out+"/"+title+": "+'Differences in Distortions for Choosen ({}) nodes.png'.format(len(Nodes)))
    plt.clf()
