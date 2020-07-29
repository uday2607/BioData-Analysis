import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
from pathlib import Path
import pickle

def PCA_analysis(Data,title,folder):

    out = "PCA"
    if not os.path.exists(Path(folder,out)):
        os.mkdir(Path(folder,out))

    ## PCA of 33 nodes of SCLC ##
    Nodes = []
    for node in open('sclcnetwork.ids').readlines():
        Nodes.append(str(node.split('\t')[0].strip()))
    Nodes = sorted(Nodes)

    tops = [10, 13]

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    if len(Remove) > 0:
        with open(folder+'/'+'Nodes_not_found.txt','w') as f:
            f.write("The nodes of SCLC which are not found in dataset are \n\n")
            for node in Remove:
                f.write(node+'\n')

    data = Data.loc[Nodes]
    data = data.astype('float64')
    scaled_data = preprocessing.scale(data.T)

    # Pearson correlation #
    for top in tops:

        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        loading_scores = pd.Series(pca.components_[0], index=Nodes)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_1 = sorted(sorted_loading_scores[0:top].index.values)

        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
        plt.imshow(data.T[top_1].corr(), cmap='seismic', interpolation='nearest')
        plt.colorbar()
        plt.clim(-1, 1)
        ax1.set_xticks(np.arange(len(top_1)))
        ax1.set_yticks(np.arange(len(top_1)))
        ax1.set_xticklabels(top_1,rotation=90, fontsize=10)
        ax1.set_yticklabels(top_1,fontsize=10)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(len(top_1)):
            for j in range(len(top_1)):
                data_corr,data_p = stats.pearsonr(data.T[top_1[i]], data.T[top_1[j]])
                if data_p < 0.001:
                    text = ax1.text(j, i, '***', ha="center", va="center", color="w", fontsize = 7)
                elif data_p < 0.005:
                    text = ax1.text(j, i, '**', ha="center", va="center", color="w", fontsize = 7)
                elif data_p < 0.05:
                    text = ax1.text(j, i, '*', ha="center", va="center", color="w", fontsize = 7)
        plt.suptitle(title+": "+"Pearson correlation of top {} genes".format(top))
        plt.savefig(folder+"/"+out+"/"+title+": "+"Pearson correlation of top {} genes of SCLC {} nodes".format(top, len(Nodes)))
        plt.clf()

    # Scree Plot #

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(folder+"/"+out+"/"+title+": "+"Scree Plot of SCLC {} Nodes".format(len(Nodes)))
    plt.clf()

    ## PCA of All nodes ##

    data = Data
    data = data.astype('float64')
    scaled_data = preprocessing.scale(data.T)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    loading_scores = pd.Series(pca.components_[0], index=list(data.index))
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_1 = sorted(sorted_loading_scores[0:top].index.values)

    fig = plt.figure(figsize=(8,8))
    ax1 = fig.add_subplot(111)
    plt.imshow(data.T[top_1].corr(), cmap='seismic', interpolation='nearest')
    plt.colorbar()
    plt.clim(-1, 1)
    ax1.set_xticks(np.arange(len(top_1)))
    ax1.set_yticks(np.arange(len(top_1)))
    ax1.set_xticklabels(top_1,rotation=90, fontsize=10)
    ax1.set_yticklabels(top_1,fontsize=10)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(top_1)):
        for j in range(len(top_1)):
            data_corr,data_p = stats.pearsonr(data.T[top_1[i]], data.T[top_1[j]])
            if data_p < 0.001:
                text = ax1.text(j, i, '***', ha="center", va="center", color="w", fontsize = 7)
            elif data_p < 0.005:
                text = ax1.text(j, i, '**', ha="center", va="center", color="w", fontsize = 7)
            elif data_p < 0.05:
                text = ax1.text(j, i, '*', ha="center", va="center", color="w", fontsize = 7)
    plt.suptitle(title+": "+"Pearson correlation of top {} genes".format(top))
    plt.savefig(folder+"/"+out+"/"+title+": "+"Pearson correlation of top {} genes of All nodes".format(top))
    plt.clf()

    # Scree Plot #

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.savefig(folder+"/"+out+"/"+title+": "+"Scree Plot of All Nodes")
    plt.clf()

    ## PCA of top 10 and 13 nodes of CCLE ##

    top_nodes = []
    top_nodes.append(sorted(['ASCL1','CBFA2T2','FOXA1','INSM1','OVOL2','PAX5','PBX1','REST','SMAD3','TEAD4']))
    top_nodes.append(sorted(['ASCL1','CBFA2T2','ETS2','FOXA1','FOXA2','INSM1','OVOL2','PAX5','PBX1','REST','SMAD3','TEAD4','MITF']))

    Nodes = []
    for node in open('sclcnetwork.ids').readlines():
        Nodes.append(str(node.split('\t')[0].strip()))
    Nodes = sorted(Nodes)

    tops = [10, 13]

    Remove = []
    for node in Nodes:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        Nodes.remove(node)

    data = Data.loc[Nodes]
    data = data.astype('float64')
    scaled_data = preprocessing.scale(data.T)

    # Pearson correlation #
    for num, top in enumerate(tops):

        pca = PCA()
        pca.fit(scaled_data)
        pca_data = pca.transform(scaled_data)
        loading_scores = pd.Series(pca.components_[0], index=Nodes)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_1 = top_nodes[num]

        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(111)
        plt.imshow(data.T[top_1].corr(), cmap='seismic', interpolation='nearest')
        plt.colorbar()
        plt.clim(-1, 1)
        ax1.set_xticks(np.arange(len(top_1)))
        ax1.set_yticks(np.arange(len(top_1)))
        ax1.set_xticklabels(top_1,rotation=90, fontsize=10)
        ax1.set_yticklabels(top_1,fontsize=10)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        for i in range(len(top_1)):
            for j in range(len(top_1)):
                data_corr,data_p = stats.pearsonr(data.T[top_1[i]], data.T[top_1[j]])
                if data_p < 0.001:
                    text = ax1.text(j, i, '***', ha="center", va="center", color="w", fontsize = 7)
                elif data_p < 0.005:
                    text = ax1.text(j, i, '**', ha="center", va="center", color="w", fontsize = 7)
                elif data_p < 0.05:
                    text = ax1.text(j, i, '*', ha="center", va="center", color="w", fontsize = 7)
        plt.suptitle(title+": "+"Pearson correlation of top {} genes of CCLE".format(top))
        plt.savefig(folder+"/"+out+"/"+title+": "+"Pearson correlation of top {} genes of CCLE".format(top))
        plt.clf()
