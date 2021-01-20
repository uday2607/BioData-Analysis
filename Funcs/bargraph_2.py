import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
import os
from pathlib import Path
import seaborn as sns

def Hier_BarGraph_2(Data, title, folder, **kwargs):

    Nodes = kwargs['Dims']

    NODES = []
    for node in open('sclcnetwork.ids').readlines():
        NODES.append(str(node.split('\t')[0].strip()))
    upper_square = ['ASCL1', 'ATF2', 'CBFA2T2', 'CEBPD','ELF3','ETS2','FOXA1','FOXA2','FLI1','INSM1','KDM5B','LEF1','MYB','OVOL2','PAX5','PBX1','POU3F2','SOX11', 'SOX2', 'TCF12','TCF3','TCF4','NEUROD1']
    lower_square = [i for i in NODES if i not in upper_square]

    top = upper_square + lower_square
    NODES = top + ["POU2F3","YAP1"]

    if Nodes == ['']:
        Nodes = NODES

    colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    Remove = []
    for node in NODES:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        try:
            Nodes.remove(node)
        except ValueError:
            pass
        try:
            upper_square.remove(node)
        except ValueError:
            pass
        try:
            lower_square.remove(node)
        except ValueError:
            pass

    NODES = upper_square + lower_square + ["POU2F3","YAP1"]

    if not os.path.exists(Path(folder,"bargraph")):
        os.mkdir(Path(folder,"bargraph"))
    data_n = Data.loc[Nodes]
    data_n = data_n.astype(float)
    scaled_data_n = preprocessing.scale(data_n.T)

    data_N = Data.loc[NODES]
    data_N = data_N.astype(float)
    scaled_data_N = preprocessing.scale(data_N.T)

    columns = np.array(data_n.columns)
    Data_n = pd.DataFrame(scaled_data_n.T, index = Nodes, columns = columns).T

    hc_labels_n = []

    patches = []
    for i in range(len(Nodes)):
        patches.append(mpatches.Patch(color=colours[i], label=Nodes[i]))

    for h in range(2,7):

        hc_n = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
        y_n = hc_n.fit_predict(scaled_data_n)
        hc_labels_n.append(np.array(hc_n.labels_))

    fig, ax = plt.subplots()
    sns.set_context("paper", font_scale=1.5)

    for h in range(2,7):

        ticks = []
        tick_labels = []
        x = np.arange(len(NODES))*3

        for i in range(h):
            #ticks.append(x[int(len(NODES)/2)])
            #tick_labels.append(np.sum(hc_labels_n[h-2] == i))
            y = np.array(np.mean(scaled_data_N[hc_labels_n[h-2] == i], axis = 0))
            y_error = np.array(np.std(scaled_data_N[hc_labels_n[h-2] == i], axis = 0))/((scaled_data_N[hc_labels_n[h-2] == i]).shape[0])**(0.5)
            barlist = ax.bar(x, y, yerr = y_error, width = 2.5, label = str(i))
            #fill the colors
            for n in range(0,len(upper_square)):
                barlist[n].set_color('r')
            for n in range(1+len(upper_square), len(upper_square)+len(lower_square)):
                barlist[n].set_color('b')
            barlist[len(upper_square)].set_color('orange')
            barlist[-2].set_color('y')
            barlist[-1].set_color('g')

            ax.set_xticks(x)
            ax.set_xticklabels(range(len(NODES)))
            plt.setp(ax.get_xticklabels(), fontsize='8')
            ax.set_xlabel('Nodes')
            ax.set_ylabel("Expression value")
            ax.set_title(title + ": Exp_of_All_Genes_hier={}_cluster={}".format(h, i))
            plt.savefig(Path(folder,"bargraph",title+"_Exp_of_All_genes_hier_{}_cluster_{}.png".format(h, i)), format='png', bbox_inches="tight")
            plt.cla()
            #x = x + 0.6

        '''
        H_data = Data_n.copy()
        H_data['Hier_labels'] = hc_labels_n[h-2]
        H_data = H_data.sort_values(by=['Hier_labels'])
        H_data.to_excel(Path(folder,"bargraph",title+"_Cellline_classify_{}_nodes_Hier_{}.xlsx".format(len(Nodes),h)))

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Cluster cardinality')
        ax.set_ylabel("Expression value")
        ax.set_title(title + ": Exp_of_{}_nodes_hier={}".format(Nodes, h))
        ax.legend(handles = patches)
        plt.savefig(Path(folder,"bargraph",title+"_Exp_of_{}_nodes_hier={}.png".format(len(Nodes), h)), format='png')
        plt.cla()
        '''

def K_BarGraph_2(Data, title, folder, **kwargs):


    Nodes = kwargs['Dims']
    NODES = []

    for node in open('sclcnetwork.ids').readlines():
        NODES.append(str(node.split('\t')[0].strip()))
    upper_square = ['ASCL1', 'ATF2', 'CBFA2T2', 'CEBPD','ELF3','ETS2','FOXA1','FOXA2','FLI1','INSM1','KDM5B','LEF1','MYB','OVOL2','PAX5','PBX1','POU3F2','SOX11', 'SOX2', 'TCF12','TCF3','TCF4','NEUROD1']
    lower_square = [i for i in NODES if i not in upper_square]

    top = upper_square + lower_square
    NODES = top + ["POU2F3","YAP1"]

    if Nodes == ['']:
        Nodes = NODES

    colours = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

    Remove = []
    for node in NODES:
        if not node in list(Data.index):
            Remove.append(node)

    for node in Remove:
        try:
            Nodes.remove(node)
        except ValueError:
            pass
        try:
            upper_square.remove(node)
        except ValueError:
            pass
        try:
            lower_square.remove(node)
        except ValueError:
            pass

    NODES = upper_square + lower_square + ["POU2F3","YAP1"]

    if not os.path.exists(Path(folder,"bargraph")):
        os.mkdir(Path(folder,"bargraph"))

    data_n = Data.loc[Nodes]
    data_n = data_n.astype(float)
    scaled_data_n = preprocessing.scale(data_n.T)

    data_N = Data.loc[NODES]
    data_N = data_N.astype(float)
    scaled_data_N = preprocessing.scale(data_N.T)

    columns = np.array(data_n.columns)
    Data_n = pd.DataFrame(scaled_data_n.T, index = Nodes, columns = columns).T

    k_labels_n = []

    patches = []
    for i in range(len(Nodes)):
        patches.append(mpatches.Patch(color=colours[i], label=Nodes[i]))

    for k in range(2,7):

        k_n = KMeans(init="random",n_clusters=k,n_init=20,max_iter=300)
        y_n = k_n.fit_predict(scaled_data_n)
        k_labels_n.append(np.array(k_n.labels_))

    fig, ax = plt.subplots()
    sns.set_context("paper", font_scale=1.5)

    for k in range(2,7):

        ticks = []
        tick_labels = []
        x = np.arange(len(NODES))*3

        for i in range(k):
            #ticks.append(x[int(len(NODES)/2)])
            #tick_labels.append(np.sum(k_labels_n[k-2] == i))
            y = np.array(np.mean(scaled_data_N[k_labels_n[k-2] == i], axis = 0))
            y_error = np.array(np.std(scaled_data_N[k_labels_n[k-2] == i], axis = 0))/(scaled_data_N[k_labels_n[k-2] == i].shape[0])**(0.5)
            barlist = ax.bar(x, y, yerr = y_error, width = 2.5, label = str(i))
            #fill the colors
            for n in range(0,len(upper_square)):
                barlist[n].set_color('r')
            for n in range(1+len(upper_square), len(upper_square)+len(lower_square)):
                barlist[n].set_color('b')
            barlist[len(upper_square)].set_color('orange')
            barlist[-2].set_color('y')
            barlist[-1].set_color('g')

            ax.set_xticks(x)
            ax.set_xticklabels(range(len(NODES)))
            plt.setp(ax.get_xticklabels(), fontsize='8')
            ax.set_xlabel('Nodes')
            ax.set_ylabel("Expression value")
            ax.set_title(title + ": Exp_of_All_Genes_KMeans={}_cluster={}".format(k, i))
            plt.savefig(Path(folder,"bargraph",title+"_Exp_of_All_genes_KMeans_{}_cluster_{}.png".format(k, i)), format='png', bbox_inches="tight")
            plt.cla()
            #x = x + 0.6

        '''
        H_data = Data_n.copy()
        H_data['k_labels'] = k_labels_n[k-2]
        H_data = H_data.sort_values(by=['k_labels'])
        H_data.to_excel(Path(folder,"bargraph",title+"_Cellline_classify_{}_nodes_Hier_{}.xlsx".format(len(Nodes),k)))

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Cluster cardinality')
        ax.set_ylabel("Expression value")
        ax.set_title(title + ": Exp_of_{}_nodes_hier={}".format(Nodes, k))
        ax.legend(handles = patches)
        plt.savefig(Path(folder,"bargraph",title+"_Exp_of_{}_nodes_hier={}.png".format(len(Nodes), k)), format='png')
        plt.cla()
        '''
