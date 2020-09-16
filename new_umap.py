import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import umap
import pickle, os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering,MeanShift

def new_UMAP_analysis(Data, title, folder):

    n_neighbors = 4
    trial = 4

    out = "new_UMAP"
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

    Node_vars = []
    Node_vars.append(['ASCL1','NEUROD1','POU2F3','YAP1'])
    Node_vars.append(['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1'])

    TADA = ['ASCL1','NEUROD1','POU2F3','ATOH1','YAP1','REST']

    DATA = Data.loc[TADA]
    DATA = DATA.astype(float)
    Scaled_data = preprocessing.scale(DATA.T)

    Scaled_data = pd.DataFrame(Scaled_data,columns = TADA)

    for Nodes in Node_vars:

        scaled_data = np.array(Scaled_data[Nodes])

        hc_labels = []
        for h in range(2,8):
            hc_n = AgglomerativeClustering(n_clusters = h,affinity='euclidean',linkage='ward')
            y_n = hc_n.fit_predict(scaled_data)

            hc_labels.append(hc_n.labels_)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for j in range(0,trial):
            print("starting trial:",j+1)
            reducer = umap.UMAP(n_neighbors = n_neighbors, n_epochs = 1000)
            embedding = reducer.fit_transform(scaled_data)
            for h in range(2,8):
                label_color = []
                for lab in hc_labels[h-2]:
                    for i in range(0,h):
                        if lab == i:
                            label_color.append(colors[i])

                plt.scatter(embedding[:, 0],embedding[:, 1],color = label_color,s=20)
                plt.title('Nodes={}_UMAP_{}_Hier:{}_n={}'.format(str(len(Nodes)),str(j),str(h),str(n_neighbors)))
                plt.savefig(Path(folder,out,'Nodes={}_UMAP_{}_Hier:{}_n={}'.format(str(len(Nodes)),str(j),str(h),str(n_neighbors))))
                plt.close()

            for node in TADA:
                plt.scatter(embedding[:, 0],embedding[:, 1],c = Scaled_data[node],cmap = 'RdYlGn',s=15)
                plt.title('Nodes={}_UMAP_{}_Exp:{}_n={}'.format(str(len(Nodes)),str(j),node,str(n_neighbors)))
                plt.colorbar()
                plt.savefig(Path(folder,out,'Nodes={}_UMAP_{}_Exp:{}_n={}'.format(str(len(Nodes)),str(j),node,str(n_neighbors))))
                plt.close()
