import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import os, pickle
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn

def Scatter2D(Data, title, folder,**kwargs):

    # def density_scatter(x, y, xlabel, ylabel, sort = True, bins = [20,20], **kwargs):

    # 	"""
    # 	Scatter plot colored by 2d histogram
    # 	"""

    # 	fig , ax = plt.subplots()
    # 	data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True)
    # 	z = interpn((0.5*(x_e[1:] + x_e[:-1]),0.5*(y_e[1:]+y_e[:-1])) ,data ,np.vstack([x,y]).T ,method = "splinef2d", bounds_error = False)

    # 	#To be sure to plot all data
    # 	z[np.where(np.isnan(z))] = 0.0

    # 	# Sort the points by density, so that the densest points are plotted last
    # 	if sort :
    # 		idx = z.argsort()
    # 		x, y, z = x[idx], y[idx], z[idx]

    # 	ax.scatter(x, y, s = 5, c=z, **kwargs)
    # 	ax.set_xlabel(xlabel)
    # 	ax.set_ylabel(ylabel)

    # 	norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # 	cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # 	cbar.ax.set_ylabel('Density')

    # 	return fig, ax

    # out = "Scatter"
    # if not os.path.exists(Path(folder,'Scatter')):
    #     os.mkdir(Path(folder,'Scatter'))

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

    if len(Remove) > 0:
        with open(Path(folder,'Nodes_not_found.txt'),'w') as f:
            f.write("The nodes of SCLC which are not found in dataset are \n\n")
            for node in Remove:
                f.write(node+'\n')

    data = Data.loc[Nodes].copy()
    data = data.astype('float64')
    scaled_data = pd.DataFrame(data = preprocessing.scale(data.T).T, index = Nodes)

    ## append list of nodes that needs to be scattered plotted

    plot_nodes = []
    plot_nodes.append(["ASCL1",'NEUROD1']) ######################################
    
    for pnodes in plot_nodes:
        X = np.array(scaled_data.loc[pnodes[0]])
        Y = np.array(scaled_data.loc[pnodes[1]])
        Z = np.array(scaled_data.loc["ASCL1"])
        clr = np.array([0 for i in range(len(X))])
        clr[np.logical_and(Y>0,Z<0)] = 1
        clr[np.logical_and(Y<0,Z>0)] = 2
        clr[np.logical_and(Y>0,Z>0)] = 3
        
    # fig, ax = density_scatter(X, Y, pnodes[0], pnodes[1], bins = [200,200])
        fig , ax = plt.subplots()
        plt.suptitle(title+": Scatter plot of {} and {}.png".format(pnodes[0],pnodes[1]))
        ax.scatter(X, Y, s = 25, c = clr)
        ax.set_xlabel(pnodes[0])
        ax.set_ylabel(pnodes[1])
        plt.savefig(str(folder)+"/Scatter"+pnodes[0]+"_"+pnodes[1]+".png",format="png")
        plt.close()
