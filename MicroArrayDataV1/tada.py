import numpy as np
import pandas as pd
import os

data = pd.read_csv("bulk_rna_1.txt",delimiter="\t")
print(data)
data.drop('Entrez_Gene_Id', axis=1, inplace=True)
print(data)
data = data.astype(float)

NODES = []
for node in open('sclcnetwork.ids').readlines():
    NODES.append(str(node.split('\t')[0].strip().strip('\n')))
Nodes = ['POU2F3','ATOH1','YAP1']

for i in Nodes:
    NODES.append(i)

data = data.loc[NODES]
data = data.groupby(data.index).mean()
data.to_csv('bulk_rna_1.csv')
