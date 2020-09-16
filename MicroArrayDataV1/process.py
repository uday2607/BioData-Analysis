import numpy as np
import pandas as pd
import os

for file in ['SC4L_norm.csv','SC4_norm.csv','SC4T_norm.csv','SC16_norm.csv','SC39_norm.csv','SC49_norm.csv']:

    data = pd.read_csv(file,index_col = 0)

    nodes = ['ASCL1','ATOH1','NEUROD1','POU2F3','YAP1']

    df = data.loc[nodes]
    cols = data.columns[(df == 0).all()]

    data = data.drop(cols, axis = 1)

    data.to_csv(file)
