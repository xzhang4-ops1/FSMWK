import numpy as np
import pandas as pd
from MWKpp import MWKmeans

def FSMWKpp(csv_path, k, top_n=1):
    """
    :param csv_path: absolute path of csv file
    :param k: number of clusters
    :param top_n: number of features to select
    :return: list of feature indices
    """
    df = pd.read_csv(csv_path, header=None)
    data = df.values
    max_p = 31
    weights_all = []
    for p in range(11, max_p):
        mwkpp = MWKmeans(k, p / 10, 'mwkmeans++', replications=25)
        mwkpp.fit(data)
        for weights_k in mwkpp.weights:
            weights_all.append(weights_k)
    weights_all = np.array(weights_all)
    median_weights = np.median(weights_all, axis=0)
    sorted_indices = np.argsort(median_weights)[::-1]
    selected_features = sorted_indices[:top_n]
    return selected_features

# example of usage
# selected_features = FSMWKpp('./datasets/Data1000x10_3k_5NF_1.csv',3,10)
# print(selected_features)

