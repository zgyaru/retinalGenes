import pandas as pd
import numpy as np
import os
import clean_graph_data
import time

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML
import matplotlib.pyplot as plt



if __name__ == "__main__":
    test_number = 'test3'

    edge_path = os.path.join('.','CPDB_multiomics.txt')
    edge = clean_graph_data.loadEdge(edge_path)
    feature_path = os.path.join('..','scRNA_exp','features.csv')
    feature_data, positive_list, negative_list = clean_graph_data.load_feature_labels(feature_path, test_number)


    X, y, X_no_label, labeled_nodes, no_label_nodes, edges = clean_graph_data.get_network_info(feature_data, positive_list, negative_list, edge)

    node_index = np.append(labeled_nodes, no_label_nodes)
    total_feature_matrix = np.append(X, X_no_label, axis = 0)

    print('node_index', len(node_index))
    print('feature matrix', total_feature_matrix.shape)

    nodes = sg.IndexedArray(total_feature_matrix, index=node_index)
    graph = sg.StellarGraph(nodes, edges)

    print(graph)
    print(graph.info)



    # train_subjects, test_subjects = model_selection.train_test_split(node_subjects, train_size=140, test_size=None, stratify=node_subjects)
    # val_subjects, test_subjects = model_selection.train_test_split(test_subjects, train_size=500, test_size=None, stratify=test_subjects)

    
