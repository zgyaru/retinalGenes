import os
import pandas as pd
import numpy as np
from functools import reduce

# load information of all edges
def loadEdge(edge_path, sep='\t',header=None):
    edge_df = pd.read_csv(edge_path,
                         sep=sep,  
                         header=header)
    if edge_df.shape[1] == 2:
        edge_df.columns = ["source", "target"]
    else:
        edge_df = edge_df.iloc[:,0:2]
        edge_df.columns = ["source", "target"]
    return edge_df


# load feature information and labels 
# return 3 elements: 
# 1. feature_data: in DataFrame, contains the feature vector for genes, first colume is gene name
# 2. positive_list: a list names of positive genes
# 3. negative_list: a list names of negative genes
def load_feature_labels(feature_path, test_number):
    positive_path = os.path.join('.', test_number, 'positive.txt')
    negative_path = os.path.join('.', test_number, 'negative.txt')
    try:
        feature_data = pd.read_csv(feature_path)
        positive_list = open(positive_path).read().split("\n")
        negative_list = open(negative_path).read().split("\n")
        # remove the last elements, which is an empty string
        positive_list.pop()
        negative_list.pop()
        return feature_data, positive_list, negative_list
    except:
        print('Failed to read datasets')
        sys.exit(1)


# get all network information
# return 6 elements: 
# 1. X: in np.array, feature map for genes with label (row: gene)
# 2. y: in np.array, labels for genes in X, respectively
# 3. X_no_label: in np.array, feature map for genes without label (row: gene)
# 4. labeled_genes: in np.array, gene name for X, respectively
# 5. no_label_genes: in np.array, gene name for X_no_label, respectively
# 6. edge: in DataFrame, stores edge information
def get_network_info(feature_data, positive_list, negative_list, edge):
    labeled_feature, no_labeled_feature, label= construct_feature_label(feature_data, positive_list, negative_list)
    
    source_list = edge["source"].values.tolist()
    target_list = edge["target"].values.tolist()
    labeled_genes = labeled_feature['Gene'].values.tolist()
    no_label_genes =  no_labeled_feature['Gene'].values.tolist()

    scRNA_genes = np.union1d(labeled_genes, no_label_genes)
    network_genes = np.union1d(source_list, target_list)
    common_genes = np.intersect1d(scRNA_genes, network_genes)

    # clean edges
    edge = edge[(edge['source'].isin(common_genes)) & (edge['target'].isin(common_genes))]

    # clean feature matrix
    X = labeled_feature[labeled_feature['Gene'].isin(common_genes)]
    labeled_genes = X['Gene'].values.flatten()
    X = np.array(X.iloc[:,1:].values).astype(np.float64)
    y = label[labeled_feature['Gene'].isin(common_genes).values.tolist()].astype(np.float64)

    X_no_label = no_labeled_feature[no_labeled_feature['Gene'].isin(common_genes)]
    no_label_genes = X_no_label['Gene'].values.flatten()
    X_no_label = np.array(X_no_label.iloc[:,1:].values).astype(np.float64)

    print('X:', X.shape)
    print('y:', y.shape)
    print('X_no_label:', X_no_label.shape)
    print('edge:', edge.shape)
    
    return X, y, X_no_label, labeled_genes, no_label_genes, edge

    

# construct feature matrix, X, and label vector y
# return 3 elements: 
# 1. labeled_feature: in DataFrame, contains the feature vector for genes that has labels, first colume is gene name
# 2. no_labeled_feature: in DataFrame, contains the feature vector for genes without labels, first colume is gene name
# 3. label: in np array, a list labels of 1's and 0's
def construct_feature_label(feature_data, positive_list, negative_list):
    feature_data.rename( columns={'Unnamed: 0':'Gene'}, inplace=True )

    positive_feature = feature_data[feature_data['Gene'].isin(positive_list)]
    negative_feature = feature_data[feature_data['Gene'].isin(negative_list)]
    labeled_feature = positive_feature.append(negative_feature, ignore_index=True,)
    label = np.ones(positive_feature.shape[0])
    label = np.append(label, np.zeros(negative_feature.shape[0]))

    labeled_genes = labeled_feature.iloc[:,:1].values.flatten()
    no_labeled_feature = feature_data[~feature_data['Gene'].isin(labeled_genes)]

    return labeled_feature, no_labeled_feature, label
