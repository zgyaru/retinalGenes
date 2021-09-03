import os
import pandas as pd
import numpy as np

# load feature matrix and labels
def load_datasets(test_number):
    positive_path = os.path.join('.', test_number, 'positive.txt')
    negative_path = os.path.join('.', test_number, 'negative.txt')
    try:
        feature_data = pd.read_csv('features.csv')
        positive_list = open(positive_path).read().split("\n")
        negative_list = open(negative_path).read().split("\n")
        # remove the last elements, which is an empty string
        positive_list.pop()
        negative_list.pop()
        return feature_data, positive_list, negative_list
    except:
        print('Failed to read datasets')
        sys.exit(1)


def construct_training_set(feature_data, positive_list, negative_list):
    n, d = feature_data.shape
    d = d - 1
    all_genes = feature_data.get(feature_data.columns[0]).values.tolist()

    X = np.array([])
    i = 0
    for gene_name in positive_list:
        try: 
            idx = all_genes.index(gene_name)
        except:
            # print(gene_name, 'is not found in feature matrix')
            continue
        feature_vector = feature_data[idx: idx+1].values.tolist()
        feature_vector = np.delete(feature_vector, 0)
        X = np.append(X, feature_vector)
        i += 1

    X = X.reshape((i, d))
    y = np.ones(i)

    j = 0
    for gene_name in negative_list:
        try: 
            idx = all_genes.index(gene_name)
        except:
            # print(gene_name, 'is not found in feature matrix')
            continue
        feature_vector = feature_data[idx: idx+1].values.tolist()
        feature_vector = np.delete(feature_vector, 0).reshape((1,d))
        X = np.append(X, feature_vector, axis=0)
        j += 1

    y = np.append(y, np.zeros(j))
    return X.astype(np.float64), y.astype(np.float64)
