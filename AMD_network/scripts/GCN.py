import pandas as pd
import numpy as np
import os
import clean_graph_data
import time

import stellargraph as sg
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from stellargraph.layer import GCN, DeepGraphInfomax

import pandas as pd
from sklearn import model_selection, preprocessing
from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, callbacks
import matplotlib.pyplot as plt



def make_gcn_model():
    # function because we want to create a second one with the same parameters later
    return GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=fullbatch_generator,
        dropout=0.4,
    )


if __name__ == "__main__":
    test_number = 'test4'

    edge_path = os.path.join('.','CPDB_multiomics.txt')
    edge = clean_graph_data.loadEdge(edge_path)
    feature_path = os.path.join('..','scRNA_exp','features.csv')
    feature_data, positive_list, negative_list = clean_graph_data.load_feature_labels(feature_path, test_number)

    X, y, X_no_label, labeled_nodes, no_label_nodes, edges = clean_graph_data.get_network_info(feature_data, positive_list, negative_list, edge)

    node_index = np.append(labeled_nodes, no_label_nodes)
    total_feature_matrix = np.append(X, X_no_label, axis = 0)
    print('node_index', len(node_index))
    print('feature matrix', total_feature_matrix.shape)
    # make graph
    nodes = sg.IndexedArray(total_feature_matrix, index=node_index)
    graph = sg.StellarGraph(nodes, edges)
    # print(graph)
    # print(graph.info())
    # print(graph.nodes())

    fullbatch_generator = FullBatchNodeGenerator(graph)
    corrupted_generator = CorruptedGenerator(fullbatch_generator)
    gen = corrupted_generator.flow(graph.nodes())

    pretrained_gcn_model = make_gcn_model()

    infomax = DeepGraphInfomax(pretrained_gcn_model, corrupted_generator)
    x_in, x_out = infomax.in_out_tensors()

    dgi_model = Model(inputs=x_in, outputs=x_out)
    dgi_model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, optimizer=optimizers.Adam(learning_rate=1e-3))

    epochs = 500
    dgi_es = callbacks.EarlyStopping(monitor="loss", patience=50, restore_best_weights=True)
    dgi_history = dgi_model.fit(gen, epochs=epochs, verbose=0, callbacks=[dgi_es])
    sg.utils.plot_history(dgi_history, return_figure=True).savefig("no_label_loss.png")


    # split train, test sets
    train_classes, test_classes, train_nodes, test_nodes = model_selection.train_test_split(
        y, labeled_nodes, train_size=80, stratify=y, random_state=1)

    val_classes, test_classes, val_nodes, test_nodes = model_selection.train_test_split(
        test_classes, test_nodes, train_size=80, stratify=test_classes)   

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_classes)
    val_targets = target_encoding.transform(val_classes)
    test_targets = target_encoding.transform(test_classes)

    train_gen = fullbatch_generator.flow(train_nodes, train_targets)
    test_gen = fullbatch_generator.flow(test_nodes, test_targets)
    val_gen = fullbatch_generator.flow(val_nodes, val_targets)

    # Fine-tuning model
    pretrained_x_in, pretrained_x_out = pretrained_gcn_model.in_out_tensors()
    print(pretrained_x_in)

    pretrained_predictions = tf.keras.layers.Dense(units=train_targets.shape[1], activation="softmax")(pretrained_x_out)
    pretrained_model = Model(inputs=pretrained_x_in, outputs=pretrained_predictions)
    # pretrained_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["acc"],)
    pretrained_model.compile(optimizer=optimizers.Adam(lr=1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy()],)
    # prediction_es = callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
    prediction_es = callbacks.EarlyStopping(monitor="val_binary_accuracy", patience=50)
    pretrained_history = pretrained_model.fit(train_gen, epochs=epochs, verbose=0, validation_data=val_gen, callbacks=[prediction_es],)
    sg.utils.plot_history(pretrained_history, return_figure=True).savefig('second.png')

    pretrained_test_metrics = dict(
        zip(pretrained_model.metrics_names, pretrained_model.evaluate(test_gen))
    )
    print(pretrained_test_metrics)
 


    direct_gcn_model = make_gcn_model()
    direct_x_in, direct_x_out = direct_gcn_model.in_out_tensors()
    direct_predictions = tf.keras.layers.Dense(units=train_targets.shape[1], activation="softmax")(direct_x_out)
    direct_model = Model(inputs=direct_x_in, outputs=direct_predictions)
    # direct_model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss="categorical_crossentropy", metrics=["acc"],)
    direct_model.compile(optimizer=optimizers.Adam(lr=1e-3), loss="binary_crossentropy", metrics=[tf.keras.metrics.BinaryAccuracy()],)
    direct_history = direct_model.fit(train_gen, epochs=epochs, verbose=0, validation_data=val_gen, callbacks=[prediction_es],)
    sg.utils.plot_history(direct_history, return_figure=True).savefig('third.png')

    direct_test_metrics = dict(
        zip(direct_model.metrics_names, direct_model.evaluate(test_gen))
    )
    print(direct_test_metrics)  


    result = pd.DataFrame(
        [pretrained_test_metrics, direct_test_metrics],
        index=["with DGI pre-training", "without pre-training"],
    ).round(3)

    print(result)

    
