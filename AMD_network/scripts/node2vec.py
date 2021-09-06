import pandas as pd
import numpy as np

import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.data import BiasedRandomWalk
from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import Node2VecLinkGenerator, Node2VecNodeGenerator, GraphSAGENodeGenerator,Attri2VecLinkGenerator, Attri2VecNodeGenerator
from stellargraph.layer import Node2Vec, link_classification, GraphSAGE,Attri2Vec


from sklearn import model_selection

import tensorflow as tf
from tensorflow import keras

import argparse, os, sys


def loadNetwork(path,sep='\t',header=None):
    net_df = pd.read_csv(path,
                         sep=sep,  
                         header=header)
    if net_df.shape[1] == 2:
        net_df.columns = ["source", "target"]
    else:
        net_df = net_df.iloc[:,0:2]
        net_df.columns = ["source", "target"]
    network = StellarGraph(edges = net_df)
    return network

def write_hyper_params(args, input_file, file_name):
    """Write hyper parameters to disk.
    Writes a set of hyper parameters of the model to disk.
    See `load_hyper_params` for information on how to load
    the hyper parameters.
    Parameters:
    ----------
    args:               The parameters to save as dictionary
    input_file:         The input data hdf5 container. Only
                        present for legacy reasons
    file_name:          The file name to write the data to.
                        Should be 'hyper_params.txt' in order
                        for the load function to work properly.
    """
    with open(file_name, 'w') as f:
        for arg in args:
            f.write('{}\t{}\n'.format(arg, args[arg]))
        f.write('{}\n'.format(input_file))
    print("Hyper-Parameters saved to {}".format(file_name))

def embedding_node2vec(network, 
                       n_rw=1000, ## Total number of random walks per root node
                       length=50, ## Maximum length of each random walk
                       p=0.5,  # defines probability, 1/p, of returning to source node
                       q=2.0,  # defines probability, 1/q, for moving to a node away from the source node
                       batch_size=100,
                       embs=128, # embedding size
                       lr=.001, # learning rate
                       epochs=1000,n_cores=8):
    walker = BiasedRandomWalk(network, 
                              n=n_rw, ## Total number of random walks per root node
                              length=length, ## Maximum length of each random walk
                              p=p,  # defines probability, 1/p, of returning to source node
                              #q=q,  # defines probability, 1/q, for moving to a node away from the source node
                             )
    unsupervised_samples = UnsupervisedSampler(network, nodes=list(network.nodes()), walker=walker)
    generator = Node2VecLinkGenerator(network, batch_size)
    node2vec = Node2Vec(embs, generator=generator)
    x_inp, x_out = node2vec.in_out_tensors()
    prediction = link_classification(output_dim=1, 
                                     output_act="sigmoid", 
                                     edge_embedding_method="dot")(x_out)
    model = keras.Model(inputs=x_inp, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss=keras.losses.binary_crossentropy,
        metrics=[keras.metrics.binary_accuracy]
    )
    callback = keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    history = model.fit(generator.flow(unsupervised_samples),
                        epochs=epochs,
                        verbose=1,
                        use_multiprocessing=False,
                        workers=n_cores,
                        shuffle=True,
                        callbacks=[callback],
                       )
    x_inp_src = x_inp[0]
    x_out_src = x_out[0]
    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    #node_gen = Node2VecNodeGenerator(network, batch_size).flow(list(network.nodes()))
    #node_embeddings = embedding_model.predict(node_gen, workers=n_cores, verbose=1)
    return(embedding_model)
def parse_args():
    parser = argparse.ArgumentParser(
        description='Train node2vec and save embedding to file')
    parser.add_argument('-net', '--net_path', 
                        help='File path of network',
                        dest='net',
                        type=str,
                        required=True
                        )
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=.001,
                        type=float
                        )
    parser.add_argument('-embs', '--embedding_size',
                        help='Size of embedding by node2vec.',
                        dest='embs',
                        default=128,
                        type=int
                       )
    parser.add_argument('-n_rw', '--number_randomwalker', 
                        help='Total number of random walks per root node',
                        dest='n_rw',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-length', '--length', 
                        help='Maximum length of each random walk',
                        dest='length',
                        default=50,
                        type=int
                        )
    parser.add_argument('-p', '--p', 
                        help='defines probability, 1/p, of returning to source node',
                        dest='p',
                        default=.5,
                        type=float
                        )
    parser.add_argument('-q', '--q', 
                        help='defines probability, 1/q, for moving to a node away from the source node',
                        dest='q',
                        default=2.0,
                        type=float
                        )
    parser.add_argument('-bs', '--batch_size', 
                        help='defines probability, 1/q, for moving to a node away from the source node',
                        dest='batch_size',
                        default=100,
                        type=int
                        )
    parser.add_argument('-nc', '--number_cores', 
                        help='Number of cores for parallel',
                        dest='n_cores',
                        default=8,
                        type=int
                        )
    parser.add_argument('-out', '--out_folder', 
                        help='Folder of output',
                        dest='out',
                        type=str,
                        required=True
                        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    network = loadNetwork(args.net)
    model = embedding_node2vec(network, 
                                   n_rw=args.n_rw,
                                   length=args.length,
                                   p=args.p,
                                   q=args.q,
                                   batch_size=args.batch_size,
                                   embs=args.embs, 
                                   lr=args.lr, 
                                   epochs=args.epochs,
                                   n_cores=args.n_cores)
    #print(embedding.shape)
    #np.save(os.path.join(args.out,'embedding.npy'),embedding)
    model.save(os.path.join(args.out,'embedding.h5'))
    param_path = os.path.join(args.out,'hyper_params.txt')
    write_hyper_params(args_dict, args.net, param_path)





