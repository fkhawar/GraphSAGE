from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os
import subprocess
import tensorflow as tf
import graph_tool

# import networkx as nx
# from networkx.readwrite import json_graph
# version_info = list(map(int, nx.__version__.split('.')))
# major = version_info[0]
# minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


SHUF_UTIL = (sys.platform.startswith('linux') and 'shuf') or \
            (sys.platform.startswith('darwin') and 'gshuf') or ''


flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)

    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map


def load_data_from_graph(graph_file, features_file, labels_file, map_file, walks_file=None):
    g = graph_tool.load_graph(graph_file)

    # out_degrees = g.get_out_degrees(np.arange(0, g.num_vertices()))
    # in_degrees = g.get_in_degrees(np.arange(0, g.num_vertices()))
    #
    # nodes = (set(np.arange(0, g.num_vertices())[out_degrees > 0]) &
    #          set(np.arange(0, g.num_vertices())[in_degrees > 0]))
    #
    # id_map = {n: idx for idx, n in enumerate(nodes)}
    with open(map_file, 'r') as vertices:
        id_map = {int(v): idx for idx, v in enumerate(vertices)}

    print("IdMap loaded", len(id_map))

    class random_walks(object):
        def __init__(self, filename):
            self.filename = filename
            self._len = None

            self._open()

        def _open(self):
            print("Shuffling...")
            # p = subprocess.Popen([SHUF_UTIL, '-o', self.filename + '.shuffle', self.filename])
            # p.wait()
            self.f = open(self.filename + '.shuffle', 'r')

        def __len__(self):
            if self._len is not None:
                return self._len

            p = subprocess.Popen(['wc', '-l', self.filename],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result, err = p.communicate()
            if p.returncode != 0:
                raise IOError(err)

            self._len = int(result.strip().split()[0])
            return self._len

        def __iter__(self):
            return self

        def next(self):
            l = next(self.f)
            src, dst = l.strip().split('\t')
            return int(src), int(dst)

    nodes = []
    labels = []

    with open(labels_file, 'r') as lines:
        for line in lines:
            line = line.strip()
            node_id, label = line.split('\t')
            nodes.append(int(node_id))
            labels.append(label)

    def load_labels():
        if os.path.isfile(FLAGS.train_prefix + '/labels.npz'):
            return load_npz(FLAGS.train_prefix + '/labels.npz')

        encoder = LabelEncoder()
        encoder.fit(np.array(list(set(labels))))

        labels_csr = csr_matrix((60000000, encoder.classes_.size), dtype=np.int8)

        chunks_size = 10000
        for idx in range(0, len(labels), chunks_size):
            node_idx = nodes[idx: idx + chunks_size]
            label_idx = encoder.transform(np.array(labels[idx: idx + chunks_size]))

            labels_csr += csr_matrix((np.ones(len(node_idx)), (node_idx, label_idx)),
                                     shape=labels_csr.shape, dtype=np.int8)

        labels_csr[labels_csr > 1] = 1

        save_npz(FLAGS.train_prefix + '/labels.npz', labels_csr)
        return labels_csr

    labels = load_labels()

    print('Labels loaded')

    features = np.load(features_file)
    #features = csr_matrix((55000000, 1))
    features[0, :] = np.zeros(features.shape[1])  # unknown vertex

    print('Features loaded')

    return g, features, id_map, None, labels, nodes
    #return g, features, id_map(), random_walks(walks_file), clusters, None
