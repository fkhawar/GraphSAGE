from __future__ import print_function

import numpy as np
import itertools
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os
import tensorflow as tf
import graph_tool


SHUF_UTIL = (sys.platform.startswith('linux') and 'shuf') or \
            (sys.platform.startswith('darwin') and 'gshuf') or ''


flags = tf.app.flags
FLAGS = flags.FLAGS


def load_data_from_graph(graph_file, features_file, labels_file, map_file, walks_file=None, walks_per_user=20):
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

    nodes = []
    labels = []
    walks = []

    if walks_file and walks_per_user:
        with open(walks_file, 'r') as users:
            for user in users:
                try:
                    pages = json.loads(user)['pageIds']
                except (ValueError, KeyError):
                    continue

                pages = [p for p in pages if p in id_map]
                pairs = np.array([(x, y) for x, y in itertools.permutations(pages, 2) if x < y])

                if len(pairs) < walks_per_user:
                    walks.extend(pairs)
                else:
                    idxs = np.random.choice(np.arange(len(pairs)), walks_per_user)
                    walks.extend(pairs[idxs])

    print("Walk file", len(walks))

    if labels_file:
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

            labels_csr = csr_matrix((g.num_vertices(), encoder.classes_.size), dtype=np.int8)

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
    print('Features loaded')

    return g, features, id_map, nodes, labels, walks
