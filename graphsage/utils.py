from __future__ import print_function

import numpy as np
import random
import json
import sys
import os
import subprocess

import graph_tool

import networkx as nx
from networkx.readwrite import json_graph
version_info = list(map(int, nx.__version__.split('.')))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"

WALK_LEN = 5
N_WALKS = 20


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


def load_data_from_graph(graph_file, walks_file):
    g = graph_tool.load_graph(graph_file)

    class id_map(object):
        def __getitem__(self, item):
            return int(item)

        def __len__(self):
            return g.num_vertices()

    class random_walks(object):
        def __init__(self, filename):
            self.filename = filename
            self._len = None

            self._open()

        def _open(self):
            print("Shuffling...")
            p = subprocess.Popen(['gshuf', '-o', self.filename + '.shuffle', self.filename])
            p.wait()
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
            return next(self.f)

    return g, None, id_map(), random_walks(walks_file), None


def run_random_walks(G, num_walks=N_WALKS, walk_len=WALK_LEN):
    for count, node in enumerate(G.vertices()):
        if not node.in_degree() and not node.out_degree():
            continue

        for i in range(num_walks):
            curr_node = node
            for j in range(walk_len):
                next_node = random.choice(list(curr_node.all_neighbors()))
                # self co-occurrences are useless
                if curr_node != node:
                    yield (node, curr_node)

                curr_node = next_node

        if count % 1000 == 0:
            print("Done walks for", count, "nodes")


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G = graph_tool.load_graph(graph_file)
    G.set_vertex_filter(G.vertex_properties.test, inverted=True)

    with open(out_file, "w") as fp:
        for node1, node2 in run_random_walks(G):
            fp.write(str(node1) + "\t" + str(node2) + '\n')

