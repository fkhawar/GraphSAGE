import random
from cython.parallel import prange


def run_random_walks(G, int num_walks, int walk_len):
    for count, node in enumerate(G.vertices()):
        if not node.out_degree():
            continue

        for i in prange(num_walks, nogil=True):
            curr_node = node
            for j in range(walk_len):
                neighbors = list(curr_node.out_neighbors())
                if not neighbors:
                    break

                next_node = random.choice(neighbors)
                # self co-occurrences are useless
                if curr_node != node:
                    yield (node, curr_node)

                curr_node = next_node

        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
