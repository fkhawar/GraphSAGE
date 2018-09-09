import _random_walk
import sys
import graph_tool

WALK_LEN = 5
N_WALKS = 20

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G = graph_tool.load_graph(graph_file)
    G.set_vertex_filter(G.vertex_properties.test, inverted=True)

    with open(out_file, "w") as fp:
        for node1, node2 in _random_walk.run_random_walks(G, N_WALKS, WALK_LEN):
            fp.write(str(node1) + "\t" + str(node2) + '\n')
