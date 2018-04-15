#
# Grafy E-R metoda klasyczna, wypelnianie macierzy sąsiedztwa,
# 1 i 0 z prawdop p, narysować P(k), p=(0.1, 0.5, 0.7), N~100

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import random
import scipy.stats

GRAPHS_DIR = 'graphs'


class Graph(nx.Graph):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        super().__init__()

    def __call__(self, build_fn=None, p: float=0.5):
        build_fn(self, p)


def classic_build_fn(graph: Graph, p: float=0.5):
        for node in range(graph.num_nodes):
            for neighbour in range(node):
                current_p = random.random()
                if current_p < p:
                    graph.add_edge(node, neighbour)
                    graph.adjacency_matrix[node, neighbour] = 1
                    graph.adjacency_matrix[neighbour, node] = 1


def node_degree_dist(adj_matrix: np.array):
    return np.sum(adj_matrix, axis=1)


def build_and_plot_graph(num_nodes: int, probability: float):

    g = Graph(num_nodes)
    g(classic_build_fn, probability)

    position = nx.circular_layout(g)
    nx.draw(g, pos=position)
    labels = nx.draw_networkx_labels(g, pos=position)
    plt.savefig(GRAPHS_DIR + "/graph" + str(num_nodes) + "_p" + str(probability) + ".png")
    plt.show()

    degrees = node_degree_dist(g.adjacency_matrix)

    h, edges = scipy.histogram(degrees, bins=10, normed=True)
    k = edges[:-1] + (edges[1] - edges[0]) / 2
    plt.plot(k, h, label='hist')

    plt.title('P(k)')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()
    plt.savefig(GRAPHS_DIR + "/graph" + str(num_nodes) + "_p" + str(probability) + "_P(k).png")
    plt.show()

    print("Images saved")

if __name__ == '__main__':

    build_and_plot_graph(100, 0.1)
    build_and_plot_graph(100, 0.5)
    build_and_plot_graph(100, 0.7)


