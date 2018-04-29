#
# Grafy E-R metoda klasyczna, wypelnianie macierzy sąsiedztwa,
# 1 i 0 z prawdop p, narysować P(k), p=(0.1, 0.5, 0.7), N~100

import random
from pathlib import Path

import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats

GRAPHS_DIR = 'graphs'


class Graph(nx.Graph):
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.adjacency_matrix = np.zeros((num_nodes, num_nodes))
        super().__init__()

    def __call__(self, build_fn=None, p: float=None):
        if p:
            build_fn(self, p)
        else:
            build_fn(self)


def classic(graph: Graph, p: float=0.5):
        for node in range(graph.num_nodes):
            for neighbour in range(node):
                current_p = random.random()
                if current_p < p:
                    graph.add_edge(node, neighbour)
                    graph.adjacency_matrix[node, neighbour] = 1
                    graph.adjacency_matrix[neighbour, node] = 1


def node_degree_dist(adj_matrix: np.array):
    return np.sum(adj_matrix, axis=1)


def build_and_plot_graph(num_nodes: int, build_fn, probability: float=None):

    Path(GRAPHS_DIR).mkdir(exist_ok=True)
    figs_path = Path(GRAPHS_DIR + "/" + str(build_fn.__name__))
    figs_path.mkdir(exist_ok=True)

    g = Graph(num_nodes)
    g(build_fn, probability)
    probability = str(probability) if probability is not None else '_'

    position = nx.circular_layout(g)
    nx.draw(g, pos=position)
    labels = nx.draw_networkx_labels(g, pos=position)
    plt.savefig(str(figs_path / ("graph" + str(num_nodes) + "_p" + probability + ".png")))
    plt.show()

    degrees = node_degree_dist(g.adjacency_matrix)

    h, edges = scipy.histogram(degrees, bins=10, normed=True)
    k = edges[:-1] + (edges[1] - edges[0]) / 2
    plt.plot(k, h, label='hist')

    plt.title('P(k)')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.legend()
    plt.savefig(str(figs_path / ("P(k)_" + str(num_nodes) + "_p" + probability + ".png")))
    plt.show()

    print("Images saved")


# M-C algorytm Metropolis, E(t), p(k)

# o zadanym rozkładzie p(k) ~ k^-3 np. metoda odwrotnej dystrybuanty

# x_min = 1
# P(k) = k^-3
# P_cum = k ^ 2


def inv_distr(graph: Graph):
    # P(k) = k^-3
    degrees = np.round((1 - np.random.random(size=graph.num_nodes)) ** (-1 / 2))

    for node in range(graph.num_nodes):
        k = degrees[node]
        for neighbour in range(node):
            iter = 0
            while sum(graph.adjacency_matrix[node]) < k and iter < 1000:
                neighbour = random.randint(0, graph.num_nodes - 1)
                if neighbour == node or sum(graph.adjacency_matrix[node]) == degrees[neighbour]:
                    iter += 1
                    continue
                else:
                    graph.add_edge(node, neighbour)
                    graph.adjacency_matrix[node, neighbour] = 1
                    graph.adjacency_matrix[neighbour, node] = 1


if __name__ == '__main__':

    build_and_plot_graph(100, classic, 0.1)
    build_and_plot_graph(100, classic, 0.5)
    build_and_plot_graph(100, classic, 0.7)
    build_and_plot_graph(100, inv_distr)
