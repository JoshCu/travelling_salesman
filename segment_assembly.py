import numpy as np
import igraph as ig
from utils import load_file, parse_args, evaluate_solution
from tqdm import tqdm
from time import sleep

def get_edges_by_length(matrix):
    """
    Get the edges of the matrix sorted by length.

    Args:
        matrix (np.ndarray): The matrix of weights.

    Returns:
        list[tuple[int, int, int]]: The edges sorted by length.
    """
    edges = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            edges.append((i, j, matrix[i, j]))
    edges.sort(key=lambda x: x[2], reverse=True)
    return edges

# Load your adjacency matrix
adj_matrix = load_file("g10.graph")
edges_with_weights = get_edges_by_length(adj_matrix)
# Convert the adjacency matrix to a list of edges and weights
edges = []
weights = []
g = ig.Graph(directed=False)
g.add_vertices(adj_matrix.shape[0])
node_connection_count = {}
for vertex in g.vs:
    node_connection_count[vertex.index] = 0
for edge in edges_with_weights:
    start, end, weight = edge

    if node_connection_count[start] + node_connection_count[end] > 1:
        # loop would be created
        continue
    node_connection_count[start] += 1
    node_connection_count[end] += 1
    g.add_edge(start, end)
    g.es[-1]['weight'] = weight

start = None
end = None
for vertex in g.vs:
    if node_connection_count[vertex.index] == 1:
        if start is None:
            start = vertex.index
        else:
            end = vertex.index
            break


g.add_edge(start, end)
g.es[-1]['weight'] = adj_matrix[start, end]


def optimize_graph(graph, adj_matrix, optimization_iterations=10):
    # remove the two edges and swap their ends
    # get the two edges with the largest weights
    previously_swapped = []
    path = create_path_from_graph(graph)
    for i in range(optimization_iterations):
        largest_edges = [*graph.es[-2:]]
        # make sure the largest is first
        if largest_edges[0]['weight'] < largest_edges[1]['weight']:
            largest_edges = largest_edges[::-1]

        for edge in graph.es:
            if edge['weight'] > largest_edges[0]['weight']:
                largest_edges[0] = edge
            elif edge['weight'] > largest_edges[1]['weight']:
                # if the largest and second largest share a vertex, skip
                source_in = (edge.source == largest_edges[0].source or edge.target == largest_edges[0].source)
                target_in = (edge.source == largest_edges[0].target or edge.target == largest_edges[0].target)
                if not (source_in or target_in):
                    largest_edges[1] = edge

        print(largest_edges)
        # add the new edges
        a = largest_edges[0].source
        b = largest_edges[0].target
        c = largest_edges[1].source
        d = largest_edges[1].target
        # arrange pairs in the order of the path
        if path.index(a) > path.index(b):
            a, b = b, a
        if path.index(c) > path.index(d):
            c, d = d, c
        print(a, b, c, d)
        ig.plot(g,edge_label=g.es['weight'],vertex_label=range(len(g.vs)), target="a_before.png")
        graph.delete_edges(largest_edges)
        ig.plot(g,edge_label=g.es['weight'],vertex_label=range(len(g.vs)), target="a_after_delete.png")
        graph.add_edge(a, c)
        graph.add_edge(b, d)
        graph.es[-2]['weight'] = adj_matrix[largest_edges[0].target, largest_edges[1].source]
        graph.es[-1]['weight'] = adj_matrix[largest_edges[1].target, largest_edges[0].source]
        ig.plot(g,edge_label=g.es['weight'],vertex_label=range(len(g.vs)), target="a_after.png")
        # update the weights
        # remove the two old edges        
        path = create_path_from_graph(graph)

    return graph
    
def create_path_from_graph(graph):
    # Create a path from the graph
    # find vertex with only one connection

    path = [start]
    while len(path) < len(graph.vs):
        last_vertex = graph.vs[path[-1]]
        neighbours = last_vertex.neighbors()
        # Select the neighbour that isn't already in the path
        for neighbour in neighbours:
            if neighbour.index not in path:
                path.append(neighbour.index)
                break
    # Close the loop
    path.append(start)
    print(np.array(path))
    print(evaluate_solution(adj_matrix, path))
    return path

#prune_large_edges(g)
#reduce_to_two_edges_per_node(g, pruned_g)
g = optimize_graph(g, adj_matrix)
create_path_from_graph(g)

# Optionally, plot the graph. Uncomment the line below to plot if you're in a suitable environment
ig.plot(g, target="g15.png")
