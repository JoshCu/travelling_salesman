import numpy as np
import igraph as ig
from utils import load_file, parse_args, evaluate_solution

# Load your adjacency matrix
adj_matrix = load_file("g15.graph")

# Convert the adjacency matrix to a list of edges and weights
edges = []
weights = []
for i in range(adj_matrix.shape[0]):
    for j in range(i + 1, adj_matrix.shape[1]):
        if adj_matrix[i, j] > 0:
            edges.append((i, j))
            weights.append(adj_matrix[i, j])  # Store the weight of the edge

# Create a graph from the edges
g = ig.Graph(edges=edges)

# Add edge weights
g.es['weight'] = weights

# Print the graph summary, which now includes edges with weights
print(g.summary())

def prune_edges_to_two_per_node(graph, adj_matrix):
    # Iterate over all nodes in the graph
    
    for node in graph.vs():
        # Get the edges connected to the node
        connected_edges = node.incident()
        
        # Continue only if the node has more than two connected edges
        if len(connected_edges) > 2:
            # Sort the connected edges by weight, keeping the heaviest
            sorted_edges = sorted(connected_edges, key=lambda e: e['weight'], reverse=True)
            for edge in sorted_edges:
                if len(edge.source_vertex.incident()) > 2 and len(edge.target_vertex.incident()) > 2:
                    graph.delete_edges(edge.index)

    # fix connections > 2
    for node in graph.vs():
        connected_edges = node.incident()
        if len(connected_edges) <= 2:
            continue
        print("pain")
        




prune_edges_to_two_per_node(g)
print(g)

# Optionally, plot the graph. Uncomment the line below to plot if you're in a suitable environment
ig.plot(g, edge_label=g.es['weight'], target="g15.png")
