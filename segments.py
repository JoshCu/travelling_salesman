from utils import load_file, evaluate_solution, parse_args
import numpy as np


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
            edges.append(i, j, matrix[i, j])
    edges.sort(key=lambda x: x[2])
    return edges

def main(matrix):
    """
    Find the shortest path through a given matrix.

    Args:
        matrix (np.ndarray): The matrix of weights.

    Returns:
        List[int]: The shortest path.
    """
    edges = get_edges_by_length(matrix)
    pairs = []
    start_nodes = set()
    end_nodes = set()
    for edge in edges:        
        if edge[0] not in start_nodes or edge[1] not in end_nodes:
            pairs.append(edge)
            start_nodes.add(edge[0])
            end_nodes.add(edge[1])
    print(pairs)
    print(len(pairs))
    graph = {}
    for i in range(matrix.shape[0]):
        graph[i] = []
    for start, end, distance in edges:
        graph[start].append((end, distance))
        graph[end].append((start, distance))

    # fix this bit
    start_nodes = set(graph.keys())
    end_nodes = {end for _, end, _ in edges}
    start_node = (start_nodes - end_nodes).pop()
    # Construct the path
    path = []
    current_node = start_node
    path.append(current_node)
    while current_node in graph:
        next_node, distance = graph[current_node]
        path.append((next_node))
        current_node = next_node
    path.append(start_node)
    


    return path


if __name__ == "__main__":
    filename = parse_args()
    matrix = load_file(filename)
    path = main(matrix)
    print(evaluate_solution(matrix, path))
    print(path)