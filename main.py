import numpy as np
from pathlib import Path
from numba import njit


@njit
def nearest_neighbor(adjacency_matrix: np.ndarray) -> (int, np.ndarray):
    n = adjacency_matrix.shape[0]
    visited = np.zeros(n, dtype=np.bool_)
    path = np.zeros(n + 1, dtype=np.int32)  # Modified to include return to start
    total_weight = 0

    # Start from node 0
    current_city = 0
    visited[current_city] = True
    path[0] = current_city

    for i in range(1, n):
        nearest_city = -1
        nearest_distance = np.iinfo(np.int32).max

        for j in range(n):
            if not visited[j] and adjacency_matrix[current_city, j] < nearest_distance:
                nearest_distance = adjacency_matrix[current_city, j]
                nearest_city = j

        visited[nearest_city] = True
        path[i] = nearest_city
        total_weight += nearest_distance
        current_city = nearest_city

    # Add distance back to the start (node 0)
    total_weight += adjacency_matrix[current_city, 0]
    path[n] = 0  # Ensure the path ends at node 0

    return total_weight, path


@njit
def evaluate_path(adjacency_matrix, path):
    weight = 0
    for i in range(len(path) - 1):
        weight += int(adjacency_matrix[path[i], path[i + 1]])
    return weight


@njit
def heaps_algorithm(
    n: int,
    a: np.ndarray,
    adjacency_matrix: np.ndarray,
    best_path=np.array([]),
    best_weight=np.iinfo(np.int32).max,
):
    if n == 1:
        current_weight = evaluate_path(adjacency_matrix, a)
        if current_weight < best_weight:
            best_weight = current_weight
            best_path = a.copy()  # Make a copy of the current path
    else:
        for i in range(n - 1):
            best_path, best_weight = heaps_algorithm(
                n - 1, a, adjacency_matrix, best_path, best_weight
            )
            if n % 2 == 0:
                a[i], a[n - 1] = a[n - 1], a[i]
            else:
                a[0], a[n - 1] = a[n - 1], a[0]
        best_path, best_weight = heaps_algorithm(
            n - 1, a, adjacency_matrix, best_path, best_weight
        )

    return best_path, best_weight


def read_file(file_path: Path, trim=0) -> np.ndarray:
    with open(file_path, "r") as file:
        data = file.readlines()
    # remove new line characters
    data = [line.strip() for line in data]

    # trim the graph size
    data = data[:trim]

    # create a new numpy array of zeros
    adjacency_matrix = np.zeros((len(data), len(data)), dtype=np.int32)

    # populate the adjacency matrix
    for i, line in enumerate(data):
        for j, weight in enumerate(line.split(" ")):
            adjacency_matrix[i, j] = int(weight)

    # mirror across the diagonal to ensure symmetry
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    return adjacency_matrix


def bf():
    n = 5  # Example size, adjust as needed
    fn = float(n)
    file = f"Size100.graph"
    data = read_file(Path(file), 5)
    best_path, best_weight = heaps_algorithm(n, np.arange(fn), data)
    print(f"Path: {best_path} with total weight: {best_weight}")


def nn():
    n = 1000  # Example size, adjust as needed
    file = f"Size{n}.graph"
    data = read_file(Path(file))
    weight, path = nearest_neighbor(data)
    print(f"Path: {path} with total weight: {weight}")


if __name__ == "__main__":
    bf()
