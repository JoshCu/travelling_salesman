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


def read_file(file_path: Path) -> np.ndarray:
    with open(file_path, "r") as file:
        data = file.readlines()
    # remove new line characters
    data = [line.strip() for line in data]

    # create a new numpy array of zeros
    adjacency_matrix = np.zeros((len(data), len(data)), dtype=np.int32)

    # populate the adjacency matrix
    for i, line in enumerate(data):
        for j, weight in enumerate(line.split(" ")):
            adjacency_matrix[i, j] = int(weight)

    # mirror across the diagonal to ensure symmetry
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    return adjacency_matrix


def main():
    n = 100  # Example size, adjust as needed
    file = f"Size{n}.graph"
    data = read_file(Path(file))
    weight, path = nearest_neighbor(data)
    print(f"Path: {path} with total weight: {weight}")


if __name__ == "__main__":
    main()
