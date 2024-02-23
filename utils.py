import numpy as np
import argparse
from pathlib import Path
from functools import lru_cache

@lru_cache
def load_file(filename: str) -> np.ndarray:
    """
    Load a file containing a matrix of weights and convert it into a numpy array.

    Args:
        filename (str): The path to the file.

    Returns:
        np.ndarray: The matrix of weights.
    """
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]

    matrix = np.zeros((len(file_content), len(file_content)))
    for i, row in enumerate(file_content):
        for j, weight in enumerate(row[:-1]):
            matrix[i, j] = weight
    # mirror over the diagonal
    matrix = np.maximum(matrix, matrix.T)
    return matrix

def evaluate_solution(matrix: np.ndarray, path: list[int]) -> int:
    """
    Evaluate the total weight of a given path in the matrix.

    Args:
        matrix (np.ndarray): The matrix of weights.
        path (List[int]): The path to evaluate.

    Returns:
        int: The total weight of the path.
    """
    if (type(path) is tuple):
        path = list(path)
    
    if path[0] != path[-1]:
        path.append(path[0])
    start_nodes = path[:-1]
    end_nodes = path[1:]
    return matrix[start_nodes, end_nodes].sum()


def parse_args():
    """
    Parse command line arguments.

    Returns:
        str: The file name.
    """
    parser = argparse.ArgumentParser(description="Travelling Salesman Problem")
    parser.add_argument("filename", type=str, help="Path to the file")
    args = parser.parse_args()
    return Path(args.filename)
