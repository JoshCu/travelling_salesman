import numpy as np
from itertools import permutations
# load the file into numpy array
def load_file(filename):
    
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]
    file_content = file_content[:15]
    matrix = np.zeros((len(file_content), len(file_content)))
    for i, row in enumerate(file_content):
        for j, weight in enumerate(row[:-1]):
            matrix[i, j] = weight
    # mirror over the diagonal
    matrix = np.maximum(matrix, matrix.T)
    print(matrix)
    return matrix

def evaluate_solution(matrix, path):
    start_nodes = path[:-1]
    end_nodes = path[1:]
    return matrix[start_nodes, end_nodes].sum()

def brute_force(matrix):
    n = matrix.shape[0]
    best_path = None
    best_score = float("inf")
    for path in permutations(range(1, n)):
        npath = np.array([0] + list(path) + [0])
        score = evaluate_solution(matrix, path)
        if score < best_score:
            best_score = score
            best_path = npath
    return best_path, best_score

def main():
    filename = "Size1000.graph"
    matrix = load_file(filename)
    path, score = brute_force(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")

if __name__ == "__main__":
    main()

