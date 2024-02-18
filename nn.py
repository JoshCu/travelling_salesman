import numpy as np
import numpy.ma as ma
from utils import load_file, evaluate_solution, parse_args

def nearest_neighbor(matrix):
    n = matrix.shape[0]
    path = [0]
    visited = set([0])
    for _ in range(n - 1):
        current = path[-1]
        best_next = None
        best_score = float("inf")
        for i in range(n):
            if i not in visited:
                if matrix[current, i] < best_score:
                    best_score = matrix[current, i]
                    best_next = i
        path.append(best_next)
        visited.add(best_next)
    path.append(0)
    score = evaluate_solution(matrix, path)
    return np.array(path), score

if __name__ == "__main__":
    file_path = parse_args()
    matrix = load_file(file_path)
    path, score = nearest_neighbor(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")


