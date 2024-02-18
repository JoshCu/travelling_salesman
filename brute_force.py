import numpy as np
from itertools import permutations
from utils import load_file, evaluate_solution, parse_args

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

if __name__ == "__main__":
    file_path = parse_args()
    matrix = load_file(file_path)
    path, score = brute_force(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")


