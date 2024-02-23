import numpy as np
from itertools import permutations
from utils import load_file, evaluate_solution, parse_args

def brute_force(matrix):
    # get the number of nodes
    n = matrix.shape[0]
    best_path = None
    best_score = float("inf")
    # iterate over all permutations of the nodes (excluding 0)
    for path in permutations(range(1, n)):
        # add 0 to the start and end of the path
        npath = np.array([0] + list(path) + [0])
        score = evaluate_solution(matrix, npath)
        # if the score is better than the best score, update the best score and path
        if score < best_score:
            best_score = score
            best_path = npath
    return best_path, best_score

if __name__ == "__main__":
    file_path = parse_args()
    # load adjacency matrix from file
    matrix = load_file(file_path)
    path, score = brute_force(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")
    print(f"score: {evaluate_solution(matrix, path)}")


