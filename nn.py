import numpy as np
import numpy.ma as ma
from utils import load_file, evaluate_solution, parse_args
from tqdm import tqdm
def nearest_neighbor(matrix):
    # get the number of nodes
    n = matrix.shape[0]
    # start at node 0
    path = [0]
    # keep track of visited nodes to avoid cycles
    visited = set([0])
    # iterate over the number of nodes
    for _ in tqdm(range(n - 1)):
        # get the current node
        current = path[-1]
        best_next = None
        best_score = float("inf")
        # iterate over possible next nodes
        for i in range(n):
            if i not in visited:
                # save the best next node
                if matrix[current, i] < best_score:
                    best_score = matrix[current, i]
                    best_next = i
        # add the best next node to the path and mark it as visited
        path.append(best_next)
        visited.add(best_next)
    # add the starting node to the end of the path
    path.append(0)
    score = evaluate_solution(matrix, path)
    return np.array(path), score

if __name__ == "__main__":
    file_path = parse_args()
    # load adjacency matrix from file
    matrix = load_file(file_path)
    path, score = nearest_neighbor(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")


