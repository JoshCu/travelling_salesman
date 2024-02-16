import numpy as np
import numpy.ma as ma
from itertools import permutations

# load the file into numpy array
def load_file(filename):
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]

    #file_content = file_content[:10]

    matrix = np.zeros((len(file_content), len(file_content)))
    for i, row in enumerate(file_content):
        for j, weight in enumerate(row[:-1]):
            matrix[i, j] = weight
    # mirror over the diagonal
    matrix = np.maximum(matrix, matrix.T)

    return matrix

def evaluate_solution(matrix, path):    
    start_nodes = path[:-1]
    end_nodes = path[1:]

    return matrix[start_nodes, end_nodes].sum()


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

def permute_path(matrix, path):
    best_score = evaluate_solution(matrix, path)
    best_path = path 
    list_path = list(path)
    front = []
    end = []
    for j in range(int((len(path)-1)/2)):
        front = front + list_path[:1]
        end = list_path[-1:] + end
        middle = list_path[1:-1]
        middle = np.array(middle)
        for i in range(len(middle)):
            # shift path by one
            middle = np.roll(middle, 1)
            test_path = np.array(front + list(middle) + end)
            score = evaluate_solution(matrix, test_path)
            if score < best_score:
                best_score = score
                best_path = test_path
                list_path = list(middle)
        if len(middle) == 2:
            break
    return best_path, best_score



def main():
    filename = "Size1000.graph"
    matrix = load_file(filename)
    masked_matrix = ma.masked_array(matrix, mask=matrix==0)
    min_rows_sum = masked_matrix.min(axis=1).sum()
    print(f"Minimum row sum: {min_rows_sum}")
    #path, score = brute_force(matrix)
    path, score = nearest_neighbor(matrix)
    print(f"NN Best path: {path}")
    print(f"NN Score: {score}")
    path, score = permute_path(matrix, path)
    print(f"Best path: {path}")
    print(f"Score: {score}")



if __name__ == "__main__":
    main()


