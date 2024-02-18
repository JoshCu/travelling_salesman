import numpy as np
import numpy.ma as ma
from itertools import permutations
from utils import load_file, evaluate_solution, parse_args
from nn import nearest_neighbor

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
    file_path = parse_args()
    matrix = load_file(file_path)
    path, score = nearest_neighbor(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")
    path, score = permute_path(matrix, path)
    print(f"Best path: {path}")
    print(f"Score: {score}")


