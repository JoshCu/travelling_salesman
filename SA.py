import numpy as np
import random
from multiprocessing import Pool
from nn import nearest_neighbor

def load_file(filename):
    with open(filename, "r") as file:
        file_content = [[int(weight) for weight in line.strip().split()] for line in file]

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

class SA:
    def __init__(self, iterations, temp, adjacency_matrix, gamma, starting_path=None):
        self.iterations = iterations
        self.temp = temp
        self.adjacency_matrix = adjacency_matrix
        self.gamma = gamma
        self.nodes = list(range(len(adjacency_matrix)))
        if starting_path is not None:
            self.starting_path = starting_path
        else:
            self.starting_path = nearest_neighbor(adjacency_matrix)[0]

    def total_distance(self, path):
        start_nodes = path[:-1]
        end_nodes = path[1:]
        return self.adjacency_matrix[start_nodes, end_nodes].sum()

    def check_accept(self, temp, new_solution, current_solution):
        delta = new_solution - current_solution
        prob = np.exp(-delta / temp) if delta > 0 else 1
        return prob > random.uniform(0, 1)

    def cooling_temp(self, temp):
        return temp / (1 + self.gamma * temp)

    def swap_elements(self, path):
        path_new = path.copy()
        i, j = random.sample(range(1, len(path_new) - 1), 2)
        path_new[i], path_new[j] = path_new[j], path_new[i]
        return path_new

    def run(self):
        temp = self.temp
        random.shuffle(self.nodes)
        current_path = nearest_neighbor(self.adjacency_matrix)[0]
        current_distance = self.total_distance(current_path)
        best_path = current_path
        best_distance = current_distance

        for _ in range(self.iterations):
            new_path = self.swap_elements(current_path)
            new_distance = self.total_distance(new_path)

            if new_distance < best_distance:
                best_path = new_path
                best_distance = new_distance

            if self.check_accept(temp, new_distance, current_distance):
                current_path = new_path
                current_distance = new_distance

            temp = self.cooling_temp(temp)

        return best_path, best_distance

def run_sa(params):
    sa = SA(*params)
    return sa.run()

def parallel_sa(iterations, temp, adjacency_matrix, gamma, starting_path, num_processes):
    pool = Pool(processes=num_processes)
    params = [(iterations, temp, adjacency_matrix, gamma, starting_path) for _ in range(num_processes)]
    results = pool.map(run_sa, params)
    pool.close()
    pool.join()
    return results



adjacency_matrix = load_file("Size100.graph")

# SA parameters
iterations = 1000000
temp = 1000
gamma = 0.99
epochs = 10
num_processes = 20 
starting_path = nearest_neighbor(adjacency_matrix)[0]

# Run parallel SA
best_distance = float('inf')
best_path = None
for i in range(epochs):
    results = parallel_sa(iterations, temp, adjacency_matrix, gamma, starting_path, num_processes)
    for result in results:
        if result[1] < best_distance:
            best_distance = result[1]
            best_path = result[0]
    print(f"Epoch {i}: Best distance: {best_distance}")
    starting_path = best_path

print("Best distance:", best_distance)

