import numpy as np
import random
from multiprocessing import Pool, cpu_count
from nn import nearest_neighbor
from utils import load_file, evaluate_solution, parse_args
from tqdm import tqdm
class SA:
    def __init__(self, iterations, temp, adjacency_matrix, gamma, starting_path=None):
        self.iterations = iterations
        self.temp = temp
        self.adjacency_matrix = adjacency_matrix
        self.gamma = gamma
        if starting_path is not None:
            self.starting_path = starting_path
        else:
            self.starting_path = random.shuffle(self.nodes)


    def check_accept(self, temp, new_solution, current_solution):
        delta = new_solution - current_solution
        prob = np.exp(-delta / temp) if delta > 0 else 1
        return prob > random.uniform(0, 1)

    def cooling_temp(self, temp):
        return temp / (1 + self.gamma * temp)

    def random_swap_elements(self, path):
        path_new = path.copy()
        i, j = random.sample(range(1, len(path_new) - 1), 2)
        path_new[i], path_new[j] = path_new[j], path_new[i]
        return path_new

    def run(self):
        temp = self.temp
        
        current_path = self.starting_path.copy()
        current_distance = evaluate_solution(self.adjacency_matrix, current_path)
        best_path = current_path
        best_distance = current_distance

        for _ in range(self.iterations):
            new_path = self.random_swap_elements(current_path)
            new_distance = evaluate_solution(self.adjacency_matrix, new_path)

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



def run_epochs(adjacency_matrix, s_path=None):
    # SA parameters
    # Run parallel SA
    if s_path is None:
        s_path = nearest_neighbor(adjacency_matrix)[0]

    best_distance = float('inf')
    best_path = None
    epochs = 50
    iterations = 100000
    temp = 1000
    gamma = 0.99
    epochs = 50
    num_processes = cpu_count()
    distances_list = []
    for i in tqdm(range(epochs)):
        starting_path = s_path        
        results = parallel_sa(iterations, temp, adjacency_matrix, gamma, starting_path, num_processes)
        for result in results:
            if result[1] < best_distance:
                best_distance = result[1]
                best_path = result[0]
        print(f"best_path: {best_path}")
        print(f"Epoch {i}: Best distance: {best_distance}")
        distances_list.append(best_distance)
        starting_path = best_path.copy()
        if len(distances_list) > 3:
            if sum(distances_list[-3:]) / 3 == best_distance:
                break

    return best_path, best_distance

if __name__ == "__main__":
    file_path = parse_args()
    matrix = load_file(file_path)
    path, score = run_epochs(matrix)
    print(f"Best path: {path}")
    print(f"Score: {score}")

