import cupy as cp
from numba import cuda
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32


# Assuming you have a function to load the adjacency matrix
def load_adjacency_matrix(file_path: str) -> cp.ndarray:
    with open(file_path, "r") as file:
        data = file.readlines()
    # remove new line characters
    data = [line.strip() for line in data]

    # create a new numpy array of zeros
    adjacency_matrix = cp.zeros((len(data), len(data)), dtype=cp.int32)

    # populate the adjacency matrix
    for i, line in enumerate(data):
        for j, weight in enumerate(line.split(" ")):
            adjacency_matrix[i, j] = int(weight)

    # mirror across the diagonal to ensure symmetry
    adjacency_matrix = cp.maximum(adjacency_matrix, adjacency_matrix.T)
    return adjacency_matrix


@cuda.jit
def tsp_simulated_annealing(
    solutions,
    best_solutions,
    objective_values,
    best_objective,
    temperature,
    adjacency_matrix,
    rng_states,
):
    idx = cuda.grid(1)
    if idx >= solutions.shape[0]:
        return

    # Create a new solution by swapping two cities in the tour
    city1, city2 = (
        xoroshiro128p_uniform_float32(rng_states, idx) % solutions.shape[1],
        xoroshiro128p_uniform_float32(rng_states, idx) % solutions.shape[1],
    )
    new_solution = solutions[idx].copy()
    new_solution[city1], new_solution[city2] = new_solution[city2], new_solution[city1]

    # Calculate the total distance of the new solution
    new_objective = 0
    for i in range(new_solution.shape[0] - 1):
        new_objective += adjacency_matrix[new_solution[i], new_solution[i + 1]]
    new_objective += adjacency_matrix[
        new_solution[-1], new_solution[0]
    ]  # Complete the tour

    # Acceptance criterion
    if new_objective < objective_values[idx] or math.exp(
        (objective_values[idx] - new_objective) / temperature
    ) > xoroshiro128p_uniform_float32(rng_states, idx):
        solutions[idx] = new_solution
        objective_values[idx] = new_objective
        if new_objective < best_objective[idx]:
            best_solutions[idx] = new_solution
            best_objective[idx] = new_objective


def main():
    file_path = "Size100.graph"
    adjacency_matrix = load_adjacency_matrix(file_path)

    # Initialize parameters for TSP SA
    num_solutions = 256  # Number of parallel solutions
    num_cities = adjacency_matrix.shape[0]
    solutions = cp.zeros((num_solutions, num_cities), dtype=cp.int32)
    for i in range(num_solutions):
        solutions[i] = cp.random.permutation(num_cities)
    best_solutions = cp.copy(solutions)

    # Initialize objective values and best objectives
    # Placeholder for initial objective value calculation
    objective_values = cp.zeros(num_solutions, dtype=cp.float32)
    best_objective = cp.full(num_solutions, cp.inf, dtype=cp.float32)

    # Simulated annealing parameters
    temperature = 1.0
    cooling_rate = 0.99
    num_iterations = 1000

    # CUDA random states
    rng_states = cuda.random.create_xoroshiro128p_states(256, seed=42)

    # Main SA loop
    for _ in range(num_iterations):
        tsp_simulated_annealing[32, 8](
            solutions,
            best_solutions,
            objective_values,
            best_objective,
            temperature,
            adjacency_matrix,
            rng_states,
        )
        temperature *= cooling_rate

    # Copy results back to host and process
    best_solutions_host = best_solutions.copy_to_host()
    print("Best solutions:", best_solutions_host)


if __name__ == "__main__":
    main()
