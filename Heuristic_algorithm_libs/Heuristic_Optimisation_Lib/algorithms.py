import networkx as nx
import matplotlib.pyplot as plt
from typing import List
import timeit
import time
import random
from IPython.display import clear_output
import numpy as np
import math
from problems import Knapsack, TSP


#  dynamic programming solution for TSP
class DP_TSP:
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance
        self.min_cost = float('inf')
        self.best_path = []

    def solve(self):
        n = len(self.tsp.cities)
        dp = [[float('inf')] * n for _ in range(1 << n)]
        parent = [[None] * n for _ in range(1 << n)]
        
        dp[1][0] = 0  # starting from city 0
        
        # Iterate through all subsets of vertices
        for mask in range(1, 1 << n):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue

                for v in range(n):
                    if mask & (1 << v) and self.tsp.distance_matrix_list[u][v]:
                        if dp[mask][u] > dp[mask ^ (1 << u)][v] + self.tsp.distance_matrix_list[u][v]:
                            dp[mask][u] = dp[mask ^ (1 << u)][v] + self.tsp.distance_matrix_list[u][v]
                            parent[mask][u] = v

        # Reconstruct the shortest path and compute the minimum cost
        mask = (1 << n) - 1  # All cities have been visited
        u = 0
        last_city = min(range(1, n), key=lambda v: dp[mask][v] + self.tsp.distance_matrix_list[v][0])
        self.min_cost = dp[mask][last_city] + self.tsp.distance_matrix_list[last_city][u]

        # Reconstruct the best path
        self.best_path = [0]  # start from city 0
        while True:
            self.best_path.append(last_city)
            u = last_city
            last_city = parent[mask][u]
            if last_city is None:  # break the loop when no more parent to trace
                break
            mask ^= (1 << u)

        return self.best_path[::-1], self.min_cost  # reverse the path to get the correct order

# dynamic programming solution for knapsack problem
class DP_knapsack:
    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity
        self.dp_table = [[0] * (self.capacity + 1) for _ in range(self.n + 1)]

    def solve(self):
        # Implementing the dynamic programming solution
        for i in range(1, self.n + 1):
            for w in range(self.capacity + 1):
                if self.weights[i - 1] <= w:
                    self.dp_table[i][w] = max(
                        self.dp_table[i - 1][w],
                        self.dp_table[i - 1][w - self.weights[i - 1]] + self.values[i - 1]
                    )
                else:
                    self.dp_table[i][w] = self.dp_table[i - 1][w]

        # Backtracking to find the items included in the optimal solution
        w = self.capacity
        solution = [0] * self.n
        for i in range(self.n, 0, -1):
            if self.dp_table[i][w] != self.dp_table[i - 1][w]:
                solution[i - 1] = 1
                w -= self.weights[i - 1]

        return solution, self.dp_table[self.n][self.capacity]

# LocalSearch algorithm for TSP
class LS_TSP:
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance

    def delta_eval(self, i, j, path):
        """
        Performs Delta Evaluation to calculate the change in distance 
        when two cities at index i and j in the given path are swapped.
        """
        n = len(path)
        prev_i, next_i = path[(i - 1) % n], path[(i + 1) % n]
        prev_j, next_j = path[(j - 1) % n], path[(j + 1) % n]
        
        if abs(i - j) % n == 1 or abs(i - j) % n == n - 1:
            return 0  # In this case, the delta is zero as the cities are adjacent
        
        old_distance = (self.tsp.distance_matrix_dict[prev_i][path[i]] + 
                        self.tsp.distance_matrix_dict[next_i][path[i]] + 
                        self.tsp.distance_matrix_dict[prev_j][path[j]] + 
                        self.tsp.distance_matrix_dict[next_j][path[j]])
        
        new_distance = (self.tsp.distance_matrix_dict[prev_i][path[j]] + 
                        self.tsp.distance_matrix_dict[next_i][path[j]] + 
                        self.tsp.distance_matrix_dict[prev_j][path[i]] + 
                        self.tsp.distance_matrix_dict[next_j][path[i]])
        
        delta = new_distance - old_distance
        return delta

    def optimize(self, initial_solution, distance_calculation_method='delta'):
        current_solution = initial_solution.copy()
        improved = True

        while improved:
            improved = False
            for i in range(len(current_solution)):
                for j in range(i+1, len(current_solution)):
                    delta = 0
                    if distance_calculation_method == 'delta':
                        delta = self.delta_eval(i, j, current_solution)
                    elif distance_calculation_method == 'direct':
                        # Calculate the total distance before and after the swap
                        old_distance = self.tsp.total_distance_direct(current_solution)
                        current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
                        new_distance = self.tsp.total_distance_direct(current_solution)
                        delta = new_distance - old_distance
                        # Swap back
                        current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
                    elif distance_calculation_method == 'matrix':
                        # Calculate the total distance before and after the swap using the matrix
                        old_distance = self.tsp.total_distance_matrix(current_solution)
                        current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
                        new_distance = self.tsp.total_distance_matrix(current_solution)
                        delta = new_distance - old_distance
                        # Swap back
                        current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
                    
                    if delta < 0:
                        # Swap the cities and update the current solution
                        current_solution[i], current_solution[j] = current_solution[j], current_solution[i]
                        improved = True
        return current_solution
    
# LocalSearch algorithm for knapsack problem
class LS_knapsack:
    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by flipping each bit."""
        neighbors = []
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate a solution with penalty for infeasible solutions."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def solve(self):
        """Implement the local search algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution
        current_value = self.evaluate(current_solution)

        while True:
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = max(neighbors_values)
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            if best_neighbor_value <= current_value:
                break  # If no improvement is found, stop the search

            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value
    

# Variable Neighborhood Search (VNS) algorithm algorithm for TSP
class VNS_TSP:
    def __init__(self, tsp_instance, neighborhood_structures):
        self.tsp = tsp_instance
        self.neighborhood_structures = neighborhood_structures  # A list of neighborhood structures
    
    def shake(self, k, solution):
        """
        Perform the shaking step based on the k-th neighborhood structure.
        """
        return self.neighborhood_structures[k](solution)
    
    def local_search(self, initial_solution):
        """
        Perform a local search on the initial_solution using the first neighborhood structure.
        """
        # Using the first neighborhood structure for local search
        return self.neighborhood_structures[0](initial_solution)
    
    def optimize(self, initial_solution, max_iter=100):
        """
        Optimize the TSP problem using VNS.
        """
        current_solution = initial_solution[:]
        best_solution = initial_solution[:]
        best_distance = self.tsp.total_distance_direct(initial_solution)
        
        k_max = len(self.neighborhood_structures)
        
        for i in range(max_iter):
            k = 1
            while k <= k_max:
                # Shaking step
                new_starting_solution = self.shake(k - 1, current_solution)
                
                # Local Search
                new_solution = self.local_search(new_starting_solution)
                
                # Calculate the new distance
                new_distance = self.tsp.total_distance_direct(new_solution)
                
                # Update the current and best solutions
                if new_distance < best_distance:
                    best_solution = new_solution[:]
                    best_distance = new_distance
                    current_solution = new_solution[:]
                    k = 1  # Reset the neighborhood index
                else:
                    k += 1  # Move to the next neighborhood
                
        return best_solution, best_distance

# Defining some example neighborhood structures
def swap_two_cities(solution):
    """
    Swap two randomly chosen cities in the solution.
    """
    new_solution = solution[:]
    i, j = random.sample(range(len(new_solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution

def reverse_subsequence(solution):
    """
    Reverse a random subsequence in the solution.
    """
    new_solution = solution[:]
    i, j = sorted(random.sample(range(len(new_solution)), 2))
    new_solution[i:j+1] = reversed(new_solution[i:j+1])
    return new_solution

# Hill Climbing algorithm for TSP

class HC_TSP:
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by swapping two cities."""
        neighbors = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = solution[j], solution[i]  # Swap two cities to create a neighbor
                neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate the total distance of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def solve(self, max_iterations=100):
        """Implement the hill climbing algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = random.sample(range(self.n), self.n)  # Random initial solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = min(neighbors_values)  # We want to minimize the total distance
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            if best_neighbor_value >= current_value:
                break  # If no improvement is found, stop the search

            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value

# Random Restart Hill Climbing algorithm for TSP
class HCrr_TSP:
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by swapping two cities."""
        neighbors = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = solution[j], solution[i]  # Swap two cities to create a neighbor
                neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate the total distance of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def hill_climbing(self, initial_solution, max_iterations=100):
        """Implement the basic hill climbing algorithm starting with a given initial solution."""
        current_solution = initial_solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = min(neighbors_values)  # We want to minimize the total distance
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            if best_neighbor_value >= current_value:
                break  # If no improvement is found, stop the search

            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value

    def solve(self, restarts=10, max_iterations=100):
        """Implement the random restart hill climbing algorithm with a specified number of restarts."""
        best_solution = None
        best_value = float('inf')  # Initialize to a high value because we want to minimize the distance

        for _ in range(restarts):
            initial_solution = random.sample(range(self.n), self.n)  # Random initial solution for each restart
            solution, value = self.hill_climbing(initial_solution, max_iterations)
            
            if value < best_value:  # If a better solution is found, update the best solution and value
                best_solution = solution
                best_value = value

        return best_solution, best_value
# Steepest Ascent Hill Climbing algorithm for TSP
class HCsa_TSP:
    def __init__(self, tsp_instance):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by swapping two cities."""
        neighbors = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                neighbor = solution.copy()
                neighbor[i], neighbor[j] = solution[j], solution[i]  # Swap two cities to create a neighbor
                neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate the total distance of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def solve(self, max_iterations=100):
        """Implement the steepest ascent hill climbing algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = random.sample(range(self.n), self.n)  # Random initial solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = min(neighbors_values)  # We want to minimize the total distance
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            # In this variation, we move to the best neighbor even if it doesn't improve the solution
            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value
    
# TabuSearch algorithm for TSP
from collections import deque
class TS_TSP:
    def __init__(self, tsp_instance, tabu_list_size=5):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list
        self.tabu_list_size = tabu_list_size
        self.tabu_list = deque(maxlen=tabu_list_size)  # A queue to store the recent moves

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by swapping two cities."""
        neighbors = []
        for i in range(self.n):
            for j in range(i+1, self.n):
                if ((i, j) not in self.tabu_list) and ((j, i) not in self.tabu_list):  # Check if the move is not tabu
                    neighbor = solution.copy()
                    neighbor[i], neighbor[j] = solution[j], solution[i]  # Swap two cities to create a neighbor
                    neighbors.append((neighbor, (i, j)))  # Store the move along with the neighbor
        return neighbors

    def evaluate(self, solution):
        """Evaluate the total distance of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def solve(self, max_iterations=50):
        """Implement the tabu search algorithm."""
        current_solution = random.sample(range(self.n), self.n)  # Random initial solution
        best_solution = current_solution.copy()
        current_value = self.evaluate(current_solution)
        best_value = current_value  # Initialize the best value to the value of the initial solution

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            if not neighbors:  # If there are no non-tabu neighbors, terminate the search
                break

            neighbors_values = [self.evaluate(neighbor[0]) for neighbor in neighbors]

            best_neighbor_value = min(neighbors_values)  # We want to minimize the total distance
            best_neighbor_index = neighbors_values.index(best_neighbor_value)
            best_neighbor, best_move = neighbors[best_neighbor_index]

            self.tabu_list.append(best_move)  # Add the move to the tabu list

            current_solution = best_neighbor
            current_value = best_neighbor_value

            if current_value < best_value:  # If a better solution is found, update the best solution and value
                best_solution = current_solution.copy()
                best_value = current_value

        return best_solution, best_value
    
# Simulated Annealing algorithm for TSP
class SA_TSP:
    def __init__(self, tsp_instance, cooling_schedule='lundy_and_mees', alpha=0.95, beta=0.01, initial_temperature=1000, final_temperature=1):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list
        self.cooling_schedule = cooling_schedule
        self.alpha = alpha
        self.beta = beta
        self.temperature = initial_temperature
        self.final_temperature = final_temperature
        self.initial_temperature = initial_temperature  # Added for logarithmic cooling

    def generate_neighbor(self, solution):
        """Generate a neighbor of the current solution by randomly swapping two cities."""
        i, j = random.sample(range(self.n), 2)  # Randomly select two distinct cities
        neighbor = solution.copy()
        neighbor[i], neighbor[j] = solution[j], solution[i]  # Swap two cities to create a neighbor
        return neighbor

    def evaluate(self, solution):
        """Evaluate the total distance of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def accept_solution(self, delta_e, temperature):
        """Determine whether to accept the new solution according to the Metropolis criterion."""
        if delta_e < 0:
            return True
        else:
            return random.random() < math.exp(-delta_e / temperature)

    def cool_down(self, temperature, iteration):
        """Cool down the temperature according to the selected cooling schedule."""
        if self.cooling_schedule == 'geometric':
            return self.alpha * temperature
        elif self.cooling_schedule == 'linear':
            return temperature - self.alpha
        elif self.cooling_schedule == 'lundy_and_mees':
            return temperature / (1 + self.beta * temperature)
        elif self.cooling_schedule == 'logarithmic':
            return self.initial_temperature / (1 + math.log1p(iteration))

    def solve(self, max_iterations=100):
        """Implement the simulated annealing algorithm."""
        current_solution = random.sample(range(self.n), self.n)  # Random initial solution
        best_solution = current_solution.copy()
        current_value = self.evaluate(current_solution)
        best_value = current_value  # Initialize the best value to the value of the initial solution
        iteration = 0

        while self.temperature > self.final_temperature and iteration < max_iterations:
            iteration += 1

            neighbor = self.generate_neighbor(current_solution)
            neighbor_value = self.evaluate(neighbor)

            delta_e = neighbor_value - current_value

            if self.accept_solution(delta_e, self.temperature):
                current_solution = neighbor
                current_value = neighbor_value

                if current_value < best_value:  # If a better solution is found, update the best solution and value
                    best_solution = current_solution.copy()
                    best_value = current_value

            self.temperature = self.cool_down(self.temperature, iteration)

        return best_solution, best_value
    

# Genetic Algorithm for TSP
class GA_TSP:
    def __init__(self, tsp_instance, population_size=200, mutation_rate=0.1, generations=1000):
        self.tsp = tsp_instance
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.distance_matrix = self.tsp.distance_matrix_list
        # self.num_cities = self.tsp.num_cities

    def initialize_population(self):
        population = []
        base_solution = list(range(self.tsp.num_cities))
        for _ in range(self.population_size):
            random.shuffle(base_solution)
            population.append(base_solution.copy())
        return population

    def fitness(self, solution):
        total_distance = sum(self.distance_matrix[solution[i-1]][solution[i]] for i in range(self.tsp.num_cities))
        return 1 / total_distance

    def selection(self, population):
        fitnesses = [self.fitness(ind) for ind in population]
        selected_parents = random.choices(population, weights=fitnesses, k=self.population_size)
        return selected_parents

    def crossover(self, parent1, parent2):
        half_size = self.tsp.num_cities // 2
        child1 = parent1[:half_size] + [gene for gene in parent2 if gene not in parent1[:half_size]]
        child2 = parent2[:half_size] + [gene for gene in parent1 if gene not in parent2[:half_size]]
        return child1, child2

    def mutation(self, individual):
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.tsp.num_cities), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]

    def evolve(self, population):
        new_population = []
        for i in range(0, self.population_size, 2):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = self.crossover(parent1, parent2)
            self.mutation(child1)
            self.mutation(child2)
            new_population.extend([child1, child2])
        return new_population

    def solve(self):
        population = self.initialize_population()
        for _ in range(self.generations):
            population = self.selection(population)
            population = self.evolve(population)
        best_solution = min(population, key=lambda x: 1/self.fitness(x))
        best_fitness = self.fitness(best_solution)
        return best_solution, 1 / best_fitness
    
# Ant Colony Optimization for TSP
class ACO_TSP:
    def __init__(self, tsp_instance, num_ants=20, evaporation_rate=0.1, intensification_factor=2, alpha=1, beta=2):
        self.tsp = tsp_instance
        self.n = len(tsp_instance.cities)
        self.distance_matrix = tsp_instance.distance_matrix_list
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate 
        self.intensification_factor = intensification_factor 
        self.alpha = alpha  # Importance of pheromone levels
        self.beta = beta  # Importance of heuristic information
        self.pheromone_levels = [[1 for _ in range(self.n)] for _ in range(self.n)]

    def heuristic_information(self, i, j):
        """Calculate the heuristic information (inverse of distance) from city i to city j."""
        if i != j:
            return 1 / self.distance_matrix[i][j]
        return 0

    def transition_probability(self, remaining_cities, current_city, next_city, ant_index):
        """Calculate the transition probability of moving from the current city to the next city."""
        numerator = (self.pheromone_levels[current_city][next_city] ** self.alpha) * (self.heuristic_information(current_city, next_city) ** self.beta)
        denominator = sum((self.pheromone_levels[current_city][k] ** self.alpha) * (self.heuristic_information(current_city, k) ** self.beta) for k in remaining_cities)
        return numerator / denominator if denominator != 0 else 0

    def construct_solution(self, ant_index):
        """Construct a solution using the probabilistic transition rules."""
        solution = [random.choice(range(self.n))]  # Start with a random city
        remaining_cities = set(range(self.n)) - {solution[0]}

        while remaining_cities:
            current_city = solution[-1]
            probabilities = [self.transition_probability(remaining_cities, current_city, next_city, ant_index) for next_city in remaining_cities]
            total_probability = sum(probabilities)
            probabilities = [prob / total_probability if total_probability != 0 else 0 for prob in probabilities]
            next_city = random.choices(list(remaining_cities), probabilities)[0]
            solution.append(next_city)
            remaining_cities.remove(next_city)

        return solution

    def update_pheromone_levels(self, solutions, fitness_values):
        """Update the pheromone levels on the paths based on the solutions found."""
        best_solution_index = fitness_values.index(min(fitness_values))
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone_levels[i][j] = (1 - self.evaporation_rate) * self.pheromone_levels[i][j]
                if j in solutions[best_solution_index]:
                    self.pheromone_levels[i][j] += self.intensification_factor / fitness_values[best_solution_index]

    def fitness(self, solution):
        """Evaluate the fitness of a solution."""
        total_distance = sum(self.distance_matrix[solution[i]][solution[i+1]] for i in range(self.n - 1))
        total_distance += self.distance_matrix[solution[-1]][solution[0]]  # Add the distance from the last city to the first city
        return total_distance

    def solve(self, generations=100):
        """Solve the problem using the ant colony optimization algorithm."""
        best_solution = None
        best_fitness = float('inf')

        for _ in range(generations):
            solutions = [self.construct_solution(i) for i in range(self.num_ants)]
            fitness_values = [self.fitness(sol) for sol in solutions]
            self.update_pheromone_levels(solutions, fitness_values)

            generation_best_solution = solutions[fitness_values.index(min(fitness_values))]
            generation_best_fitness = min(fitness_values)

            if generation_best_fitness < best_fitness:
                best_solution = generation_best_solution
                best_fitness = generation_best_fitness

        return best_solution, best_fitness
    
# Local Iterative Search algorithm for knapsack problem
class LIS_knapsack(LS_knapsack):
    def __init__(self, knapsack, max_iterations=100):
        super().__init__(knapsack)
        self.max_iterations = max_iterations

    def solve(self):
        """Implement the local iterative search algorithm starting with a random solution and performing a local search in each iteration."""
        best_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution
        best_value = self.evaluate(best_solution)

        for _ in range(self.max_iterations):
            # Generate a random solution as the starting point of the local search
            current_solution = [random.randint(0, 1) for _ in range(self.n)]
            local_search = LS_knapsack(self.knapsack)
            # Perform a local search starting from the current solution
            current_solution, current_value = local_search.solve()

            # If the new solution is better than the best solution found so far, update the best solution
            if current_value > best_value:
                best_solution = current_solution
                best_value = current_value

        return best_solution, best_value
    
# TabuSearch algorithm for knapsack problem
class TS_knapsack:
    def __init__(self, knapsack, tabu_list_size=5):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity
        self.tabu_list_size = tabu_list_size
        self.tabu_list = []

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by flipping each bit."""
        neighbors = []
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate a solution with penalty for infeasible solutions."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def solve(self, max_iterations=100):
        """Implement the tabu search algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution
        current_value = self.evaluate(current_solution)

        best_solution = current_solution
        best_value = current_value

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            # Finding the best neighbor that is not in the tabu list
            best_neighbor_value = -1
            best_neighbor = None
            for neighbor, neighbor_value in zip(neighbors, neighbors_values):
                if neighbor_value > best_neighbor_value and neighbor not in self.tabu_list:
                    best_neighbor_value = neighbor_value
                    best_neighbor = neighbor

            if best_neighbor is None:  # If all neighbors are in the tabu list, stop the search
                break

            current_solution = best_neighbor
            current_value = best_neighbor_value

            # Update the tabu list
            self.tabu_list.append(current_solution)
            if len(self.tabu_list) > self.tabu_list_size:
                self.tabu_list.pop(0)

            # Update the best solution found so far
            if current_value > best_value:
                best_solution = current_solution
                best_value = current_value

        return best_solution, best_value

# Hill Climbing algorithm for knapsack problem
class HC_knapsack:
    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by flipping each bit."""
        neighbors = []
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate a solution with penalty for infeasible solutions."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def solve(self, max_iterations=100):
        """Implement the hill climbing algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = max(neighbors_values)
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            if best_neighbor_value <= current_value:
                break  # If no improvement is found, stop the search

            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value
    
# Steepest Ascent Hill Climbing algorithm for knapsack problem
class HCsa_knapsack:
    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by flipping each bit."""
        neighbors = []
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate a solution with penalty for infeasible solutions."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def solve(self, max_iterations=100):
        """Implement the steepest ascent hill climbing algorithm starting with a random solution and moving to the best neighbor at each step."""
        current_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = max(neighbors_values)
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            # In this variation, we move to the best neighbor even if it doesn't improve the solution
            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value

# Random Restart Hill Climbing algorithm for knapsack problem 
class HCrr_knapsack:
    def __init__(self, knapsack):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity

    def generate_neighbors(self, solution):
        """Generate all possible neighbors of the current solution by flipping each bit."""
        neighbors = []
        for i in range(self.n):
            neighbor = solution.copy()
            neighbor[i] = 1 - neighbor[i]
            neighbors.append(neighbor)
        return neighbors

    def evaluate(self, solution):
        """Evaluate a solution with penalty for infeasible solutions."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def hill_climbing(self, initial_solution, max_iterations=100):
        """Implement the basic hill climbing algorithm starting with a given initial solution."""
        current_solution = initial_solution
        current_value = self.evaluate(current_solution)

        for _ in range(max_iterations):
            
            neighbors = self.generate_neighbors(current_solution)
            neighbors_values = [self.evaluate(neighbor) for neighbor in neighbors]

            best_neighbor_value = max(neighbors_values)
            best_neighbor = neighbors[neighbors_values.index(best_neighbor_value)]

            if best_neighbor_value <= current_value:
                break  # If no improvement is found, stop the search

            current_solution = best_neighbor
            current_value = best_neighbor_value

        return current_solution, current_value

    def solve(self, restarts=10, max_iterations=100):
        """Implement the random restart hill climbing algorithm with a specified number of restarts."""
        best_solution = None
        best_value = -1

        for _ in range(restarts):
            initial_solution = [random.randint(0, 1) for _ in range(self.n)]  # Random initial solution for each restart
            solution, value = self.hill_climbing(initial_solution, max_iterations)
            
            if value > best_value:
                best_solution = solution
                best_value = value

        return best_solution, best_value

# Genetic Algorithm for knapsack problem   
class GA_knapsack:
    def __init__(self, knapsack, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self):
        """Initialize the population with random solutions."""
        return [[random.randint(0, 1) for _ in range(self.n)] for _ in range(self.population_size)]

    def fitness(self, solution):
        """Evaluate the fitness of a solution."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def selection(self, population):
        """Select individuals from the population based on their fitness."""
        fitnesses = [self.fitness(ind) for ind in population]
        total_fitness = sum(fitnesses)
        probabilities = [fit / total_fitness for fit in fitnesses]
        selected_individuals = random.choices(population, probabilities, k=self.population_size)
        return selected_individuals

    def crossover(self, parent1, parent2):
        """Perform crossover between two parents to produce two offspring."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, self.n - 1)
            offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
            offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        else:
            offspring1, offspring2 = parent1, parent2

        return offspring1, offspring2

    def mutation(self, individual):
        """Perform mutation on an individual with a certain probability."""
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def evolve_population(self, population):
        """Evolve the population through selection, crossover and mutation."""
        new_population = []
        selected_individuals = self.selection(population)
        for i in range(0, self.population_size, 2):
            parent1 = selected_individuals[i]
            parent2 = selected_individuals[i+1]
            offspring1, offspring2 = self.crossover(parent1, parent2)
            new_population.append(self.mutation(offspring1))
            new_population.append(self.mutation(offspring2))
        return new_population

    def solve(self, generations=100):
        """Solve the problem using the genetic algorithm."""
        population = self.initialize_population()
        best_solution = max(population, key=self.fitness)
        best_fitness = self.fitness(best_solution)

        for _ in range(generations):
            population = self.evolve_population(population)
            current_best_solution = max(population, key=self.fitness)
            current_best_fitness = self.fitness(current_best_solution)

            if current_best_fitness > best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness

        return best_solution, best_fitness 
    
# Ant Colony Optimization for knapsack problem
class ACO_knapsack:
    def __init__(self, knapsack, num_ants=20, evaporation_rate=0.1, intensification_factor=2, alpha=1, beta=2):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate 
        self.intensification_factor = intensification_factor 
        self.alpha = alpha  # Importance of pheromone levels
        self.beta = beta  # Importance of heuristic information
        self.pheromone_levels = [[1 for _ in range(self.n)] for _ in range(self.num_ants)]

    def heuristic_information(self, solution, item_index):
        """Calculate the heuristic information (value-to-weight ratio) of adding an item to the solution."""
        if solution[item_index] == 0 and sum(w if s else 0 for w, s in zip(self.weights, solution)) + self.weights[item_index] <= self.capacity:
            return self.values[item_index] / self.weights[item_index]
        return 0

    def transition_probability(self, solution, item_index, ant_index):
        """Calculate the transition probability of adding an item to the solution."""
        numerator = (self.pheromone_levels[ant_index][item_index] ** self.alpha) * (self.heuristic_information(solution, item_index) ** self.beta)
        denominator = sum((self.pheromone_levels[ant_index][j] ** self.alpha) * (self.heuristic_information(solution, j) ** self.beta) for j in range(self.n))
        return numerator / denominator if denominator != 0 else 0

    def construct_solution(self, ant_index):
        """Construct a solution using the probabilistic transition rules."""
        solution = [0] * self.n
        for _ in range(self.n):
            probabilities = [self.transition_probability(solution, i, ant_index) for i in range(self.n)]
            total_probability = sum(probabilities)
            probabilities = [prob / total_probability if total_probability != 0 else 0 for prob in probabilities]
            item_index = random.choices(range(self.n), probabilities)[0]
            solution[item_index] = 1 if self.heuristic_information(solution, item_index) > 0 else 0
        return solution

    def update_pheromone_levels(self, solutions, fitness_values):
        """Update the pheromone levels on the paths based on the solutions found."""
        best_solution_index = fitness_values.index(max(fitness_values))
        for i in range(self.n):
            for j in range(self.num_ants):
                self.pheromone_levels[j][i] = (1 - self.evaporation_rate) * self.pheromone_levels[j][i]
                if solutions[best_solution_index][i] == 1:
                    self.pheromone_levels[j][i] += self.intensification_factor / fitness_values[best_solution_index]

    def fitness(self, solution):
        """Evaluate the fitness of a solution."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            return 0  # The solution is not feasible

        return total_value

    def solve(self, generations=100):
        """Solve the problem using the ant colony optimization algorithm."""
        best_solution = None
        best_fitness = 0

        for _ in range(generations):
            solutions = [self.construct_solution(i) for i in range(self.num_ants)]
            fitness_values = [self.fitness(sol) for sol in solutions]
            self.update_pheromone_levels(solutions, fitness_values)

            generation_best_solution = solutions[fitness_values.index(max(fitness_values))]
            generation_best_fitness = max(fitness_values)

            if generation_best_fitness > best_fitness:
                best_solution = generation_best_solution
                best_fitness = generation_best_fitness

        return best_solution, best_fitness
    
# Simulated Annealing algorithm for knapsack problem
class SA_knapsack:
    def __init__(self, knapsack, initial_temperature=100, cooling_schedule='linear', alpha=0.95, delta_t=1, lambda_val=0.01, beta=0.1):
        self.knapsack = knapsack
        self.n = knapsack.n
        self.weights = knapsack.weights
        self.values = knapsack.values
        self.capacity = knapsack.capacity
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.cooling_schedule = cooling_schedule
        self.alpha = alpha
        self.delta_t = delta_t
        self.lambda_val = lambda_val
        self.beta = beta

    def initialize_solution(self):
        """Initialize a random solution."""
        return [random.randint(0, 1) for _ in range(self.n)]

    def neighbor_solution(self, solution):
        """Get a neighbor solution by flipping a random bit."""
        neighbor_solution = solution.copy()
        flip_index = random.randint(0, self.n - 1)
        neighbor_solution[flip_index] = 1 - neighbor_solution[flip_index]
        return neighbor_solution

    def fitness(self, solution):
        """Evaluate the fitness of a solution."""
        total_value = sum(val if sel else 0 for val, sel in zip(self.values, solution))
        total_weight = sum(wt if sel else 0 for wt, sel in zip(self.weights, solution))

        if total_weight > self.capacity:
            penalty = 1000  # Penalty for exceeding the capacity
            total_value -= penalty

        return total_value

    def cool_down(self, time_step):
        """Cool down the temperature according to the selected schedule."""
        if self.cooling_schedule == 'linear':
            self.temperature -= self.delta_t
        elif self.cooling_schedule == 'geometric':
            self.temperature *= self.alpha
        elif self.cooling_schedule == 'exponential':
            self.temperature = self.initial_temperature * math.exp(-self.lambda_val * time_step)
        elif self.cooling_schedule == 'lundy_and_mees':
            self.temperature = self.temperature / (1 + self.beta * self.temperature)
        elif self.cooling_schedule == 'logarithmic':
            self.temperature = self.initial_temperature / (1 + math.log(1 + time_step))

    def solve(self, generations=100):
        """Solve the problem using the simulated annealing algorithm."""
        current_solution = self.initialize_solution()
        current_fitness = self.fitness(current_solution)
        best_solution = current_solution
        best_fitness = current_fitness

        for t in range(generations):
            self.cool_down(t)
            if self.temperature <= 0:
                break
            
            neighbor_sol = self.neighbor_solution(current_solution)
            neighbor_fitness = self.fitness(neighbor_sol)

            if neighbor_fitness > current_fitness or random.random() < math.exp((neighbor_fitness - current_fitness) / self.temperature):
                current_solution = neighbor_sol
                current_fitness = neighbor_fitness

            if current_fitness > best_fitness:
                best_solution = current_solution
                best_fitness = current_fitness

        return best_solution, best_fitness