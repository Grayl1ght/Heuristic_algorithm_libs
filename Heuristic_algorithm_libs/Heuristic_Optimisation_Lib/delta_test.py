# Import required libraries
import math
import random
import numpy as np
import timeit

class TSP :
    def __init__(self, num_cities=3, cities_coordinates=None, distance_method='E', seed=42):
        self.distance_method = distance_method
        self.seed = seed
        random.seed(self.seed)
        if cities_coordinates:
            if isinstance(cities_coordinates, dict):
                self.cities = cities_coordinates
                self.num_cities = len(self.cities)
            elif isinstance(cities_coordinates, list):
                self.cities = {i+1: coord for i, coord in enumerate(cities_coordinates)}
                self.num_cities = len(self.cities)
        else:
            self.num_cities = num_cities
            self.generate_cities(self.num_cities)
        self.distance_matrix_list = None
        self.distance_matrix_dict = None

    def generate_cities(self, num_cities, x_limit=100, y_limit=100):
        self.cities = {}
        for i in range(1, num_cities + 1):
            x = random.randint(0, x_limit)
            y = random.randint(0, y_limit)
            self.cities[i] = (x, y)

    def calculate_distance(self, city1, city2):
        if self.distance_method == 'E':
            return self.Euclidean_distance(city1, city2)
        elif self.distance_method == 'G':
            return self.great_circle_distance(*city1, *city2)

    def Euclidean_distance(self, city1, city2):
        x1, y1 = city1
        x2, y2 = city2
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def great_circle_distance(self, lat1, lon1, lat2, lon2):
        PI = 3.141592
        RRR = 6378.388

        deg = int(lat1)
        minute = lat1 - deg
        latitude1 = PI * (deg + 5.0 * minute / 3.0) / 180.0

        deg = int(lon1)
        minute = lon1 - deg
        longitude1 = PI * (deg + 5.0 * minute / 3.0) / 180.0

        deg = int(lat2)
        minute = lat2 - deg
        latitude2 = PI * (deg + 5.0 * minute / 3.0) / 180.0

        deg = int(lon2)
        minute = lon2 - deg
        longitude2 = PI * (deg + 5.0 * minute / 3.0) / 180.0

        q1 = math.cos(longitude1 - longitude2)
        q2 = math.cos(latitude1 - latitude2)
        q3 = math.cos(latitude1 + latitude2)

        dij = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
        return dij

    def matrix(self):
        num_cities = len(self.cities)
        self.distance_matrix_list = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(i+1, num_cities):
                dist = self.calculate_distance(self.cities[i+1], self.cities[j+1])
                self.distance_matrix_list[i, j] = dist
                self.distance_matrix_list[j, i] = dist
        self.distance_matrix_dict = self.matrix_to_dict()

    def matrix_to_dict(self):
        matrix_dict = {}
        for i, row in enumerate(self.distance_matrix_list):
            matrix_dict[i+1] = {}
            for j, value in enumerate(row):
                matrix_dict[i+1][j+1] = value
        return matrix_dict

    def generate_initial_solution(self):
        self.initial_solution = list(self.cities.keys())
        return self.initial_solution

    # Method 1: Directly calculate the total distance for the entire path
    def total_distance_direct(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.calculate_distance(self.cities[path[i]], self.cities[path[i+1]])
        total_distance += self.calculate_distance(self.cities[path[-1]], self.cities[path[0]])
        return total_distance

    # Method 2: Use the matrix to find the total distance
    def total_distance_matrix(self, path):
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.distance_matrix_dict[path[i]][path[i+1]]
        total_distance += self.distance_matrix_dict[path[-1]][path[0]]
        return total_distance

    # Method 3: Use delta evaluation to find the total distance
    def total_distance_delta(self, current_total_distance, path, swap_indices):
        i, j = swap_indices
        if i == 0:
            pre_i, post_j = path[-1], path[j+1]
        elif j == len(path) - 1:
            pre_i, post_j = path[i-1], path[0]
        else:
            pre_i, post_j = path[i-1], path[j+1]
        
        delta = (self.distance_matrix_dict[pre_i][path[j]] + self.distance_matrix_dict[path[i]][post_j]) - \
                (self.distance_matrix_dict[pre_i][path[i]] + self.distance_matrix_dict[path[j]][post_j])
        return current_total_distance + delta

# Adding the delta_eval method to the LocalSearch class
class LocalSearch:
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
# Testing the updated LocalSearch class with delta_eval and other methods
cities_data = [
    (16.47, 96.10),
    (16.47, 94.44),
    (20.09, 92.54),
    (22.39, 93.37),
    (25.23, 97.24),
    (22.00, 96.05),
    (20.47, 97.02),
    (17.20, 96.29),
    (16.30, 97.38),
    (14.05, 98.12),
    (16.53, 97.38),
    (21.52, 95.59),
    (19.41, 97.13),
    (20.09, 94.55)
]

tsp_instance_10 = TSP(cities_coordinates=cities_data, distance_method='G', seed=5)
tsp_instance_10.matrix()

# Initialize LocalSearch with the new TSP instance
ls = LocalSearch(tsp_instance_10)

# Generate an initial solution
initial_solution_10 = tsp_instance_10.generate_initial_solution()

# Perform optimization using different distance calculation methods
optimized_solution_delta_10 = ls.optimize(initial_solution_10, distance_calculation_method='delta')
optimized_solution_direct_10 = ls.optimize(initial_solution_10, distance_calculation_method='direct')
optimized_solution_matrix_10 = ls.optimize(initial_solution_10, distance_calculation_method='matrix')

# Calculate the total distance for each optimized solution
total_distance_delta_10 = tsp_instance_10.total_distance_direct(optimized_solution_delta_10)
total_distance_direct_10 = tsp_instance_10.total_distance_direct(optimized_solution_direct_10)
total_distance_matrix_10 = tsp_instance_10.total_distance_direct(optimized_solution_matrix_10)

import timeit

# Function to time the optimization process for each method
def time_optimization(seed, cities_data, method):
    tsp_instance = TSP(cities_coordinates=cities_data, distance_method='G', seed=seed)
    tsp_instance.matrix()
    ls = LocalSearch(tsp_instance)
    initial_solution = tsp_instance.generate_initial_solution()
    
    start_time = timeit.default_timer()
    _ = ls.optimize(initial_solution, distance_calculation_method=method)
    end_time = timeit.default_timer()
    
    return end_time - start_time

# Initialize variables
seeds = list(range(1, 31))
time_data = {'Seed': [], 'Delta_Time': [], 'Direct_Time': [], 'Matrix_Time': []}

# Perform the timing tests
for seed in seeds:
    time_data['Seed'].append(seed)
    time_data['Delta_Time'].append(time_optimization(seed, cities_data, 'delta'))
    time_data['Direct_Time'].append(time_optimization(seed, cities_data, 'direct'))
    time_data['Matrix_Time'].append(time_optimization(seed, cities_data, 'matrix'))


def display_time_efficiency_boxplot(time_data, title="Time Efficiency Comparison", y_label="Time (s)"):
        """
        Display the time efficiency of different methods as a boxplot using Matplotlib.
        
        Parameters:
            time_data (dict): The dictionary containing time efficiency data. Expected to be in the form {'Method': [time1, time2, ...]}.
            title (str): The title of the boxplot.
            y_label (str): The label for the y-axis.
        """
        # Extract method names and corresponding time data
        methods = list(time_data.keys())
        time_values = [time_data[method] for method in methods]
        
        # Create the boxplot
        plt.figure(figsize=(10, 6))
        plt.boxplot(time_values, labels=methods, showfliers=False)
        
        # Add titles and labels
        plt.title(title)
        plt.ylabel(y_label)
        
        # Show the plot
        plt.show()

# Prepare the time data for boxplot visualization
boxplot_time_data = {
                     'Direct_Time': time_data['Direct_Time'], 
                     'Matrix_Time': time_data['Matrix_Time'],
                     'Delta_Time': time_data['Delta_Time']
                     }

# Display the time efficiency comparison as a boxplot
display_time_efficiency_boxplot(boxplot_time_data, title="Time Efficiency Comparison for Different Methods", y_label="Time (s)")

