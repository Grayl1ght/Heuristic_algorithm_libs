import math
import random
import numpy as np



class Knapsack:
    def __init__(self, n=None, capacity=None, weights=None, values=None):
        if n is None:
            self.random_init()
        else:
            self.n = n
            self.capacity = capacity
            self.weights = weights
            self.values = values
            self.selection = [0] * n  # Initializing the selection with all items not selected

    # Method to initialize with random data
    def random_init(self):
        self.n = 5
        self.capacity = random.randint(10, 100)
        self.weights = [random.randint(1, 20) for _ in range(self.n)]
        self.values = [random.randint(1, 100) for _ in range(self.n)]
        self.selection = [0] * self.n  # Initializing the selection with all items not selected

    # Method to return the representation of the knapsack selection
    def representation(self):
        return self.selection

    # Class method to evaluate a solution
    @classmethod
    def evaluate_solution(cls, solution, weights, values, capacity):
        total_value = 0
        total_weight = 0
        for i in range(len(solution)):
            if solution[i] == 1:
                total_value += values[i]
                total_weight += weights[i]

        feasibility = "Feasible" if total_weight <= capacity else "Infeasible" # Check if the solution is feasible
        
        return total_value, total_weight, feasibility

    def __str__(self):
        return f"Knapsack(n={self.n}, capacity={self.capacity}, weights={self.weights}, values={self.values}, selection={self.selection})"
    



class TSP :
    def __init__(self, num_cities=3, cities_coordinates=None, distance_method='E', seed=42):
        self.distance_method = distance_method
        self.seed = seed
        self.num_cities = num_cities
        random.seed(self.seed)
        if cities_coordinates:
            if isinstance(cities_coordinates, dict):
                self.cities = cities_coordinates
                num_cities = len(self.cities)
            elif isinstance(cities_coordinates, list):
                self.cities = {i+1: coord for i, coord in enumerate(cities_coordinates)}
                num_cities = len(self.cities)
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
        return self.distance_matrix_dict

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

