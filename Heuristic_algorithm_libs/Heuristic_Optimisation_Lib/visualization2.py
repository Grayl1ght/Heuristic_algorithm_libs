
from problems import Knapsack, TSP
from algorithms import*

# Initializing a list to store the test results for TSP class
tsp_test_results = []

# Test case 1: Creating an instance of TSP class and checking the initialization of attributes
try:
    tsp_instance1 = TSP(num_cities=5, distance_method='E')
    if (tsp_instance1.num_cities == 5 and 
        tsp_instance1.distance_method == 'E' and 
        tsp_instance1.matrix() is not None):
        tsp_test_results.append("TSP Test case 1: Passed")
    else:
        tsp_test_results.append("TSP Test case 1: Failed")
except Exception as e:
    tsp_test_results.append(f"TSP Test case 1: Error - {str(e)}")

# Test case 2: Testing the calculate_distance method
try:
    city1 = (0, 0)
    city2 = (3, 4)
    distance = tsp_instance1.calculate_distance(city1, city2)
    if distance == 5.0:
        tsp_test_results.append("TSP Test case 2: Passed")
    else:
        tsp_test_results.append("TSP Test case 2: Failed")
except Exception as e:
    tsp_test_results.append(f"TSP Test case 2: Error - {str(e)}")

# Test case 3: Testing the generate_initial_solution method
try:
    initial_solution = tsp_instance1.generate_initial_solution()
    if len(initial_solution) == 5:
        tsp_test_results.append("TSP Test case 3: Passed")
    else:
        tsp_test_results.append("TSP Test case 3: Failed")
except Exception as e:
    tsp_test_results.append(f"TSP Test case 3: Error - {str(e)}")

# Displaying the test results
# print("TSP Test Results:",tsp_test_results)

import unittest

# 定义测试类
class TestTSP(unittest.TestCase):

    def setUp(self):
        self.cities_data = [
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
        self.tsp_instance = TSP(cities_coordinates=self.cities_data, distance_method='G')

 
    def test_generate_cities(self):
        self.tsp_instance.generate_cities(5)
        self.assertEqual(len(self.tsp_instance.cities), 5)


    def test_Euclidean_distance(self):
        city1 = (0, 0)
        city2 = (3, 4)
        self.assertEqual(self.tsp_instance.Euclidean_distance(city1, city2), 5.0)


    def test_great_circle_distance(self):
        lat1, lon1 = 16.47, 96.10
        lat2, lon2 = 16.47, 94.44
        self.assertTrue(isinstance(self.tsp_instance.great_circle_distance(lat1, lon1, lat2, lon2), int))

   
    def test_calculate_distance(self):
        city1 = (16.47, 96.10)
        city2 = (16.47, 94.44)
        self.assertTrue(isinstance(self.tsp_instance.calculate_distance(city1, city2), int))


    def test_matrix_and_matrix_to_dict(self):
        matrix_dict = self.tsp_instance.matrix()
        self.assertTrue(isinstance(matrix_dict, dict))
        self.assertEqual(len(matrix_dict), len(self.cities_data))


    def test_total_distance_direct(self):
        path = list(self.tsp_instance.cities.keys())
        self.assertTrue(isinstance(self.tsp_instance.total_distance_direct(path), (int, float)))


    def test_total_distance_matrix(self):
        path = list(self.tsp_instance.cities.keys())
        self.assertTrue(isinstance(self.tsp_instance.total_distance_matrix(path), (int, float)))




class TestTSP(unittest.TestCase):

    def setUp(self):
        self.cities_data = [
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
        self.tsp_instance = TSP(cities_coordinates=self.cities_data, distance_method='G')
        self.tsp_instance.matrix()  # Ensuring the matrix method is called to initialize distance_matrix_dict

    def test_generate_cities(self):
        self.tsp_instance.generate_cities(5)
        self.assertEqual(len(self.tsp_instance.cities), 5)

    def test_Euclidean_distance(self):
        city1 = (0, 0)
        city2 = (3, 4)
        self.assertEqual(self.tsp_instance.Euclidean_distance(city1, city2), 5.0)

    def test_great_circle_distance(self):
        lat1, lon1 = 16.47, 96.10
        lat2, lon2 = 16.47, 94.44
        self.assertTrue(isinstance(self.tsp_instance.great_circle_distance(lat1, lon1, lat2, lon2), int))

    def test_calculate_distance(self):
        city1 = (16.47, 96.10)
        city2 = (16.47, 94.44)
        self.assertTrue(isinstance(self.tsp_instance.calculate_distance(city1, city2), int))

    def test_matrix_and_matrix_to_dict(self):
        matrix_dict = self.tsp_instance.matrix()
        self.assertTrue(isinstance(matrix_dict, dict))
        self.assertEqual(len(matrix_dict), len(self.cities_data))

    def test_total_distance_direct(self):
        path = list(self.tsp_instance.cities.keys())
        self.assertTrue(isinstance(self.tsp_instance.total_distance_direct(path), (int, float)))

    def test_total_distance_matrix(self):
        path = list(self.tsp_instance.cities.keys())
        self.assertTrue(isinstance(self.tsp_instance.total_distance_matrix(path), (int, float)))


# test TSP class methods
#unittest.TextTestRunner().run(unittest.TestLoader().loadTestsFromTestCase(TestTSP)) 


# Creating knapsack problem instances
knapsack_instance_case1 = Knapsack(n=4, capacity=11, weights=[2, 4, 6, 7], values=[6, 10, 12, 13])
knapsack_instance_case2 = Knapsack(n=7, capacity=50, weights=[31, 10, 20, 19, 4, 3, 6], values=[70, 20, 39, 37, 7, 5, 10])

# Creating algorithm instances and solving knapsack problems
knapsack_algorithms = [DP_knapsack, LS_knapsack, TS_knapsack, GA_knapsack]
knapsack_results = []
for algo in knapsack_algorithms:
    knapsack_results.append([
        algo(knapsack=knapsack_instance_case1).solve(),
        algo(knapsack=knapsack_instance_case2).solve()
    ])
# print("Knapsack Results:", knapsack_results)

# Extending the list of algorithms to include the new ones
extended_knapsack_algorithms = [
    DP_knapsack, LS_knapsack, TS_knapsack, GA_knapsack,
    HC_knapsack, HCsa_knapsack, HCrr_knapsack, ACO_knapsack
]

# Extending the list of algorithm names to include the new ones
extended_knapsack_algorithm_names = [
    "DP_knapsack", "LS_knapsack", "TS_knapsack", "GA_knapsack",
    "HC_knapsack", "HCsa_knapsack", "HCrr_knapsack", "ACO_knapsack"
]

# Creating a dictionary to store the results with algorithm names as keys
extended_knapsack_results  = {}

# Creating algorithm instances and solving knapsack problems
for algo_name, algo in zip(extended_knapsack_algorithm_names, extended_knapsack_algorithms):
    extended_knapsack_results [algo_name] = [
        algo(knapsack=knapsack_instance_case1).solve(),
        algo(knapsack=knapsack_instance_case2).solve()
    ]
# 
# print("Extended Knapsack Results:", extended_knapsack_results )


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

tsp_instance1 = TSP(num_cities=14,cities_coordinates=cities_data, distance_method='G',seed=42)
tsp_instance1.matrix()



tsp_algorithm_names = ["DP_TSP", "TS_TSP", "SA_TSP","HC_TSP", "HCrr_TSP", "HCsa_TSP", "GA_TSP"]

tsp_algorithms = [DP_TSP, TS_TSP, SA_TSP, HC_TSP, HCrr_TSP, HCsa_TSP, GA_TSP]


# Creating a dictionary to store the results with algorithm names as keys
tsp_results  = {}

# Creating algorithm instances and solving TSP problems
for algo_name, algo in zip(tsp_algorithm_names, tsp_algorithms):
    tsp_results[algo_name] = [
        algo(tsp_instance1).solve()[1],
       
    ]

print(tsp_results )



