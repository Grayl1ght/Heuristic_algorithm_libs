import algorithms
import problems

KS_case_1 = problems.Knapsack.evaluate_solution(solution=[0, 0, 0, 0], weights=[2, 4, 6, 7], values=[6, 10, 12, 13], capacity=11)
# print(KS_case_1)


# Initializing the weights, values, and capacity
weights = [2, 4, 6, 7]
values = [6, 10, 12, 13]
capacity = 11

# Evaluating the given solutions
KS_case_1_solution1 = [0, 0, 1, 0]
KS_case_1_solution2 = [1, 1, 1, 1]
KS_case_1_solution3 = [1, 0, 1, 0]

evaluation1 = problems.Knapsack.evaluate_solution(solution=KS_case_1_solution1, weights=weights, values=values, capacity=capacity)
evaluation2 = problems.Knapsack.evaluate_solution(solution=KS_case_1_solution2, weights=weights, values=values, capacity=capacity)
evaluation3 = problems.Knapsack.evaluate_solution(solution=KS_case_1_solution3, weights=weights, values=values, capacity=capacity)

# print(f"Solution 1: {evaluation1}")
# print(f"Solution 2: {evaluation2}")
# print(f"Solution 3: {evaluation3}")



# Creating a Knapsack instance with given data
knapsack_instance_case1 = problems.Knapsack(n=4, capacity=11, weights=[2, 4, 6, 7], values=[6, 10, 12, 13])
knapsack_instance_case2 = problems.Knapsack(n=7, capacity=50, weights=[31, 10, 20, 19, 4, 3, 6], values=[70, 20, 39, 37, 7, 5, 10])


# Creating a DP instance with the knapsack instance
dp_instance1 = algorithms.DP_knapsack(knapsack=knapsack_instance_case1)
dp_instance2 = algorithms.DP_knapsack(knapsack=knapsack_instance_case2)
# print(dp_instance1.solve())
# print(dp_instance2.solve())

ls_instance1 = algorithms.LS_knapsack(knapsack=knapsack_instance_case1)
ls_instance2 = algorithms.LS_knapsack(knapsack=knapsack_instance_case2)
# print(ls_instance1.solve())
# print(ls_instance2.solve())

ls_instance1 = algorithms.LIS_knapsack(knapsack=knapsack_instance_case1, max_iterations=1000)
ls_instance2 = algorithms.LIS_knapsack(knapsack=knapsack_instance_case2, max_iterations=1000)
# print(ls_instance1.solve())
# print(ls_instance2.solve())

ts_instance1 = algorithms.TS_knapsack(knapsack=knapsack_instance_case1, tabu_list_size=5)
ts_instance2 = algorithms.TS_knapsack(knapsack=knapsack_instance_case2, tabu_list_size=5)

# print(ts_instance1.solve(max_iterations=1000))
# print(ts_instance2.solve())

#.......


# Creating a TSP instance with given data
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

tsp_instance1 = problems.TSP(cities_coordinates=cities_data, distance_method='G')
tsp_instance1.matrix()

tsp_instance2 = problems.TSP(6, distance_method='E')
tsp_instance2.matrix()

dp_instance1 = algorithms.DP_TSP(tsp_instance1)
dp_instance2 = algorithms.DP_TSP(tsp_instance2)

# print(dp_instance1.solve())
# print(dp_instance2.solve())
