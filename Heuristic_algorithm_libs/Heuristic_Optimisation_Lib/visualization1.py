from problems import Knapsack, TSP
from algorithms import*
import matplotlib.pyplot as plt

# use dp case in TSP

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

# tsp_instance1 = TSP(cities_coordinates=cities_data, distance_method='G',seed=9999)
# tsp_instance1.matrix()

# dp_test1 = DP_TSP(tsp_instance1)
# dp_test1.solve()

# tabu_test1 = TS_TSP(tsp_instance1,3)
# tabu_test1.solve()



# #----------------------------------------------

# tsp_instance2 = TSP(num_cities=15, distance_method='E',seed=9999)
# tsp_instance2.matrix()

# dp_test2 = DP_TSP(tsp_instance2)
# dp_test2.solve()


# tabu_test2 = TS_TSP(tsp_instance2,3)
# tabu_test2.solve()



from problems import Knapsack, TSP
from algorithms import *
import timeit

def comparative_analysis(seeds):
    results = {
        "DP_TSP": {"time": [], "distance": []},
        "TS_TSP": {"time": [], "distance": []}
    }
    
    for seed in seeds:
        tsp_instance1 = TSP(cities_coordinates=cities_data, distance_method='G', seed=seed)
        tsp_instance1.matrix()

        dp_test1 = DP_TSP(tsp_instance1)
        start_time = timeit.default_timer()
        dp_solution1 = dp_test1.solve()
        results["DP_TSP"]["time"].append(timeit.default_timer() - start_time)
        results["DP_TSP"]["distance"].append(dp_solution1[1])

        tabu_test1 = TS_TSP(tsp_instance1, 3)
        start_time = timeit.default_timer()
        tabu_solution1 = tabu_test1.solve()
        results["TS_TSP"]["time"].append(timeit.default_timer() - start_time)
        results["TS_TSP"]["distance"].append(tabu_solution1[1])


    return results

# Example usage:
seeds = [1001, 99,42,6, 103,1222,45, 404,758, 205,15,87,99,352,48]  # List of random seeds for testing
results = comparative_analysis(seeds)
print(results)



data = results

# Create a new figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the distances as a bar plot
ax1.bar(range(len(data['DP_TSP']['distance'])), data['DP_TSP']['distance'], alpha=0.8, label='DP_TSP Distance')
ax1.bar(range(len(data['TS_TSP']['distance'])), data['TS_TSP']['distance'], alpha=0.6, label='TS_TSP Distance')

# Create a second y-axis to plot the time
ax2 = ax1.twinx()
ax2.fill_between(range(len(data['DP_TSP']['time'])), data['DP_TSP']['time'], color='skyblue', alpha=0.4, label='DP_TSP Time')
ax2.fill_between(range(len(data['TS_TSP']['time'])), data['TS_TSP']['time'], color='red', alpha=0.4, label='TS_TSP Time')

# Adding labels and title
ax1.set_xlabel('Seed ID')
ax1.set_ylabel('Distance', color='blue')
ax2.set_ylabel('Time (s)', color='green')
plt.title('Comparative Analysis of DP_TSP and TS_TSP Algorithms')

# Setting x-axis ticks and labels to represent seed IDs
ax1.set_xticks(range(len(seeds))) 
ax1.set_xticklabels(seeds, rotation=45)  # Rotation for better visibility

# Adding a legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.show()
