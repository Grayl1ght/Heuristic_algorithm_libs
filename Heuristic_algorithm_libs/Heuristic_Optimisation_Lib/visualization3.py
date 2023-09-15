from problems import TSP
from algorithms import DP_TSP, TS_TSP, SA_TSP, HC_TSP, HCrr_TSP, HCsa_TSP, GA_TSP, ACO_TSP

def test_algorithms(seeds):
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

    tsp_algorithm_names = ["DP_TSP", "TS_TSP", "SA_TSP", "HC_TSP", "HCrr_TSP", "HCsa_TSP", "GA_TSP", "ACO_TSP"]
    tsp_algorithms = [DP_TSP, TS_TSP, SA_TSP, HC_TSP, HCrr_TSP, HCsa_TSP, GA_TSP, ACO_TSP]

    results = {name: [] for name in tsp_algorithm_names}

    for seed in seeds:
        tsp_instance = TSP(num_cities=14, cities_coordinates=cities_data, distance_method='G', seed=seed)
        tsp_instance.matrix()

        for algo_name, algo in zip(tsp_algorithm_names, tsp_algorithms):
            distance = algo(tsp_instance).solve()[1]
            results[algo_name].append(distance)

    return results

# Usage:
seeds = [1001, 99, 42, 6, 103, 1222, 45, 404, 758, 205, 15, 87, 99, 352, 48]
results = test_algorithms(seeds)
print(results)


import matplotlib.pyplot as plt


data = results


plt.figure(figsize=(10, 6))


for algo_name, distances in data.items():
    plt.scatter([algo_name] * len(distances), distances)

plt.xlabel('Algorithms')
plt.ylabel('Distance')
plt.title('Distance Distribution Across Different Algorithms')
plt.xticks(rotation=45)
plt.grid(True)


plt.tight_layout()
# plt.show()


data_list = [data[algo_name] for algo_name in data]


plt.figure(figsize=(10, 6))
plt.boxplot(data_list, labels=data.keys())


plt.xlabel('Algorithms')
plt.ylabel('Distance')
plt.title('Distance Distribution Across Different Algorithms (Boxplot)')
plt.xticks(rotation=45)
plt.grid(True)


plt.tight_layout()
plt.show()
