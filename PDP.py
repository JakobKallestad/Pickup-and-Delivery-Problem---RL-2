import numpy as np
from Utils import distance
import pickle

CAPACITY_MAP = {
    10: 10,  # 20,
    20: 15,  # 30,
    50: 20,  # 40,
    100: 25,  # 50
}


class PDP:
    def __init__(self, size, n_calls, locations, capacities, calls, dist_matrix):
        self.size = size
        self.n_calls = n_calls
        self.locations = locations
        self.capacities = capacities
        self.calls = calls
        self.dist_matrix = dist_matrix

    def save_problem(self):
        pass


def generate_problem(size=20):
    locations = np.random.uniform(size=(size+1, 2))  # location 0 is depot
    capacities = np.random.randint(1, 10, size=(size // 2)).repeat(2) / CAPACITY_MAP.get(size)
    capacities[1::2] *= -1

    n_calls = size // 2
    calls = [[i, i + 1] for i in range(n_calls+1)]
    dist_matrix = np.empty((size+1, size+1), dtype=np.float)
    for i in range(size+1):
        for j in range(i, size+1):
            d = distance(locations[i], locations[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    pdp = PDP(size, n_calls, locations, capacities, calls, dist_matrix)
    pdp.save_problem()
    return pdp


def load_pdp_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

