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
        self.initialize_close_calls()

    def save_problem(self):
        pass

    def initialize_close_calls(self):
        self.distances = self.calculate_close_calls()

    def calculate_close_calls(self):
        max_sim_size = 21
        distances = [None] * max_sim_size
        prev_dists = list(zip([frozenset([i]) for i in range(1, self.n_calls + 1)], [0] * self.n_calls))
        for q in range(2, max_sim_size):
            new_dists = self.calc_new_dists(prev_dists)
            new_dists = sorted(new_dists, key=lambda x: x[1])  # memory limiting factor
            distances[q] = new_dists[:200]
            prev_dists = new_dists[:1000]  # beam width
        return distances

    def calc_new_dists(self, prev_dists):
        new_dists = []
        memory = set()
        for indexes, dist in prev_dists:
            for i in set(range(1, self.n_calls + 1)) - indexes:
                expanded_indexes = frozenset([*indexes, i])
                if expanded_indexes in memory:
                    continue
                memory.add(expanded_indexes)
                total = dist
                for ind in indexes:
                    total += (self.dist_matrix[self.calls[i][0], self.calls[ind][0]] +
                              self.dist_matrix[self.calls[i][1], self.calls[ind][1]])
                new_dists.append([expanded_indexes, total])
        return new_dists


def generate_problem(size=20):
    locations = np.random.uniform(size=(size+1, 2))  # location 0 is depot
    capacities = np.random.randint(1, 10, size=(size // 2)).repeat(2) / CAPACITY_MAP.get(size)
    capacities[1::2] *= -1

    n_calls = size // 2
    calls = [(None, None)] + [(i, i + 1) for i in range(1, size, 2)]
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

