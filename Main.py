from Simulated_annealing import simulated_annealing
from PDP import PDP, load_pdp_from_file, generate_problem
import pickle
import time

n_runs = 1
best_scores = []

dataset = load_pdp_from_file("data/pdp_20/pdp20_TEST1_seed1234.pkl")
for i in range(1, 50):

    pdp = generate_problem(size=50)
    #pdp = dataset[i-1]

    tic = time.time()
    best_solution, best_score = simulated_annealing(pdp)
    toc = time.time()
    print(f"{best_score}")
    best_scores.append(best_score)

print("Avg: ", sum(best_scores) / len(best_scores))
