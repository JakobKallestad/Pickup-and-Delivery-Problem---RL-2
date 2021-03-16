from Simulated_annealing import simulated_annealing
from PDP import PDP, load_pdp_from_file, generate_problem
import pickle
import time
import numpy as np
import random
from tensorboardX import SummaryWriter

SEED = 1234
np.random.seed(SEED)
random.seed(SEED)

n_runs = 1
best_scores = []
writer = SummaryWriter('logs_24')

dataset = load_pdp_from_file("data/pdp_100/seed1234_size100_num100.pkl")
for i in range(1, 100):

    #pdp = generate_problem(size=100)
    pdp = dataset[i-1]

    pdp.initialize_close_calls()

    tic = time.time()
    best_solution, best_score = simulated_annealing(pdp, writer=writer, instance_num=i) #, writer=writer)
    writer.add_scalar("best_cost", best_score, i)
    toc = time.time()
    print(f"{best_score}")
    best_scores.append(best_score)

print("Avg: ", sum(best_scores) / len(best_scores))
