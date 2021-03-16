from Simulated_annealing import simulated_annealing
from PDP import PDP, load_pdp_from_file, generate_problem
import pickle
import time
import numpy as np
import random
from tensorboardX import SummaryWriter

import multiprocessing as mp
from functools import partial


def solve_instance(i):
    pdp = generate_problem(size=100)
    #pdp = dataset[i-1]
    pdp.initialize_close_calls()

    tic = time.time()
    best_solution, best_score = simulated_annealing(pdp) #, writer=writer)
    toc = time.time()
    print(f"{best_score}")
    return best_score


SEED = 1234
np.random.seed(SEED)
random.seed(SEED)

n_runs = 1
best_scores = []
#writer = SummaryWriter('logs_4_Long_Run')

dataset = load_pdp_from_file("data/pdp_100/seed1234_size100_num100.pkl")

#func = partial(solve_instance, )
with mp.Pool() as pool:
    pool.map(solve_instance, range(1, 100))
