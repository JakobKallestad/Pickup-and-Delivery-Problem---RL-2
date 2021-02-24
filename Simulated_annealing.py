import time
import math
import random
from Utils import objective_function
from Operators import remove_insert, insert_first, insert_greedy, insert_semi_greedy, insert_beam_search, insert_tour, \
    remove_single_best, remove_double_best, remove_longest_tour_deviation, remove_tour_neighbors, remove_xs, remove_s, \
    remove_m, remove_l, remove_xl


def simulated_annealing(pdp):

    # constants:
    time_limit = 10
    n_iterations = 100000
    T_0 = 3000000
    alpha = 0.9995
    segment_size = 100
    memory = set()
    sigma_1 = 5
    sigma_2 = 2
    sigma_3 = 1
    reaction_factor = 0.5


    # Operators and logic
    remove_operators = [remove_single_best, remove_double_best, remove_longest_tour_deviation, remove_tour_neighbors,
                        remove_xs, remove_s, remove_m, remove_l, remove_xl]
    remove_probs = [0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111, 0.111111]
    remove_indices = list(range(len(remove_operators)))
    remove_operator_scores = [0] * len(remove_operators)
    remove_operator_count = [0] * len(remove_operators)
    # --
    insert_operators = [ insert_first, insert_greedy, insert_semi_greedy, insert_beam_search, insert_tour]
    insert_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    insert_indices = list(range(len(insert_operators)))
    insert_operators_scores = [0] * len(insert_operators)
    insert_operators_count = [0] * len(insert_operators)


    # variables
    solution = [i for i in range(1, pdp.size+1)]
    cost = objective_function(pdp, solution)
    best_solution = solution
    best_cost = cost
    T = T_0


    # main loop
    start = time.time()
    for i in range(1, n_iterations):
        if time.time()-start > (time_limit-1):
            print("total iterations:", i)
            break

        # Choose operators:
        remove_op_ind = random.choices(remove_indices, weights=remove_probs)[0]
        remove_op = remove_operators[remove_op_ind]
        insert_op_ind = random.choices(insert_indices, weights=insert_probs)[0]
        insert_op = insert_operators[insert_op_ind]
        op = (remove_op, insert_op)

        # Remove and Insert:
        new_solution, new_cost = remove_insert(pdp, solution, op)

        remove_operator_count[remove_op_ind] += 1
        insert_operators_count[insert_op_ind] += 1
        new_solution_id = str(new_solution)
        d_E = new_cost - cost
        reward = 0
        if d_E < 0:
            solution = new_solution
            cost = new_cost
            if cost < best_cost:
                best_solution = solution
                best_cost = cost
                reward = sigma_1
                #print("New best --> ", best_cost, ", iteration --> ", i)
            elif new_solution_id not in memory:
                reward = sigma_2
        elif new_cost < float('inf') and random.random() < math.e ** (-d_E / T):
            solution = new_solution
            cost = new_cost
            if new_solution_id not in memory:
                reward = sigma_3

        remove_operator_scores[remove_op_ind] += reward
        insert_operators_scores[insert_op_ind] += reward
        T *= alpha
        memory.add(new_solution_id)

        # UPDATE weights
        if i % segment_size == 0:
            for j in range(len(remove_operators)):
                if remove_operator_count[j] == 0:
                    remove_operator_count[j] = float('inf')
                remove_probs[j] = remove_probs[j] * (1 - reaction_factor) + reaction_factor * (remove_operator_scores[j] / remove_operator_count[j])
                remove_operator_count[j] = 0
                remove_operator_scores[j] = 0
            remove_probs = [ip / sum(remove_probs) for ip in remove_probs]
            for j in range(len(insert_operators)):
                if insert_operators_count[j] == 0:
                    insert_operators_count[j] = float('inf')
                insert_probs[j] = insert_probs[j] * (1 - reaction_factor) + reaction_factor * \
                                  (insert_operators_scores[j] / insert_operators_count[j])
                insert_operators_count[j] = 0
                insert_operators_scores[j] = 0
            insert_probs = [ip/sum(insert_probs) for ip in insert_probs]

    # Finished
    return best_solution, best_cost
