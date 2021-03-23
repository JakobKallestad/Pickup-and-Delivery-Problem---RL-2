import time
import math
import random
from Utils import objective_function
from Operators import remove_insert, insert_first, insert_greedy, insert_beam_search, insert_tour, \
    remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors, remove_xs, remove_s, \
    remove_m, remove_l, remove_xl


def simulated_annealing(pdp, writer=None, instance_num=0):

    # constants:
    time_limit = 100000
    n_iterations = 1000
    T_0 = 5 #3000000
    alpha = 0.996 #0.9995
    segment_size = 20
    memory = set()
    sigma_1 = 5
    sigma_2 = 2
    sigma_3 = 1
    reaction_factor = 0.3

    # Operators and logic
    remove_operators = [remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,
                        remove_xs, remove_s, remove_m, remove_l, remove_xl]
    insert_operators = [insert_greedy, insert_beam_search, insert_tour, insert_first]  # insert_first

    len_r_op = len(remove_operators)
    len_i_op = len(insert_operators)

    operators = [(remove_operators[i // len_i_op], insert_operators[i % len_i_op]) for i in range(len_i_op-1, len_r_op*len_i_op)]
    len_operators = len(operators)
    operators_probs = [1/len_operators for _ in range(len_operators)]
    operators_counts = [0] * len_operators
    operators_scores = [0.0] * len_operators

    operator_names = {
        i: f"{i:02d}" + ":    " + '   &   '.join((str(a).split()[1], str(b).split()[1]))
        for i, (a, b) in enumerate(operators)
    }
    operator_names[0] = "00:    remove_and_insert_single_best"


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
            #print("total iterations:", i)
            break

        # Choose operators:
        op_ind = random.choices(range(len_operators), weights=operators_probs)[0]
        op = operators[op_ind]

        # Remove and Insert:
        new_solution, new_cost = remove_insert(pdp, solution, op)

        operators_counts[op_ind] += 1
        new_solution_id = str(new_solution)
        d_E = new_cost - cost

        # monitoring
        if writer:
            writer.add_scalars(f"cost_{instance_num}", {
                "incumbent": cost,
                "best_cost": best_cost
            }, i)
            if d_E > 0:
                writer.add_scalar(f"accept_prob_{instance_num}", math.e ** (-d_E / T), i)
                writer.add_scalar(f"diff_{instance_num}", d_E, i)

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

        operators_scores[op_ind] += reward
        T *= alpha
        memory.add(new_solution_id)

        # UPDATE weights
        if i % segment_size == 0:  # testing fixed probabilities

            for j in range(len_operators):
                if operators_counts[j] == 0:
                    #operators_counts[j] = 1
                    continue
                operators_probs[j] = operators_probs[j] * (1 - reaction_factor) + reaction_factor * (operators_scores[j] / operators_counts[j])
            operators_counts = [0] * len_operators
            operators_scores = [0.0] * len_operators
            sum_operator_probs = sum(operators_probs)
            operators_probs = [prob / sum_operator_probs for prob in operators_probs]

            if writer:
                writer.add_scalars(f"action_probs_{instance_num}", {
                    operator_names[ind]: e for ind, e in enumerate(operators_probs)
                }, i)  # tensorboard

    # Finished
    return best_solution, best_cost
