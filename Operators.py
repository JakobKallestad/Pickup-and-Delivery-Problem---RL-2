import random
import copy
import math
from Utils import objective_function, check_best_position_change


def remove_insert(pdp, solution, op):
    solution = copy.copy(solution)
    op, op2 = op
    solution, removed_calls = op(solution, pdp.n_calls)
    new_solution, new_cost = op2(pdp, solution, removed_calls)
    return new_solution, new_cost


# Remove operators
# ==================================================================================================================

def remove_calls(solution, r_calls):
    temp_solution = [e for e in solution if math.ceil(e/2) not in r_calls]
    return temp_solution, r_calls


def remove_single_best(solution, n_calls):
    return remove_calls(solution, [1])


def remove_double_best(solution, n_calls):
    return remove_calls(solution, [1, 2])


def remove_longest_tour_deviation(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 2), min(n_calls, 5))))
    return remove_calls(solution, r_calls)


def remove_tour_neighbors(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 2), min(n_calls, 5))))
    return remove_calls(solution, r_calls)


def remove_xs(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 2), min(n_calls, 5))))
    return remove_calls(solution, r_calls)


def remove_s(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 5), min(n_calls, 10))))
    return remove_calls(solution, r_calls)


def remove_m(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 10), min(n_calls, 20))))
    return remove_calls(solution, r_calls)


def remove_l(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 20), min(n_calls, 30))))
    return remove_calls(solution, r_calls)


def remove_xl(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(min(n_calls, 30), min(n_calls, 40))))
    return remove_calls(solution, r_calls)


# Insert operators
# ==================================================================================================================


def insert_first(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_greedy(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_semi_greedy(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_beam_search(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_tour(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)
        solution.insert(ij[0], i_call * 2 - 1)
        solution.insert(ij[1], i_call * 2)
    cost = objective_function(pdp, solution)
    return solution, cost
