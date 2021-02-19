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


def remove_xs(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(2, min(n_calls, 20))))  # min(n_calls, 5)
    return remove_calls(solution, r_calls)


def remove_s(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(2, min(n_calls, 20))))
    return remove_calls(solution, r_calls)


def remove_m(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(2, min(n_calls, 20))))
    return remove_calls(solution, r_calls)


def remove_l(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(2, min(n_calls, 20))))
    return remove_calls(solution, r_calls)


def remove_xl(solution, n_calls):
    r_calls = set(random.sample(range(1, n_calls + 1), random.randint(2, min(n_calls, 20))))
    return remove_calls(solution, r_calls)


# Insert operators
# ==================================================================================================================

def insert_1(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)  # 18.02.21, 14:21 TODO
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost

def insert_2(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)  # 18.02.21, 14:21 TODO
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_3(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)  # 18.02.21, 14:21 TODO
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost


def insert_4(pdp, solution, removed_calls):
    for i_call in removed_calls:
        ij, _ = check_best_position_change(pdp, solution, i_call)  # 18.02.21, 14:21 TODO
        solution.insert(ij[0], i_call*2-1)
        solution.insert(ij[1], i_call*2)
    cost = objective_function(pdp, solution)
    return solution, cost

