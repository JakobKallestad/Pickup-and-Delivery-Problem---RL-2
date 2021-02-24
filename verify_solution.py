from Operators import objective_function
from PDP import PDP, load_pdp_from_file
from Simulated_annealing import simulated_annealing

def double_to_single(sol):
    new_sol = []
    memory = set()
    for e in sol[0]:
        if e in memory:
            new_sol.append(e*2)
        else:
            new_sol.append(e*2-1)
        memory.add(e)
    return new_sol

solution_list = [  # TEST1_seed1234.pkl solutions found by original algo from INF273 course.
    (3.5891863748332993, [[9, 7, 9, 3, 2, 2, 4, 5, 3, 6, 6, 8, 5, 8, 10, 7, 1, 4, 10, 1], []]),
    (5.150441003745926, [[2, 1, 2, 1, 9, 7, 5, 4, 4, 7, 5, 9, 8, 10, 10, 3, 8, 6, 6, 3], []]),
    (5.7705274165319915, [[6, 10, 9, 10, 1, 5, 6, 9, 1, 8, 4, 5, 8, 3, 4, 3, 2, 7, 7, 2], []]),
    (5.642166840534614, [[3, 8, 8, 3, 1, 4, 4, 10, 9, 9, 1, 5, 5, 2, 7, 7, 6, 6, 10, 2], []]),
    (6.1866800607254975, [[10, 1, 8, 1, 7, 8, 5, 7, 10, 5, 6, 9, 4, 6, 4, 2, 9, 3, 2, 3], []]),
    (3.7625980202693623, [[6, 10, 10, 2, 9, 8, 3, 6, 9, 5, 8, 5, 4, 7, 7, 2, 4, 1, 3, 1], []]),
    (5.640958938539784, [[10, 10, 4, 7, 7, 5, 4, 6, 5, 2, 6, 3, 2, 8, 9, 9, 3, 8, 1, 1], []]),
    (5.4890481193765925, [[9, 9, 2, 1, 2, 7, 3, 7, 8, 4, 1, 3, 8, 5, 10, 6, 6, 10, 4, 5], []]),
    (5.14983096825718, [[9, 3, 3, 2, 9, 7, 1, 2, 7, 4, 10, 1, 10, 4, 6, 8, 8, 5, 5, 6], []])
]
dataset = load_pdp_from_file("data/pdp_20/pdp20_TEST1_seed1234.pkl")
for i in range(9):
    pdp = dataset[i]

    best_solution, min_cost = None, float('inf')
    solution, cost = simulated_annealing(pdp)
    if cost < min_cost:
        min_cost = cost
        best_solution = solution

    new_solution = double_to_single(solution_list[i][1])
    new_cost = objective_function(pdp, new_solution)
    print(best_solution, min_cost)
    print(new_solution, new_cost, " reported: ", solution_list[i][0])
    print()
