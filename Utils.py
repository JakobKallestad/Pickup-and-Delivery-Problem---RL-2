

# euclidean distance
def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


# Calculates cost and feasibility of the solution:
def objective_function(pdp, solution):
    cost = 0
    current_weight = 0
    current_node = 0
    visited = [False] * (pdp.size+1)
    for e in solution:
        if e % 2 == 0 and not visited[e-1]:
            return float('inf')  # infeasible due to delivery before pickup
        visited[e] = True
        cost += pdp.dist_matrix[current_node, e]
        current_weight += pdp.capacities[e-1]
        current_node = e
        if current_weight > 1:
            return float('inf')  # infeasible due to capacity constraint
    return cost


def check_best_position_change(pdp, solution, i_call):
    len_solution = len(solution)
    i_call_weight = pdp.capacities[(i_call*2)-2]

    if len_solution > 0:
        cumulative_weight = [0] + [None]*len(solution)
        for i, e in enumerate(solution, 1):
            cumulative_weight[i] = cumulative_weight[i-1] + pdp.capacities[e-1]
    best_ij = None
    best_cost = float('inf')
    for i in range(len_solution+1):

        # SIZE CHECK:
        if len_solution > 0 and cumulative_weight[i] + i_call_weight > 1:
            continue  # if inserting i_call here already makes the vehicle overloaded

        for j in range(i+1, len_solution+2):

            # SIZE CHECK:
            if len_solution > 0 and cumulative_weight[j - 1] + i_call_weight > 1:
                break  # Go to next i

            if i + 1 == j:
                if i == 0 and j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A1 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                elif j == len_solution + 1:
                    R1 = 0
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A1 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A2 = pdp.dist_matrix[pdp.calls[i_call][0], pdp.calls[i_call][1]]
                    A3 = pdp.dist_matrix[pdp.calls[i_call][1], solution[i]]
                add_cost = -R1 + A1 + A2 + A3
            else:
                if i == 0 and j == len_solution + 1:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                elif i == 0:
                    R1 = pdp.dist_matrix[0, solution[i]]
                    A11 = pdp.dist_matrix[0, pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                elif j == len_solution + 1:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = 0
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = 0
                else:
                    R1 = pdp.dist_matrix[solution[i-1], solution[i]]
                    A11 = pdp.dist_matrix[solution[i-1], pdp.calls[i_call][0]]
                    A12 = pdp.dist_matrix[pdp.calls[i_call][0], solution[i]]
                    R2 = pdp.dist_matrix[solution[j-2], solution[j-1]]
                    A21 = pdp.dist_matrix[solution[j-2], pdp.calls[i_call][1]]
                    A22 = pdp.dist_matrix[pdp.calls[i_call][1], solution[j-1]]
                add_cost = - R1 - R2 + A11 + A12 + A21 + A22

            # CHECK IF NEW BEST POSITION:
            if add_cost < best_cost:
                best_ij = (i, j)
                best_cost = add_cost

    return best_ij, best_cost


