from Operators import remove_insert, insert_first, insert_greedy, insert_beam_search, insert_tour, \
    remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors, remove_xs, remove_s, \
    remove_m, remove_l, remove_xl

remove_operators = [remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,
                    remove_xs, remove_s, remove_m, remove_l, remove_xl]
insert_operators = [insert_greedy, insert_beam_search, insert_tour] # insert_first

action = 12#4#22

remove_op = remove_operators[action // 3]
insert_op = insert_operators[action % 3]
op = (remove_op, insert_op)

print(op)