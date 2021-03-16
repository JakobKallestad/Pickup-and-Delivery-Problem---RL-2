import torch
import torch.optim as optim
from model.model import QNet
from tensorboardX import SummaryWriter

from model.memory import Memory
from config import log_interval, device, lr

from PDP import generate_problem, load_pdp_from_file, PDP
from Utils import objective_function, embed_solution
from Operators import remove_insert, insert_first, insert_greedy, insert_beam_search, insert_tour, \
    remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors, remove_xs, remove_s, \
    remove_m, remove_l, remove_xl
import random
import pickle


def main():
    torch.manual_seed(500)

    SIZE = 100
    num_inputs = SIZE*11  # STEP * FEATURES (look at paper LU)
    num_actions = 8*4  # REMOVE_OPERATORS * INSERT_OPERATORS
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = QNet(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    writer = SummaryWriter('logs4')

    net.to(device)
    net.train()
    steps = 0
    loss = 0

    # Operators and logic
    remove_operators = [remove_single_best, remove_longest_tour_deviation, remove_tour_neighbors,
                        remove_xs, remove_s, remove_m, remove_l, remove_xl]
    insert_operators = [insert_first, insert_greedy, insert_beam_search, insert_tour]

    # If dataset instead:
    dataset = load_pdp_from_file("data/pdp_20/pdp20_TEST1_seed1234.pkl")

    cost_list = []
    for e in range(500):
        memory = Memory()

        # generate random data:
        pdp = generate_problem(size=SIZE)
        # or fetch instance from file:
        #pdp = dataset[e]

        pdp.initialize_close_calls()



        best_solution = [i for i in range(1, pdp.size + 1)]
        best_cost = objective_function(pdp, best_solution)

        solution = best_solution
        state = embed_solution(pdp, solution)
        state = torch.tensor(state.flatten(), device=device, dtype=torch.float32)

        score = 0

        for i in range(10000):  # number of steps to improve solution
            steps += 1

            # GET ACTION
            action = net.get_action(state)

            # ENV ACT
            remove_op = remove_operators[action//4]
            insert_op = insert_operators[action%4]
            op = (remove_op, insert_op)
            solution, cost = remove_insert(pdp, solution, op)

            if cost < best_cost:
                best_solution = solution
                best_cost = cost
                reward = 1
            else:
                reward = -1

            next_state = embed_solution(pdp, solution)
            next_state = torch.tensor(next_state.flatten(), device=device, dtype=torch.float32)

            action_one_hot = torch.zeros(num_actions, device=device)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward)

            score += reward
            state = next_state

        cost_list.append(best_cost)
        loss = QNet.train_model(net, memory.sample(), optimizer, device)  # carefull with sample(), maybe just go through it sequentially
        if e % log_interval == 0:
            mean_cost = sum(cost_list[-50:]) / len(cost_list[-50:])
            print('{} episode | mean_cost: {:.2f}'.format(e, mean_cost))
            writer.add_scalar('log/score', float(score), e)
            writer.add_scalar('log/loss', float(loss), e)
            writer.add_scalar('log/mean_cost', mean_cost, e)


if __name__ == "__main__":
    main()