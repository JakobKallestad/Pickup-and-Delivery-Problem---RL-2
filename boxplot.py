# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Creating dataset
np.random.seed(10)

data_1 = np.loadtxt('results/pdp_100_No_RL_only_rm_ins_single_best.txt')
data_2 = np.loadtxt('results/pdp_100_No_RL.txt')
data_3 = np.loadtxt('results/pdp_100_No_RL_fine_tuned.txt')

# logs_6_reward_exploitation???
data_4 = np.loadtxt('results/pdp_100_RL1.txt')

# logs_26_Great_results
data_5 = np.loadtxt('results/pdp_100_RL2_Improved.txt')  # This avoided using epsilon greedy for testing (and in training I think?)

# logs_26_Great_results
data_6 = np.loadtxt('results/pdp_100_No_RL_SA_logs23.txt')  # Somewhat expertly tuned SA acceptance criteria this time.

# I think the main difference between data_4 and data_5 COULD be because of random initialization.
# For some reason data_5 came up with different operators than data_4, but there is no clear reason why that I can see.
# They both use the same (bad) reward function.
# I think maybe the data_4 used epsilon greedy for training?


print(data_1)

data = [data_1, data_2, data_3, data_4, data_5, data_6]
plt.figure(figsize=(16, 9), dpi=1200)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=data)
ax.set(xlabel="Solvers", ylabel="Cost", title='Performance on fixed test set consisting of 100 instances')
plt.xticks([0, 1, 2, 3, 4, 5], ["No_RL_action=1", 'No_RL', "No_RL_fine_tuned", 'RL', 'RL2', 'No_RL_SA'])
ax.figure.savefig("boxplot.png", dpi=300)
# show plot
plt.show()
