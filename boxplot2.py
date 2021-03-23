# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Creating dataset
np.random.seed(10)

# logs_6_reward_exploitation???
data_1 = np.loadtxt('results/pdp_100_RL1.txt')

# logs_26_Great_results
data_2 = np.loadtxt('results/pdp_100_RL2_Improved.txt')

# logs_23_Long_Run_SA
data_3 = np.loadtxt('results/pdp_100_No_RL_SA_logs23.txt')

data_4 = np.loadtxt('results/logs_33_TESTING_model_32_results.txt')  # not trained for that long

data_5 = np.loadtxt('results/pdp_100_No_RL_Fixed_Probabilities_logs27.txt')

data_6 = np.loadtxt('results/pdp_100_No_RL_logs41.txt')


data = [data_1, data_2, data_3, data_4, data_5, data_6]
plt.figure(figsize=(16, 9), dpi=1200)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=data)
ax.set(xlabel="Solvers", ylabel="Cost", title='Performance on fixed test set consisting of 100 instances')
plt.xticks([0, 1, 2, 3, 4, 5], ['RL', 'RL2', 'No_RL_SA', 'RL4', 'No_RL_Fixed', 'No_RL_Better'])
ax.figure.savefig("boxplot_3.png", dpi=300)
# show plot
plt.show()
