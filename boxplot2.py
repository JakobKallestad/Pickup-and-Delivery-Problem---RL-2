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

# logs_30 (on unseen training data, will benchmark on the real training dataset soon.)
data_4 = np.loadtxt('results/random.txt')

data = [data_1, data_2, data_3, data_4]
plt.figure(figsize=(16, 9), dpi=1200)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=data)
ax.set(xlabel="Solvers", ylabel="Cost", title='Performance on fixed test set consisting of 100 instances')
plt.xticks([0, 1, 2, 3], ['RL', 'RL2', 'No_RL_SA', 'RL3'])
ax.figure.savefig("boxplot_2.png", dpi=300)
# show plot
plt.show()
