# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Creating dataset
np.random.seed(10)

data = {
    "logs23_Long_Run_SA": np.loadtxt('results/pdp_100_No_RL_SA_logs23.txt'),
    "logs46_Fixed_Uniform_Prob": np.loadtxt("results/logs_46_Fixed_Uniform_Prob_Results.txt"),
    "logs50_Fixed_Uniform_Prob2": np.loadtxt("results/logs_50_results.txt"),
    "logs51_Fixed_Uniform_Prob3": np.loadtxt("results/logs_51_results.txt")
}
plt.figure(figsize=(16, 9), dpi=1200)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=list(data.values()))
ax.set(xlabel="Solvers", ylabel="Cost", title='Performance on fixed test set consisting of 100 instances')
plt.xticks(range(len(data.keys())), data.keys())
ax.figure.savefig("boxplot_4.png", dpi=300)
# show plot
plt.show()
