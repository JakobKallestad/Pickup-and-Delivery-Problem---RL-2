# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Creating dataset
np.random.seed(10)

data = {
    #"logs23_Long_Run_SA": np.loadtxt('results/pdp_100_No_RL_SA_logs23.txt'),
    #"logs46_Fixed_Uniform_Prob": np.loadtxt("results/logs_46_Fixed_Uniform_Prob_Results.txt"),
    #"logs50_Fixed_Uniform_Prob2": np.loadtxt("results/logs_50_results.txt"),
    "logs51_Fixed_Uniform_Prob3": np.loadtxt("results/logs_51_results.txt"),  # all operators
    "logs_60_SA": np.loadtxt("results/logs_60_oslo1_SA_results.txt"),  # 0.996, segment_size50, all operators
    "logs60_SA_10k": np.loadtxt("results/logs60_oslo2_SA_10k_results.txt"),
    "logs61_SA_100k": np.loadtxt("results/logs_61_oslo1_SA_100k_results.txt"),
    "logs61_SA_100k_2": np.loadtxt("results/logs_61_oslo2_SA_100k_results.txt"),
    #"RL_PPO_1_training_results": np.loadtxt("results/RL_PPO_1_training_results.txt"),  # logs_14 after about 1k training instances
    #"RL_PPO_2_training_results": np.loadtxt("results/RL_PPO_2_training_results.txt"),  # logs_14 after about 1.4k training instances
    "RL_PPO_3": np.loadtxt("results/logs_28_bergen1_PPO_5310_results.txt"),  # Based on model: logs_24_actor_critic_torch_ppo (TODO: WRITE ABOUT THIS IN THESIS!)
    "RL_PPO_4": np.loadtxt("results/logs_30_bergen1_PPO_5310_results.txt"),  # Based on model: logs_29_actor_critic_torch_ppo (TODO: WRITE ABOUT THIS IN THESIS!)
    "RL_PPO_5": np.loadtxt("results/logs_31_bergen1_PPO_+-1_results.txt")    # Based on model: logs_30+-1_actor_critic_torch_ppo
}

# Notes on PPO3 and PPO4: They both looked 10 steps into the future in terms of reward and had gae_lambda = 0.5
# Next I will try a sparse reward function, so I will therefore change ppo_main.py to look far into the future with a higher gae_lambda
# Perhaps try to lower the n_epochs paramter in PPO. Maybe this causes the sudden twist and turns (overfitting) of the agent
# Perhaps try to partition and episode into batches of size 64 instead of simply a single batch of size 1000. Who knows? might work.

#data = {"pdp_20_SA_logs_55": np.loadtxt("results/pdp_20_SA_logs_55_results.txt")}

plt.figure(figsize=(16, 9), dpi=1200)
sns.set_theme(style="whitegrid")
ax = sns.boxplot(data=list(data.values()))
ax.set(xlabel="Solvers", ylabel="Cost", title='Performance on fixed test set consisting of 100 instances')
plt.xticks(range(len(data.keys())), data.keys())
ax.figure.savefig("boxplot_14.png", dpi=300)
# show plot
plt.show()
