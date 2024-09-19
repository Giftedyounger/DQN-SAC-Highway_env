import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


Reward_SAC = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\0903_15_40_sac\SAC_new_avg_reward.npy"
# Reward_SAC_Pri = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\0904_14_15_sac_pri\SAC_prioritized_new_avg_reward.npy"
# Reward_SAC_Cnn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\0904_15_25_saccnn\SAC_cnn_new_avg_reward.npy"
Reward_DBSAC = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\DBsac\DBSAC_new_avg_reward.npy"
Reward_Double_Dqn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0905_13_55_ddqn\double_dqn_avg_reward.npy"
Reward_Dueling_Dqn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0905_16_03_dueling_dqn\dueling_dqn_avg_reward.npy"
avg_reward_list_1 = np.load(Reward_SAC)
avg_reward_list_2 = np.load(Reward_DBSAC)
avg_reward_list_3 = np.load(Reward_Double_Dqn)
avg_reward_list_4 = np.load(Reward_Dueling_Dqn)
smoothed_rewards_1 = gaussian_filter1d(avg_reward_list_1, sigma=5)  # 调整 sigma 值以控制平滑程度
smoothed_rewards_2 = gaussian_filter1d(avg_reward_list_2, sigma=5)
smoothed_rewards_3 = gaussian_filter1d(avg_reward_list_3, sigma=5)
smoothed_rewards_4 = gaussian_filter1d(avg_reward_list_4, sigma=5)
plt.figure(figsize=(10, 6))
plt.plot(smoothed_rewards_1)
plt.plot(smoothed_rewards_2)
plt.plot(smoothed_rewards_3)
plt.plot(smoothed_rewards_4)
plt.grid()
plt.title("Avg Reward of Different Models")
plt.xlabel("epochs")
plt.ylabel("reward")
plt.legend(["SAC", "DBSAC", "Double Dqn", "Dueling Dqn"])
plt.savefig(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\Avg Reward new.png", dpi=150)
plt.show()
