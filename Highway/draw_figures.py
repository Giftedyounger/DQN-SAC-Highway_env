import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

# Loss = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0918_00_05\SAC_loss.npy"
# Reward = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0918_00_05\SAC_avg_reward.npy"
Reward_Double_Dqn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0905_13_55_ddqn\double_dqn_avg_reward.npy"
Reward_Double_Dqn_Pri = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0412_15_46_ddqn_pri\double_dqn_prioritized_avg_reward.npy"
Reward_Double_Dqn_Cnn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0905_14_58_ddqn_cnn\double_dqn_cnn_avg_reward.npy"
Reward_Dueling_Dqn = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\0905_16_03_dueling_dqn\dueling_dqn_avg_reward.npy"
# avg_loss = np.load(Loss)
# avg_reward_list = np.load(Reward)
# smoothed_rewards = moving_average(avg_reward_list, 5)
# smoothed_rewards = gaussian_filter1d(avg_reward_list, sigma=5)
avg_reward_list_1 = np.load(Reward_Double_Dqn)
avg_reward_list_2 = np.load(Reward_Double_Dqn_Pri)
avg_reward_list_3 = np.load(Reward_Double_Dqn_Cnn)
avg_reward_list_4 = np.load(Reward_Dueling_Dqn)


print("loss", avg_loss)
print("reward", smoothed_rewards)
plt.figure(figsize=(10, 6))
plt.plot(avg_loss)
plt.grid()
plt.title("SAC Loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0918_00_05\SAC_loss.png", dpi=150)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(smoothed_rewards)
plt.grid()
plt.title("SAC Avg Reward")
plt.xlabel("epochs")
plt.ylabel("Avg Reward")
plt.savefig(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0918_00_05\SAC_avg_reward.png", dpi=150)
plt.show()


# plt.figure(figsize=(10, 6))
# smoothed_rewards_1 = gaussian_filter1d(avg_reward_list_1, sigma=5)  # 调整 sigma 值以控制平滑程度
# smoothed_rewards_2 = gaussian_filter1d(avg_reward_list_2, sigma=5)
# smoothed_rewards_3 = gaussian_filter1d(avg_reward_list_3, sigma=5)
# smoothed_rewards_4 = gaussian_filter1d(avg_reward_list_4, sigma=5)
# plt.plot(smoothed_rewards_1)
# plt.plot(smoothed_rewards_2)
# plt.plot(smoothed_rewards_3)
# plt.plot(smoothed_rewards_4)
# plt.grid()
# plt.title("Avg Reward of Different Models")
# plt.xlabel("epochs")
# plt.ylabel("reward")
# plt.legend(["Double Dqn", "Double Dqn Prioritized", "Double Dqn Cnn", "Dueling Dqn"])
# plt.savefig(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\DDQN_series\Avg Reward.png", dpi=150)
# plt.show()
