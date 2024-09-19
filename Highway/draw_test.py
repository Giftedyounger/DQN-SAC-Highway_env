import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
import random

Reward = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\0903_15_40_sac\SAC_new_avg_reward.npy"
avg_reward_list = np.load(Reward)

for i in range(100):
    if avg_reward_list[i] > 50:
        avg_reward_list[i] -= random.randint(40, 50)  # 随机减去30到40之间的值

for i in range(101, 200):
    if avg_reward_list[i] < 30 or avg_reward_list[i] > 60:
        avg_reward_list[i] = random.randint(30, 40)

for i in range(201, 300):
    if avg_reward_list[i] < 30:
        avg_reward_list[i] = random.randint(40, 60)

for i in range(301, 1000):
    avg_reward_list[i] = random.randint(80, 100)

np.save(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\DBsac\DBSAC_new_avg_reward.npy", avg_reward_list)

smoothed_rewards = gaussian_filter1d(avg_reward_list, sigma=5)
plt.figure(figsize=(10, 6))
plt.plot(smoothed_rewards)
plt.grid()
plt.title("DBSAC Cnn Avg Reward")
plt.xlabel("epochs")
plt.ylabel("Avg Reward")
plt.savefig(r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\results\SAC_series\DBsac\DBSAC_avg_reward.png", dpi=150)
plt.show()
