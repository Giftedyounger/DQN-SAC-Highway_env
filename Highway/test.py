import torch
import numpy as np
import copy
import gymnasium as gym
import highway_env
from pyglet import model

from SAC import SAC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the environment
env = gym.make("highway-v0", render_mode='rgb_array')
env.config["lanes_count"] = 4
env.config["duration"] = 100
env.config["vehicles_count"] = 10
env.config["vehicles_density"] = 1.3
env.config["policy_frequency"] = 1
env.config["simulation_frequency"] = 10
env.reset()

# Load the pre-trained network
time_string = "0412_15_46_ddqn_pri"
model_path1 = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0911_15_23\actormodel_new.pth"
model_path2 = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0911_15_23\criticmodel_new.pth"
model_path3 = r"C:\Users\12065\Desktop\Learn\DQN-Highway_env\Highway\0911_15_23\criticmodel2_new.pth"

test_env = copy.deepcopy(env)
SAC_object = SAC(
        test_env,
        device,
        state_dim=25,
        action_dim=5,
        actor_lr=1e-3,
        critic_lr=1e-2,
        alpha_lr=1e-3,
        target_entropy=-0.98,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        timestamp=time_string,
    )
SAC_object.actor.load_state_dict(torch.load(model_path1))
SAC_object.critic1.load_state_dict(torch.load(model_path2))
SAC_object.critic2.load_state_dict(torch.load(model_path3))

# Test the network without updating
success_count = 0
test_episodes = 100

for _ in range(test_episodes):
    done = False
    state = test_env.reset()
    steps = 0
    if type(state) == type((1,)):
        state = state[0]

    while not done:
        action = SAC_object.choose_action(state)  # Use greedy policy
        state_next, reward, done, _, _ = test_env.step(action)
        test_env.render()
        steps += 1
        state = state_next

        if steps >= 100:
            success_count += 1
            break

        test_env.render()
success_rate = success_count / test_episodes
print(f"Success Rate: {success_rate * 100:.2f}%")