import os
import copy
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import gymnasium as gym
import highway_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the environment
env = gym.make("highway-v0", render_mode='rgb_array')
env.configure({
    "action": {
        "type": "ContinuousAction"
    }
})
env.config["lanes_count"] = 4
env.config["duration"] = 100
env.config["vehicles_count"] = 10  # 10
env.config["vehicles_density"] = 1.3  #1.3
env.config["policy_frequency"] = 2
env.config["simulation_frequency"] = 10
env.config["other_vehicles_velocity"] = [15,20]  # 添加条件
env.reset()

# 终止条件1: 车辆发生碰撞
def collision_termination(env):
    return env.vehicle.crashed

# 终止条件2: 车辆驶离道路
def lane_departure_termination(env):
    return not env.vehicle.on_road

env.custom_termination_condition = lambda env: (
            collision_termination(env) or
            lane_departure_termination(env)
    )


class Prioritized_Replay:
    def __init__(
        self,
        buffer_size,
        init_length,
        state_dim,
        action_dim,
        actor,
        gamma,
    ):
        """
        A function to initialize the replay buffer.

        : param init_length: int, initial number of transitions to collect
        : param state_dim: int, size of the state space
        : param action_dim: int, size of the action space
        : param env: gym environment object
        """
        self.buffer_size = buffer_size
        self.init_length = init_length
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.priority = deque(maxlen=buffer_size)
        self._storage = []
        self._init_buffer(init_length, actor)

    def _init_buffer(self, n, actor):
        """
        Init buffer with n samples with state-transitions taken from random actions

        : param n: int, number of samples
        """
        state = env.reset()
        for _ in range(n):
            action = env.action_space.sample()
            state_next, reward, done, _, _ = env.step(action)
            if type(state) == type((1,)):
                state = state[0]
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self.prioritize(actor, exp, alpha=0.6)
            self._storage.append(exp)
            state = state_next

            if done:
                state = env.reset()
                done = False

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        : param exp: a dictionary consisting of state, action, reward , next state and done flag
        """
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)

    def prioritize(self, actor, exp, alpha=0.6):
        state = torch.FloatTensor(exp["state"]).to(device).reshape(-1)

        q = actor(state)[exp["action"]].detach().cpu().numpy()
        q_next = exp["reward"] + self.gamma * torch.max(actor(state).detach())
        # TD error
        p = (np.abs(q_next.cpu().numpy() - q) + (np.e ** -10)) ** alpha
        self.priority.append(p.item())

    def get_prioritized_batch(self, N):
        prob = self.priority / np.sum(self.priority)
        sample_idxes = random.choices(range(len(prob)), k=N, weights=prob)
        importance = (1 / prob) * (1 / len(self.priority))
        sampled_importance = np.array(importance)[sample_idxes]
        sampled_batch = np.array(self._storage)[sample_idxes]
        return sampled_batch.tolist(), sampled_importance

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        : param N: int, number of samples to obtain from the buffer
        """
        return random.sample(self._storage, N)


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SAC_prioritized(nn.Module):
    def __init__(
        self,
        env,
        device,
        state_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        beta=1,
        beta_decay=0.95,
        beta_min=0.01,
        timestamp="",
    ):
        """
        : param env: object, a gym environment
        : param state_dim: int, size of state space
        : param action_dim: int, size of action space
        : param lr: float, learning rate
        : param gamma: float, discount factor
        : param batch_size: int, batch size for training
        """
        super(SAC_prioritized, self).__init__()

        self.env = env
        self.env.reset()
        self.timestamp = timestamp

        self.test_env = copy.deepcopy(env)  # for evaluation purpose
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_entropy = target_entropy  # 目标熵
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.learn_step_counter = 0

        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic1 = QValueNet(state_dim, action_dim).to(device)
        self.critic2 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic1 = QValueNet(state_dim, action_dim).to(device)
        self.target_critic2 = QValueNet(state_dim, action_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        self.ReplayBuffer = Prioritized_Replay(
            1000,
            100,
            self.state_dim,
            self.action_dim,
            self.actor,
            gamma,
        )
        self.priority = self.ReplayBuffer.priority

        self.beta = beta
        self.beta_decay = beta_decay
        self.beta_min = beta_min

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def choose_action(self, state):
        """
        Select action using epsilon greedy method

        : param state: ndarray, the state of the environment
        : param epsilon: float, between 0 and 1
        : return: ndarray, chosen action
        """
        if type(state) == type((1,)):
            state = state[0]
        temp = [exp for exp in state]
        target = []
        target = np.array(target)
        # n dimension to 1 dimension ndarray
        for i in temp:
            target = np.append(target, i)
        state = torch.FloatTensor(target).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        next_q1 = self.target_critic1(next_states)
        next_q2 = self.target_critic2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(next_q1, next_q2), dim=1, keepdim=True)
        next_q = min_qvalue + self.log_alpha.exp() * entropy
        target_q = rewards + self.gamma * (1 - dones) * next_q
        return target_q


    def train(self, num_epochs):
        """
        Train the policy for the given number of iterations

        :param num_epochs: int, number of epochs to train the policy for
        """
        count_list = []
        loss_list = []
        total_reward_list = []
        avg_reward_list = []
        epoch_reward = 0

        for epoch in range(int(num_epochs)):
            done = False
            state = self.env.reset()
            # print(state.shape)
            avg_loss = 0
            step = 0
            if type(state) == type((1,)):
                state = state[0]
            while not done:
                step += 1
                action = self.choose_action(state)
                state_next, reward, done, _, _ = self.env.step(action)
                # self.env.render()
                # store experience to replay memory

                # 设置额外奖励
                if step > 10 and self.env.unwrapped.vehicle.speed < 21:
                    reward = reward - 5
                    done = True

                if step >= 60:  # 100
                    reward = + 20
                    done = True

                exp = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "state_next": state_next,
                    "done": done,
                }
                self.ReplayBuffer.buffer_add(exp)
                state = state_next

                # importance weighting
                if self.beta > self.beta_min:
                    self.beta *= self.beta_decay

                # sample random batch from replay memory
                exp_batch = self.ReplayBuffer.buffer_sample(self.batch_size)

                # sample random batch from replay memory
                exp_batch, importance = self.ReplayBuffer.get_prioritized_batch(
                    self.batch_size
                )
                importance = torch.FloatTensor(importance ** (1 - self.beta)).to(device)

                # extract batch data
                state_batch = torch.FloatTensor(
                    [exp["state"] for exp in exp_batch]
                ).to(device)
                action_batch = torch.LongTensor(
                    [exp["action"] for exp in exp_batch]
                ).to(device)
                reward_batch = torch.FloatTensor(
                    [exp["reward"] for exp in exp_batch]
                ).to(device)
                state_next_batch = torch.FloatTensor(
                    [exp["state_next"] for exp in exp_batch]
                ).to(device)
                done_batch = torch.FloatTensor(
                    [1 - exp["done"] for exp in exp_batch]
                ).to(device)
                # state_next_temp = [exp["state_next"] for exp in exp_batch]
                # state_temp = [exp["state"] for exp in exp_batch]
                # state_temp_list = np.array(state_temp)
                # state_next_temp_list = np.array(state_next_temp)
                #
                # state_next_batch = torch.FloatTensor(state_next_temp_list).to(device)
                # state_batch = torch.FloatTensor(state_temp_list).to(device)

                # reshape
                state_batch = state_batch.reshape(self.batch_size, -1)
                action_batch = action_batch.reshape(self.batch_size, -1)
                reward_batch = reward_batch.reshape(self.batch_size, -1)
                state_next_batch = state_next_batch.reshape(self.batch_size, -1)
                done_batch = done_batch.reshape(self.batch_size, -1)

                td_target = self.calc_target(reward_batch, state_next_batch, done_batch)  # 计算TD目标
                critic1_q_value = self.critic1(state_batch).gather(1, action_batch)  # 选择动作a后的Q值
                critic2_q_value = self.critic2(state_batch).gather(1, action_batch)

                critic1_loss = torch.mean(
                    torch.multiply(torch.square(critic1_q_value - td_target.detach()), importance)
                )
                critic2_loss = torch.mean(
                    torch.multiply(torch.square(critic2_q_value - td_target.detach()), importance)
                )
                # critic1_loss = F.mse_loss(critic1_q_value, td_target.detach()).mean()
                # critic2_loss = F.mse_loss(critic2_q_value, td_target.detach()).mean()
                avg_loss += (critic1_loss.item()+critic2_loss.item())/2

                # update network
                self.critic1_optimizer.zero_grad()
                critic1_loss.backward()
                self.critic1_optimizer.step()
                self.critic2_optimizer.zero_grad()
                critic2_loss.backward()
                self.critic2_optimizer.step()

                probs = self.actor(state_batch)
                log_probs = torch.log(probs + 1e-8)
                entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)
                q1 = self.critic1(state_batch)
                q2 = self.critic2(state_batch)
                min_qvalue = torch.sum(probs * torch.min(q1, q2), dim=1, keepdim=True)  # 直接根据概率计算期望
                actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                alpha_loss = torch.mean(self.log_alpha.exp() * (entropy - self.target_entropy).detach())
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.soft_update(self.critic1, self.target_critic1)
                self.soft_update(self.critic2, self.target_critic2)

                # # update target network
                # if self.learn_step_counter % 100 == 0:
                #     self.target_net.load_state_dict(self.estimate_net.state_dict())
                self.learn_step_counter += 1

            reward, count = self.eval()
            epoch_reward += reward
            # env.render()

            # save
            period = 1
            if epoch % period == 0:
                # log
                avg_loss /= step
                epoch_reward /= period
                avg_reward_list.append(epoch_reward)
                loss_list.append(avg_loss)

                print(
                    "\nepoch: [{}/{}], \tavg loss: {:.4f}, \tavg reward: {:.3f}, \tsteps: {}".format(
                        epoch + 1, num_epochs, avg_loss, epoch_reward, count
                    )
                )

                epoch_reward = 0
                # create a new directory for saving
                try:
                    os.makedirs(self.timestamp)
                except OSError:
                    pass
                np.save(self.timestamp + "/SAC_prioritized_loss.npy", loss_list)
                np.save(self.timestamp + "/SAC_prioritized_avg_reward.npy", avg_reward_list)
                torch.save(self.actor.state_dict(), self.timestamp + '/actormodel_new.pth')
                torch.save(self.critic1.state_dict(), self.timestamp + '/criticmodel_new.pth')
                torch.save(self.critic2.state_dict(), self.timestamp + '/criticmodel2_new.pth')

        self.env.close()
        return loss_list, avg_reward_list

    def eval(self):
        """
        Evaluate the policy
        """
        count = 0
        total_reward = 0
        done = False
        state = self.test_env.reset()
        if type(state) == type((1,)):
            state = state[0]

        while not done:
            action = self.choose_action(state)
            state_next, reward, done, _, _ = self.test_env.step(action)
            # 设置额外奖励
            if count > 10 and self.env.unwrapped.vehicle.speed < 21:
                reward = reward - 5
                done = True

            if count >= 60:  # 100
                reward = + 20
                done = True
            total_reward += reward
            count += 1
            state = state_next

        return total_reward, count


if __name__ == "__main__":

    # timestamp for saving
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime(
        "%m%d_%H_%M", named_tuple
    )  # have a folder of "date+time ex: 1209_20_36 -> December 12th, 20:36"

    SAC_prioritized_object = SAC_prioritized(
        env,
        device,
        state_dim=25,
        action_dim=2,
        actor_lr=1e-3,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        target_entropy=-2,
        gamma=0.99,
        tau=0.001,
        batch_size=64,
        timestamp=time_string,
    )

    # Train the policy
    iterations = 1000
    avg_loss, avg_reward_list = SAC_prioritized_object.train(iterations)
