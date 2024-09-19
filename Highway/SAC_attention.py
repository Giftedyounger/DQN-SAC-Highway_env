import os
import copy
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym
import highway_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the environment
env = gym.make("highway-v0", render_mode='rgb_array')
env.config["lanes_count"] = 4
env.config["duration"] = 60   # 100
env.config["vehicles_count"] = 10  # 10
env.config["vehicles_density"] = 1.3  #1.3
env.config["policy_frequency"] = 2
env.config["simulation_frequency"] = 10
env.config["other_vehicles_velocity"] = [15,20]  # 添加条件
env.reset()


class Replay:
    def __init__(self, buffer_size, init_length, state_dim, action_dim, env):
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
        self.env = env

        self._storage = []
        self._init_buffer(init_length)

    def _init_buffer(self, n):
        """
        Init buffer with n samples with state-transitions taken from random actions

        : param n: int, number of samples
        """
        state = self.env.reset()
        for _ in range(n):
            action = self.env.action_space.sample()
            state_next, reward, done, _, _ = self.env.step(action)
            if type(state) == type((1,)):
                state = state[0]
            exp = {
                "state": state,
                "action": action,
                "reward": reward,
                "state_next": state_next,
                "done": done,
            }
            self._storage.append(exp)
            state = state_next

            if done:
                state = self.env.reset()
                done = False

    def buffer_add(self, exp):
        """
        A function to add a dictionary to the buffer

        : param exp: a dictionary consisting of state, action, reward , next state and done flag
        """
        self._storage.append(exp)
        if len(self._storage) > self.buffer_size:
            self._storage.pop(0)

    def buffer_sample(self, N):
        """
        A function to sample N points from the buffer

        : param N: int, number of samples to obtain from the buffer
        """
        return random.sample(self._storage, N)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.values = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.keys = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.queries = nn.Linear(embed_size, heads * self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        values = self.values(values).reshape(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).reshape(N, -1, self.heads, self.head_dim)
        queries = self.queries(query).reshape(N, -1, self.heads, self.head_dim)

        # Attention score calculation, shape alignment for batch matmul
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out



class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        hidden_dim_1 = 128  # 64
        hidden_dim_2 = 256  # 64
        self.attention = SelfAttention(state_dim, 5)
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)  # 128
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)   # 256
        self.fc3 = nn.Linear(hidden_dim_2, action_dim)   # 64

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.attention(x, x, x, None)
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=0)


class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNet, self).__init__()
        hidden_dim_1 = 128
        hidden_dim_2 = 256
        self.attention = SelfAttention(state_dim, 5)
        self.fc1 = nn.Linear(state_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, action_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.attention(x, x, x, None)
        x = x.squeeze(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SAC_Attention(nn.Module):
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
        super(SAC_Attention, self).__init__()

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

        self.ReplayBuffer = Replay(1000, 100, self.state_dim, self.action_dim, env)

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
        td_target = rewards + self.gamma * (1 - dones) * next_q
        return td_target


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
                if step > 20 and self.env.unwrapped.vehicle.speed < 22:
                    reward = reward - 5
                    done = True

                if step >= 60:   #100
                    # reward = + 20
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

                # sample random batch from replay memory
                exp_batch = self.ReplayBuffer.buffer_sample(self.batch_size)

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
                critic1_loss = F.mse_loss(critic1_q_value, td_target.detach()).mean()
                critic2_loss = F.mse_loss(critic2_q_value, td_target.detach()).mean()
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
                # log_probs = torch.log(torch.clamp(probs, min=1e-8))
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
                np.save(self.timestamp + "/SAC_Attention_loss.npy", loss_list)
                np.save(self.timestamp + "/SAC_Attention_avg_reward.npy", avg_reward_list)

                if count >= 30:
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

            if count > 20 and self.env.unwrapped.vehicle.speed < 22:
                reward = reward - 5
                done = True

            total_reward += reward
            count += 1
            state = state_next

            if count >= 60:  #100
                # reward = + 20
                done = True

        return total_reward, count


if __name__ == "__main__":

    # timestamp for saving
    named_tuple = time.localtime()  # get struct_time
    time_string = time.strftime(
        "%m%d_%H_%M", named_tuple
    )  # have a folder of "date+time ex: 1209_20_36 -> December 12th, 20:36"

    SAC_Attention_object = SAC_Attention(
        env,
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

    # Train the policy
    iterations = 1000
    avg_loss, avg_reward_list = SAC_Attention_object.train(iterations)
