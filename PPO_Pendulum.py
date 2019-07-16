import numpy as np
import torch
import torch.nn as nn
import gym
import matplotlib.pyplot as plt

# hyper parameters
LR_A = 0.0001
LR_C = 0.0002
GAMMA = 0.9
BUFFER_SIZE = 32
MAX_EPISODE = 1000
MAX_EP_STEP = 200
UPDATE_STEPS_A = 10
UPDATE_STEPS_C = 10
EPSILON = 0.2
DISPLAY_REWARD_THRESHOLD = -100

# Gym environment
ENV_NAME = 'Pendulum-v0'
RENDER = False


# build network
class ActorNet(nn.Module):

    def __init__(self, s_dim, a_dim, a_bound):
        super(ActorNet, self).__init__()
        self.a_bound = a_bound
        self.fc1 = nn.Linear(s_dim, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.mu = nn.Linear(100, a_dim)
        self.mu.weight.data.normal_(0, 0.1)
        self.sigma = nn.Linear(100, a_dim)
        self.sigma.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu)
        sigma = self.sigma(x)
        sigma = nn.functional.softplus(sigma)
        normal_dist = torch.distributions.normal.Normal(
            mu * torch.tensor(self.a_bound), sigma)
        return normal_dist


class CriticNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 100)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(100, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        action_value = self.out(x)
        return action_value


# RL brain
class PPO(object):

    def __init__(self, s_dim, a_dim, a_bound):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a_bound = a_bound

        self.Actor = ActorNet(s_dim, a_dim, a_bound)
        self.Actor_old = ActorNet(s_dim, a_dim, a_bound)
        self.Critic = CriticNet(s_dim, a_dim)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(), lr=LR_A)
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic.parameters(), lr=LR_C)

    def choose_action(self, s):
        s = torch.unsqueeze(
            torch.tensor(s, dtype=torch.float32), 0)
        normal_dist = self.Actor.forward(s)
        a = np.clip(normal_dist.sample(), -self.a_bound, self.a_bound)
        return a

    def learn(self, buffer_s, buffer_a, buffer_r, s_):
        buffer_s = torch.tensor(buffer_s, dtype=torch.float32)
        buffer_a = torch.tensor(
            buffer_a, dtype=torch.float32).reshape(
            len(buffer_a), 1)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float32), 0)

        # TD-error
        v_s_ = self.Critic.forward(s_)
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r = torch.unsqueeze(torch.tensor(
            discounted_r, dtype=torch.float32), 0).flip(1).t()

        td_error = discounted_r - self.Critic.forward(buffer_s)

        # update Actor_old
        self.Actor_old.load_state_dict(self.Actor.state_dict())

        # update Actor
        for i in range(UPDATE_STEPS_A):
            td_error_for_actor = td_error.detach()
            ratio = torch.exp(self.Actor.forward(buffer_s).log_prob(
                buffer_a)) / (torch.exp(self.Actor_old.forward(buffer_s).log_prob(buffer_a)) + 0.00001)
            actor_loss = -torch.mean(
                torch.min(ratio * td_error_for_actor, torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)))

            self.Actor_optimizer.zero_grad()
            actor_loss.backward()
            self.Actor_optimizer.step()

        # update Critic
        for i in range(UPDATE_STEPS_C):
            td_error = discounted_r - self.Critic.forward(buffer_s)
            critic_loss = torch.mean(torch.pow(td_error, 2))

            self.Critic_optimizer.zero_grad()
            critic_loss.backward()
            self.Critic_optimizer.step()


# ----------------------------------training-----------------------------------#

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    RL = PPO(state_dim, action_dim, action_bound)

    all_episode_reward = []

    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        buffer_state, buffer_action, buffer_reward = [], [], []
        episode_reward = 0

        for t in range(MAX_EP_STEP):
            # if i_episode > 200:
            #     RENDER = True
            if RENDER:
                env.render()

            action = RL.choose_action(state)
            state_, reward, done, info = env.step(action)

            buffer_state.append(state)
            buffer_action.append(action)
            buffer_reward.append((reward + 8) / 8)

            state = state_
            episode_reward += reward

            if (t + 1) % BUFFER_SIZE == 0 or t == MAX_EP_STEP - 1:
                RL.learn(buffer_state, buffer_action, buffer_reward, state_)

        if i_episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(
                all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        print('Ep: %i' % i_episode, episode_reward)

    plt.plot(np.arange(len(all_episode_reward)), all_episode_reward)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
