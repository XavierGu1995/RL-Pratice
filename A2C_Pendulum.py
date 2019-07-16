import numpy as np
import torch
import torch.nn as nn
import gym

# hyper parameters
LR_A = 0.01
LR_C = 0.01
GAMMA = 0.9
MAX_EPISODE = 1000
MAX_EP_STEP = 200
DISPLAY_REWARD_THRESHOLD = -100

ENV_NAME = 'Pendulum-v0'
RENDER = False


class ActorNet(nn.Module):

    def __init__(self, s_dim, a_bound):
        super(ActorNet, self).__init__()
        self.a_bound = a_bound
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.mu = nn.Linear(30, 1)
        self.mu.weight.data.normal_(0, 0.1)
        self.sigma = nn.Linear(30, 1)
        self.sigma.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu)
        sigma = self.sigma(x)
        sigma = nn.functional.softplus(sigma)
        normal_dist = torch.distributions.normal.Normal(
            mu * torch.tensor(self.a_bound), sigma + 0.1)
        return normal_dist


class CriticNet(nn.Module):

    def __init__(self, s_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        action_value = self.out(x)
        return action_value


class A2C(object):

    def __init__(self, s_dim, a_bound):
        self.s_dim = s_dim
        self.a_bound = a_bound

        self.Actor = ActorNet(s_dim, a_bound)
        self.Critic = CriticNet(s_dim)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(), lr=LR_A)
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic.parameters(), lr=LR_C)

    def choose_action(self, s):
        s = torch.unsqueeze(
            torch.tensor(s, dtype=torch.float32), 0).reshape(1, self.s_dim)
        normal_dist = self.Actor.forward(s)
        a = np.clip(normal_dist.sample(), -self.a_bound, self.a_bound)
        return a

    def learn(self, s, a, r, s_):
        s = torch.unsqueeze(
            torch.tensor(s, dtype=torch.float32), 0).reshape(1, self.s_dim)
        s_ = torch.unsqueeze(
            torch.tensor(s_, dtype=torch.float32), 0).reshape(1, self.s_dim)
        a = torch.tensor(a, dtype=torch.float32)

        # update CriticNet
        td_error = r + GAMMA * self.Critic.forward(s_) - self.Critic.forward(s)
        td_error_for_actor = td_error.detach()
        critic_loss = torch.pow(td_error, 2)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        # update ActorNet
        normal_dist = self.Actor.forward(s)
        actor_loss = normal_dist.log_prob(
            a) * td_error_for_actor + 0.01 * normal_dist.entropy()

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()


# ----------------------------------training--------------------------------#

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_bound = env.action_space.high

    RL = A2C(state_dim, action_bound)

    for i_episode in range(MAX_EPISODE):
        state = env.reset()
        t = 0
        ep_rs = []
        while True:
            if RENDER:
                env.render()

            action = RL.choose_action(state)

            state_, reward, done, info = env.step(action)

            RL.learn(state, action, reward / 10, state_)

            state = state_
            t += 1
            ep_rs.append(reward)

            if t > MAX_EP_STEP:
                ep_rs_sum = sum(ep_rs)
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.9 + ep_rs_sum * 0.1
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break
