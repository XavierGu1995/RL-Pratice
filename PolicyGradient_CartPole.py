import numpy as np
import torch
import torch.nn as nn
import gym

LR = 0.01
GAMMA = 0.95
DISPLAY_REWARD_THRESHOLD = 400

ENV_NAME = 'CartPole-v0'
RENDER = False


class Net(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, a_dim)
        self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        actions_prob = torch.softmax(x, dim=1)
        return actions_prob


class PolicyGradient(object):

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = Net(s_dim, a_dim)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        prob_weights = self.net.forward(s).detach()
        prob_weights.requires_grad = False
        a = np.random.choice(range(prob_weights.shape[1]),
                             p=prob_weights.squeeze().numpy())
        return a

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        r = 0
        for t in range(len(self.ep_rs)):
            r = r * GAMMA + self.ep_rs[t]

        s = torch.tensor(self.ep_obs, dtype=torch.float32)
        a = torch.tensor(self.ep_as, dtype=torch.long)

        prob_weights = self.net(s).gather(1, a.reshape(len(self.ep_rs), 1))
        log_prob_weights = torch.log(prob_weights)
        loss = -(r - 200) * torch.mean(log_prob_weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    RL = PolicyGradient(state_dim, action_dim)

    for i_episode in range(3000):
        state = env.reset()

        while True:
            if RENDER:
                env.render()

            action = RL.choose_action(state)

            state_, reward, done, info = env.step(action)

            RL.store_transition(state, action, reward)

            if done:
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = 0.99 * running_reward + ep_rs_sum * 0.01

                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True
                print("episode:", i_episode, "  reward:", int(running_reward))

                RL.learn()

                break

            state = state_
