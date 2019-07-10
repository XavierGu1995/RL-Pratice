import numpy as np
import torch
import torch.nn as nn
import gym
import time

LR = 0.001
GAMMA = 0.9
TAU = 0.01
BATCH_SIZE = 32
MAX_EPISODES = 200
MAX_EP_STEPS = 200
DISPLAY_REWARD_THRESHOLD = 400
RENDER = False  # rendering wastes time


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
        actions = self.fc2(x)
        actions_prob = torch.softmax(actions, dim=0)
        return actions_prob


class PolicyGradient(object):

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.net = Net()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LR)

    def choose_action(self, x):
        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float32), 0)
        prob_weights = self.net.forward(x)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())

    def store_transition(self, s, a, r):


    def learn(self):



if __name__ == '__main__':

    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high

    RL = PolicyGradient(s_dim, a_dim)

    var = 3  # control exploration
    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = RL.choose_action(s)
            # add randomness to action selection for exploration
            a = np.clip(np.random.normal(a, var), -2, 2)
            s_, r, done, info = env.step(a)

            RL.store_transition(s, a, r / 10)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print(
                    'Episode:',
                    i,
                    ' Reward: %i' %
                    int(ep_reward),
                    'Explore: %.2f' %
                    var,
                )
                if ep_reward > -300:
                    RENDER = True
                break
    print('Running time: ', time.time() - t1)


