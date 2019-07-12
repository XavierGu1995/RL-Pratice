import numpy as np
import torch
import torch.nn as nn
import gym

LR_A = 0.001
LR_C = 0.01
GAMMA = 0.9
MAX_EPISODES = 3000
MAX_EP_STEPS = 1000
DISPLAY_REWARD_THRESHOLD = 200

ENV_NAME = 'CartPole-v0'
RENDER = False


class ActorNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.out(x)
        actions_prob = torch.softmax(x, dim=1)
        return actions_prob


class CriticNet(nn.Module):

    def __init__(self, s_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(20, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        action_value = self.out(x)
        return action_value


class A2C(object):

    def __init__(self, s_dim, a_dim):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.Actor = ActorNet(s_dim, a_dim)
        self.Critic = CriticNet(s_dim)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor.parameters(), lr=LR_A)
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic.parameters(), lr=LR_C)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        prob_weights = self.Actor.forward(s).detach()
        prob_weights.requires_grad = False
        a = np.random.choice(range(prob_weights.shape[1]),
                             p=prob_weights.squeeze().numpy())
        return a

    def learn(self, s, a, r, s_):
        # train CriticNet
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float32), 0)

        td_error = r + GAMMA * self.Critic.forward(s_) - self.Critic.forward(s)
        td_error_for_actor = td_error.detach()
        critic_loss = torch.pow(td_error, 2)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()

        # train ActorNet
        prob_weights = self.Actor(s).gather(
            1, torch.tensor([[a]], dtype=torch.long))
        log_prob_weights = torch.log(prob_weights)
        actor_loss = -td_error_for_actor * log_prob_weights

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()


# ----------------------------------training--------------------------------#

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env.seed(1)
    env = env.unwrapped

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    RL = A2C(state_dim, action_dim)

    for i_episode in range(MAX_EPISODES):
        state = env.reset()
        t = 0
        track_r = []
        while True:
            if RENDER:
                env.render()

            action = RL.choose_action(state)

            state_, reward, done, info = env.step(action)

            if done:
                reward = -20

            track_r.append(reward)

            RL.learn(state, action, reward, state_)

            state = state_
            t += 1

            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD:
                    RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                break
