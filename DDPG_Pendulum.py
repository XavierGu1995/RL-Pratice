import numpy as np
import torch
import torch.nn as nn
import gym

# hyper parameters
MAX_EPISODE = 200
MAX_EP_STEP = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01     # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

# Gym environment
RENDER = False
ENV_NAME = 'Pendulum-v0'
DISPLAY_REWARD_THRESHOLD = -300


# build network
class ActorNet(nn.Module):

    def __init__(self, s_dim, a_dim, a_bound):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(30, a_dim)
        self.fc2.weight.data.normal_(0, 0.1)

        self.a_bound = a_bound

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)  # tanh activate to [-1, 1]
        action_value = x * torch.tensor(self.a_bound, dtype=torch.float32)  # multiply action_bound to fill action space
        return action_value


class CriticNet(nn.Module):

    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fc_state = nn.Linear(s_dim, 30)
        self.fc_state.weight.data.normal_(0, 0.1)
        self.fc_action = nn.Linear(a_dim, 30)
        self.fc_action.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        s = self.fc_state(s)
        a = self.fc_action(a)
        output = torch.relu(s + a)
        action_value = self.out(output)
        return action_value


# RL brain
class DDPG(object):

    def __init__(self, s_dim, a_dim, a_bound):
        self.s_dim, self.a_dim = s_dim, a_dim
        self.memory = np.zeros(
            (MEMORY_CAPACITY,
             s_dim * 2 + a_dim + 1),
            dtype=np.float)
        self.pointer = 0

        self.Actor_target = ActorNet(s_dim, a_dim, a_bound)
        self.Actor_eval = ActorNet(s_dim, a_dim, a_bound)
        self.Critic_target = CriticNet(s_dim, a_dim)
        self.Critic_eval = CriticNet(s_dim, a_dim)

        self.Actor_optimizer = torch.optim.Adam(
            self.Actor_eval.parameters(), lr=LR_A)
        self.Critic_optimizer = torch.optim.Adam(
            self.Critic_eval.parameters(), lr=LR_C)

        self.TD_error = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        return self.Actor_eval(s)[0].detach()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def learn(self):
        # soft target replacement
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x +
                 '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x +
                 '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # sample batch transitions
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        batch = self.memory[indices, :]
        batch_s = torch.tensor(batch[:, :self.s_dim], dtype=torch.float32)
        batch_a = torch.tensor(
            batch[:, self.s_dim:self.s_dim + self.a_dim], dtype=torch.float32)
        batch_r = torch.tensor(
            batch[:, -self.s_dim - 1:-self.s_dim], dtype=torch.float32)
        batch_s_ = torch.tensor(batch[:, -self.s_dim:], dtype=torch.float32)

        # update ActorNet
        a = self.Actor_eval(batch_s)
        q = self.Critic_eval(batch_s, a)

        actor_loss = -torch.mean(q)

        self.Actor_optimizer.zero_grad()
        actor_loss.backward()
        self.Actor_optimizer.step()

        # update CriticNet
        a_ = self.Actor_target(batch_s_)
        q_ = self.Critic_target(batch_s_, a_)

        q_target = batch_r + GAMMA * q_
        q_eval = self.Critic_eval(batch_s, batch_a)
        critic_loss = self.TD_error(q_target, q_eval)

        self.Critic_optimizer.zero_grad()
        critic_loss.backward()
        self.Critic_optimizer.step()


# -------------------------------training-------------------------------- #

if __name__ == '__main__':

    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    RL = DDPG(state_dim, action_dim, action_bound)

    var = 3  # Gauss exploration noise

    for i in range(MAX_EPISODE):
        state = env.reset()
        ep_reward = 0

        for j in range(MAX_EP_STEP):
            if RENDER:
                env.render()

            action = RL.choose_action(state)
            action = np.clip(
                np.random.normal(
                    action, var), -2, 2)  # explore action

            state_, reward, done, info = env.step(action)

            RL.store_transition(state, action, reward / 10, state_)

            if RL.pointer > MEMORY_CAPACITY:
                var *= 0.9995
                RL.learn()

            state = state_
            ep_reward += reward

            if j == MAX_EP_STEP - 1:
                print('Episode:', i,
                      'Reward: %i' % int(ep_reward),
                      'Explore: %.2f' % var)
                if ep_reward > -300:
                    RENDER = True
                break
