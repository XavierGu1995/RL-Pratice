import numpy as np
import torch
import torch.nn as nn
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(
    env.action_space.sample(),
    int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros(
            (MEMORY_CAPACITY, N_STATES * 2 + 2))

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(s)
            a = torch.max(actions_value, 1)[1].numpy()
            a = a[0] if ENV_A_SHAPE == 0 else a.reshape(
                ENV_A_SHAPE)
        else:   # random
            a = np.random.randint(0, N_ACTIONS)
            a = a if ENV_A_SHAPE == 0 else a.reshape(
                ENV_A_SHAPE)
        return a

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(b_memory[:, :N_STATES], dtype=torch.float32)
        b_a = torch.tensor(
            b_memory[:, N_STATES:N_STATES + 1].astype(int), dtype=torch.long)
        b_r = torch.tensor(
            b_memory[:, N_STATES + 1:N_STATES + 2], dtype=torch.float32)
        b_s_ = torch.tensor(b_memory[:, -N_STATES:], dtype=torch.float32)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * \
            q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# --------------------------------training---------------------------------- #

if __name__ == '__main__':

    dqn = DQN()

    print('\nCollecting experience...')
    for i_episode in range(400):
        state = env.reset()
        ep_r = 0
        while True:
            env.render()
            action = dqn.choose_action(state)

            state_, r0, done, info = env.step(action)

            x0, x_dot, theta, theta_dot = state_
            r1 = (env.x_threshold - abs(x0)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / \
                env.theta_threshold_radians - 0.5
            reward = r1 + r2

            dqn.store_transition(state, action, reward, state_)

            ep_r += reward
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            state = state_
