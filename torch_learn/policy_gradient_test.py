import torch
import torch.nn as nn
import torch.optim as optim
import gym
from torch.autograd import Variable


learning_rate = 1e-2
gamma = 0.99
env = gym.make('CartPole-v0').unwrapped

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        self._hidden_size = 128
        self.l1 = nn.Linear(self.state_space, self._hidden_size, bias=False)
        self.l2 = nn.Linear(self._hidden_size, self.action_space, bias=False)

        self.gamma = gamma

        self.policy_history = torch.tensor([])
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []
        self.model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


policy = PolicyNet()
opt = optim.Adam(policy.parameters(), lr=learning_rate)


from torch.distributions import Categorical


def select_action(state):
    state = torch.from_numpy(state).float()
    state = policy(Variable(state))

    c = Categorical(state)
    action = c.sample()

    lprob = torch.tensor([c.log_prob(action)])
    if len(policy.policy_history) != 0:
        policy.policy_history = torch.cat((policy.policy_history, lprob))
    else:
        policy.policy_history = lprob
    return action

import numpy as np

def update_policy():
    R = 0
    rewards = []

    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)

    rewards = torch.tensor(rewards).float()
    t = rewards.std() + 1e-10
    rewards = (rewards - rewards.mean()) / t

    loss = torch.sum(torch.mul(policy.policy_history, Variable(rewards)).mul(-1), -1)
    opt.zero_grad()
    loss.backward()
    opt.step()

    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_history))
    policy.policy_history = Variable(torch.tensor())
    policy.reward_episode = []

def main(episodes):
    running_reward = 10

    for episode in range(episodes):
        state = env.reset()
        done = False

        for time in range(1000):
            action = select_action(state)
            state, reward, done, _ = env.step(action.item())
            policy.reward_episode.append(reward)
            if done:
                break

        running_reward = (running_reward*gamma) + time*0.01

        update_policy()

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))


        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


if __name__ == "__main__":
    main(1000)