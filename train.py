##
## Adapted from Jupyter Notebook example here: https://github.com/toruseo/UXsim
##

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from scipy.optimize import minimize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import copy

import argparse
import sys
from subprocess import call
from TrafficSim import TrafficSim

# Increase recursion limit 
sys.setrecursionlimit(10000)

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='The number of transistions sampled from the replay buffer')
parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor')
parser.add_argument('--eps_start', type=float, default=0.9, help='The starting value of epsilon')
parser.add_argument('--eps_end', type=float, default=0.05, help='The final value of epsilon')
parser.add_argument('--eps_decay', type=int, default=1000, help='The rate of exponential decay of epsilon, higher means a slower decay')
parser.add_argument('--tau', type=float, default=0.005, help='The update rate of the target network')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of the optimizer')
parser.add_argument('--out_path', type=str, default='./outputs/', help='The path to save model and optimizer states to, DO NOT CHOOSE \'./out\'')
parser.add_argument('--device', type=str, default=None, help='Force training to run on this device')
parser.add_argument('--num_episodes', type=int, default=200, help='Number of episodes to train for')
parser.add_argument('--mem_size', type=int, default=10000, help='Replay memory capacity')
parser.add_argument('--save_fig', action='store_true', help='Save visualizations to files instead of showing them')
parser.add_argument('--best_ep_criteria', type=str, default='avg_delay', help='Choose the criteria to judge what episode is the best, \'avg_delay\' for using average delay and \'trip_ratio\' for the ratio of completed vs. taken trips')
opt = parser.parse_args()
print(opt)

# If device was not specified, use GPU if available, else use CPU
if opt.device == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If device was specified, use that 
else:
    device = opt.device

env = TrafficSim()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        n_neurals = 64
        n_layers = 3
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, n_neurals))
        for i in range(n_layers):
            self.layers.append(nn.Linear(n_neurals, n_neurals))
        self.layer_last = nn.Linear(n_neurals, n_actions)

    # Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layer_last(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = opt.eps_end + (opt.eps_start - opt.eps_end) * math.exp(-1. * steps_done / opt.eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row. second column on max result is index of where max element was found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < opt.batch_size:
        return
    transitions = memory.sample(opt.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(opt.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * opt.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=opt.lr, amsgrad=True)

# If model state was previously saved, load it and the saved optimizer state
if os.path.exists(opt.out_path+'policy_net.pth'):
    policy_net.load_state_dict(torch.load(opt.out_path+'policy_net.pth'))
    target_net.load_state_dict(torch.load(opt.out_path+'target_net.pth'))
    optimizer.load_state_dict(torch.load(opt.out_path+'optimizer.pth'))
    print(f"Loaded checkpoints from \'{opt.out_path}\'")
# If output path doesn't exist, create it
elif not os.path.isdir(opt.out_path):
    os.makedirs(opt.out_path)
    print(f"Created folder at \'{opt.out_path}\'")

memory = ReplayMemory(opt.mem_size)

steps_done = 0

log_states = []
log_epi_criteria = []
best_criteria = 9999999999999999999999999
best_W = None
best_i_episode = -1

for i_episode in range(opt.num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    log_states.append([])
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        log_states[-1].append(state)
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*opt.tau + target_net_state_dict[key]*(1-opt.tau)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            if opt.best_ep_criteria == 'trip_ratio':
                criteria = -env.W.analyzer.trip_completed/env.W.analyzer.trip_all
            else:
                criteria = env.W.analyzer.average_delay
            log_epi_criteria.append(criteria)
            print(f"{i_episode}:[{criteria : .3f}]", end=" ")
            if criteria < best_criteria:
                print(f"Current best episode!")
                best_criteria = criteria
                best_W = copy.deepcopy(env.W)
                best_i_episode = i_episode
                # Save current policy net state, one with epoch number in name and one without
                torch.save(policy_net.state_dict(), opt.out_path+f'policy_net_{i_episode:02d}.pth')
                torch.save(policy_net.state_dict(), opt.out_path+f'policy_net.pth')
                # Save current target net state, one with epoch number in name and one without
                torch.save(target_net.state_dict(), opt.out_path+f'target_net_{i_episode:02d}.pth')
                torch.save(target_net.state_dict(), opt.out_path+f'target_net.pth')
                # Save current optimizer state
                torch.save(optimizer.state_dict(), opt.out_path+'optimizer.pth')
            break

    if i_episode%50 == 0 or i_episode == opt.num_episodes-1:
        if opt.save_fig:
            env.W.save_mode=1
        env.W.analyzer.print_simple_stats(force_print=True)
        env.W.analyzer.macroscopic_fundamental_diagram()
        env.W.analyzer.time_space_diagram_traj_links([["W1I1", "I1I2", "I2I3", "I3E1"], ["N1I1", "I1I4", "I4I7", "I7S1"]], figsize=(12,3))
        for t in list(range(0,env.W.TMAX,int(env.W.TMAX/4))):
            env.W.analyzer.network(t, detailed=1, network_font_size=0, figsize=(9,9))
        call(["cp", "-r", "./out/", opt.out_path])
        call(["mv", opt.out_path+"out/", opt.out_path+f"{i_episode}"])
        plt.figure(figsize=(8,6))
        plt.plot(log_epi_criteria, "r.")
        plt.xlabel("episode")
        if opt.best_ep_criteria == 'trip_ratio':
            plt.ylabel("ratio of trips completed over trips taken")
        else:
            plt.ylabel("average delay (s)")
        plt.grid()
        if opt.save_fig:
            plt.savefig(opt.out_path+'train_stats.png')
        else:
            plt.show()

if opt.best_ep_criteria == 'trip_ratio':
    print(f"BEST EPISODE: {best_i_episode}, with {best_criteria:.0%} of trips completed")
else:
    print(f"BEST EPISODE: {best_i_episode}, with average delay {best_criteria}")
best_W.save_mode = 0
best_W.analyzer.print_simple_stats(force_print=True)
best_W.analyzer.macroscopic_fundamental_diagram()
best_W.analyzer.time_space_diagram_traj_links([["W1I1", "I1I2", "I2I3", "I3E1"], ["N1I1", "I1I4", "I4I7", "I7S1"]], figsize=(12,3))
for t in list(range(0,best_W.TMAX,int(env.W.TMAX/4))):
    best_W.analyzer.network(t, detailed=1, network_font_size=0, figsize=(9,9))
best_W.save_mode = 1
print("start anim")
best_W.analyzer.network_anim(detailed=1, network_font_size=0, figsize=(9,9))
print("end anim")
call(["mv", "./out/", opt.out_path+f"{best_i_episode}-best"])
