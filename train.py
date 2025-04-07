##
## Adapted from Jupyter Notebook example here: https://github.com/toruseo/UXsim
##

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import gymnasium as gym
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

from uxsim import *
import random
import itertools
import copy

import argparse

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='The number of transistions sampled from the replay buffer')
parser.add_argument('--gamma', type=float, default=0.99, help='The discount factor')
parser.add_argument('--eps_start', type=float, default=0.9, help='The starting value of epsilon')
parser.add_argument('--eps_end', type=float, default=0.05, help='The final value of epsilon')
parser.add_argument('--eps_decay', type=int, default=1000, help='The rate of exponential decay of epsilon, higher means a slower decay')
parser.add_argument('--tau', type=float, default=0.005, help='The update rate of the target network')
parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of the optimizer')
parser.add_argument('--out_path', type=str, default='./out/', help='The path to save model and optimizer states to')
parser.add_argument('--device', type=str, default=None, help='Force training to run on this device')
opt = parser.parse_args()
print(opt)

# If device was not specified, use GPU if available, else use CPU
if opt.device == None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If device was specified, use that 
else:
    device = opt.device

class TrafficSim(gym.Env):
    def __init__(self):
        """
        traffic scenario: 4 signalized intersections as shown below:
                N1  N2
                |   |
            W1--I1--I2--E1
                |   |
            W2--I3--I4--E2
                |   |
                S1  S2
        Traffic demand is generated from each boundary node to all other boundary nodes.
        action: to determine which direction should have greenlight for every 10 seconds for each intersection. 16 actions.
            action 1: greenlight for I1: direction 0, I2: 0, I3: 0, I4: 0, where direction 0 is E<->W, 1 is N<->S.
            action 2: greenlight for I1: 1, I2: 0, I3: 0, I4: 0
            action 3: greenlight for I1: 0, I2: 1, I3: 0, I4: 0
            action 4: greenlight for I1: 1, I2: 1, I3: 0, I4: 0
            action 5: ...
        state: number of waiting vehicles at each incoming link. 16 dimension.
        reward: negative of difference of total waiting vehicles
        """
        
        #action
        self.n_action = 2**4
        self.action_space = gym.spaces.Discrete(self.n_action) 
        
        #state
        self.n_state = 4*4
        low = np.array([0 for i in range(self.n_state)])
        high = np.array([100 for i in range(self.n_state)])
        self.observation_space = gym.spaces.Box(low=low, high=high)
        
        self.reset()
    
    def reset(self):
        """
        reset the env
        """
        seed = None #whether demand is always random or not
        W = World(
            name="",
            deltan=5,
            tmax=4000,
            print_mode=0, save_mode=0, show_mode=1,
            random_seed=seed,
            duo_update_time=600
        )
        random.seed(seed)

        #network definition
        I1 = W.addNode("I1", 0, 0, signal=[60,60])
        I2 = W.addNode("I2", 1, 0, signal=[60,60])
        I3 = W.addNode("I3", 0, -1, signal=[60,60])
        I4 = W.addNode("I4", 1, -1, signal=[60,60])
        W1 = W.addNode("W1", -1, 0)
        W2 = W.addNode("W2", -1, -1)
        E1 = W.addNode("E1", 2, 0)
        E2 = W.addNode("E2", 2, -1)
        N1 = W.addNode("N1", 0, 1)
        N2 = W.addNode("N2", 1, 1)
        S1 = W.addNode("S1", 0, -2)
        S2 = W.addNode("S2", 1, -2)
        #E <-> W direction: signal group 0
        for n1,n2 in [[W1, I1], [I1, I2], [I2, E1], [W2, I3], [I3, I4], [I4, E2]]:
            W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
            W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=0)
        #N <-> S direction: signal group 1
        for n1,n2 in [[N1, I1], [I1, I3], [I3, S1], [N2, I2], [I2, I4], [I4, S2]]:
            W.addLink(n1.name+n2.name, n1, n2, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)
            W.addLink(n2.name+n1.name, n2, n1, length=500, free_flow_speed=10, jam_density=0.2, signal_group=1)

        # random demand definition
        dt = 30
        demand = 0.22
        for n1, n2 in itertools.permutations([W1, W2, E1, E2, N1, N2, S1, S2], 2):
            for t in range(0, 3600, dt):
                W.adddemand(n1, n2, t, t+dt, random.uniform(0, demand))
        
        # store UXsim object for later re-use
        self.W = W
        self.I1 = I1
        self.I2 = I2
        self.I3 = I3
        self.I4 = I4
        self.INLINKS = list(self.I1.inlinks.values()) + list(self.I2.inlinks.values()) + list(self.I3.inlinks.values()) + list(self.I4.inlinks.values())
        
        #initial observation
        observation = np.array([0 for i in range(self.n_state)])
        
        #log
        self.log_state = []
        self.log_reward = []
        
        return observation, None
    
    def comp_state(self):
        """
        compute the current state
        """
        vehicles_per_links = {}
        for l in self.INLINKS:
            vehicles_per_links[l] = l.num_vehicles_queue #l.num_vehicles_queue: the number of vehicles in queue in link l
        return list(vehicles_per_links.values())
    
    def comp_n_veh_queue(self):
        return sum(self.comp_state())
    
    def step(self, action_index):
        """
        proceed env by 1 step = `operation_timestep_width` seconds
        """
        operation_timestep_width = 10
        
        n_queue_veh_old = self.comp_n_veh_queue()
        
        #change signal by action
        #decode action
        binstr = f"{action_index:04b}"
        i1, i2, i3, i4 = int(binstr[3]), int(binstr[2]), int(binstr[1]), int(binstr[0])
        self.I1.signal_phase = i1
        self.I1.signal_t = 0
        self.I2.signal_phase = i2
        self.I2.signal_t = 0
        self.I3.signal_phase = i3
        self.I3.signal_t = 0
        self.I4.signal_phase = i4
        self.I4.signal_t = 0
        
        #traffic dynamics. execute simulation for `operation_timestep_width` seconds
        if self.W.check_simulation_ongoing():
            self.W.exec_simulation(duration_t=operation_timestep_width)
        
        #observe state
        observation = np.array(self.comp_state())
        
        #compute reward
        n_queue_veh = self.comp_n_veh_queue()
        reward = -(n_queue_veh-n_queue_veh_old)
        
        #check termination
        done = False
        if self.W.check_simulation_ongoing() == False:
            done = True
        
        #log
        self.log_state.append(observation)
        self.log_reward.append(reward)
        
        return observation, reward, done, {}, None

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
    policy_net.load_state_dict(opt.out_path+'policy_net.pth')
    target_net.load_state_dict(opt.out_path+'target_net.pth')
    optimizer.load_state_dict(opt.out_path+'optimizer.pth')
# If output path doesn't exist, create it
elif not os.path.isdir(opt.out_path):
    os.makedirs(opt.out_path)

memory = ReplayMemory(10000)

steps_done = 0
num_episodes = 200

log_states = []
log_epi_average_delay = []
best_average_delay = 9999999999999999999999999
best_W = None
best_i_episode = -1
for i_episode in range(num_episodes):
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
            log_epi_average_delay.append(env.W.analyzer.average_delay)
            print(f"{i_episode}:[{env.W.analyzer.average_delay : .3f}]", end=" ")
            if env.W.analyzer.average_delay < best_average_delay:
                print("current best episode!")
                best_average_delay = env.W.analyzer.average_delay
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
