##
## Adapted from Jupyter Notebook example here: https://github.com/toruseo/UXsim
##

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

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

import copy
import argparse
import sys
from TrafficSim import TrafficSim

# Increase recursion limit
sys.setrecursionlimit(10000)

# ────────── Command‑line arguments ────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of transitions sampled from the replay buffer')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='The discount factor')
parser.add_argument('--eps_start', type=float, default=0.9,
                    help='The starting value of epsilon')
parser.add_argument('--eps_end', type=float, default=0.05,
                    help='The final value of epsilon')
parser.add_argument('--eps_decay', type=int, default=1000,
                    help='The rate of exponential decay of epsilon, higher means a slower decay')
parser.add_argument('--tau', type=float, default=0.005,
                    help='The update rate of the target network')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='The learning rate of the optimizer')
parser.add_argument('--out_path', type=str, default='./out/',
                    help='The path to save model and optimizer states to')
parser.add_argument('--device', type=str, default=None,
                    help='Force training to run on this device')
parser.add_argument('--num_episodes', type=int, default=200,
                    help='Number of episodes to train for')
parser.add_argument('--mem_size', type=int, default=10000,
                    help='Replay memory capacity')
parser.add_argument('--eval_only', action='store_true',
                    help='Skip training; just load weights and plot')
opt = parser.parse_args()
print(opt)

# ────────── Device ────────────────────────────────────────────
device = torch.device(opt.device) if opt.device else \
         torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── Environment ───────────────────────────────────────
env = TrafficSim()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        n_neurals = 64
        n_layers = 3
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_observations, n_neurals))
        for _ in range(n_layers):
            self.layers.append(nn.Linear(n_neurals, n_neurals))
        self.layer_last = nn.Linear(n_neurals, n_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.layer_last(x)

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = opt.eps_end + (opt.eps_start - opt.eps_end) * \
                    math.exp(-1. * steps_done / opt.eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device,
                            dtype=torch.long)

def optimize_model():
    if len(memory) < opt.batch_size:
        return
    transitions = memory.sample(opt.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(opt.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = \
            target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * opt.gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# ────────── Networks, memory, optimiser ───────────────────────
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=opt.lr, amsgrad=True)

if os.path.exists(opt.out_path + 'policy_net.pth'):
    policy_net.load_state_dict(torch.load(opt.out_path + 'policy_net.pth',
                                          map_location=device))
    target_net.load_state_dict(torch.load(opt.out_path + 'target_net.pth',
                                          map_location=device))
    optimizer.load_state_dict(torch.load(opt.out_path + 'optimizer.pth',
                                         map_location=device))
elif not os.path.isdir(opt.out_path):
    os.makedirs(opt.out_path)

memory = ReplayMemory(opt.mem_size)
steps_done = 0

# ────────── Training loop (skipped if --eval_only) ────────────
log_states = []
log_epi_average_delay = []
best_average_delay = float('inf')
best_W = None
best_i_episode = -1

if not opt.eval_only:
    for i_episode in range(opt.num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32,
                             device=device).unsqueeze(0)
        log_states.append([])
        for t in count():
            action = select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if done else \
                torch.tensor(obs, dtype=torch.float32,
                             device=device).unsqueeze(0)

            log_states[-1].append(state)
            memory.push(state, action, next_state, reward)
            state = next_state

            optimize_model()

            # Soft update
            for key in policy_net.state_dict():
                target_net.state_dict()[key] = \
                    policy_net.state_dict()[key] * opt.tau + \
                    target_net.state_dict()[key] * (1 - opt.tau)

            if done:
                log_epi_average_delay.append(env.W.analyzer.average_delay)
                print(f"{i_episode}: [{env.W.analyzer.average_delay: .3f}] ",
                      end="")
                if env.W.analyzer.average_delay < best_average_delay:
                    print("← best so far")
                    best_average_delay = env.W.analyzer.average_delay
                    best_W = copy.deepcopy(env.W)
                    best_i_episode = i_episode
                    torch.save(policy_net.state_dict(),
                               opt.out_path + f'policy_net_{i_episode:02d}.pth')
                    torch.save(policy_net.state_dict(),
                               opt.out_path + 'policy_net.pth')
                    torch.save(target_net.state_dict(),
                               opt.out_path + f'target_net_{i_episode:02d}.pth')
                    torch.save(target_net.state_dict(),
                               opt.out_path + 'target_net.pth')
                    torch.save(optimizer.state_dict(),
                               opt.out_path + 'optimizer.pth')
                else:
                    print()
                break

# ──────────────────────────────────────────────────────────────
#  LOAD BEST POLICY AND RUN ONE EPISODE FOR VISUALISATION
# ──────────────────────────────────────────────────────────────
def evaluate(policy_path=r"C:\Users\micha\OneDrive\Desktop\C2\policy_net_10.pth",
             target_path=r"C:\Users\micha\OneDrive\Desktop\C2\target_net.pth",
             num_eval_episodes=1,
             render=False):
    """Load a trained policy and run evaluation episodes."""
    policy_net.load_state_dict(torch.load(policy_path,
                                          map_location=device))
    policy_net.eval()

    global steps_done
    steps_done = 10 ** 9           # ε ≈ ε_end immediately

    all_avg_delay = []

    with torch.no_grad():
        for _ in range(num_eval_episodes):
            if best_W is not None:
                state, _ = env.reset(W=copy.deepcopy(best_W))
            else:
                state, _ = env.reset()

            state = torch.tensor(state, dtype=torch.float32,
                                 device=device).unsqueeze(0)
            done = False
            while not done:
                action = policy_net(state).max(1)[1].view(1, 1)
                obs, _, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                state = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                if render and hasattr(env, "render"):
                    env.render()
            all_avg_delay.append(env.W.analyzer.average_delay)

    return all_avg_delay

if __name__ == "__main__":
    evaluate()

    # ────────── VISUALISATIONS ────────────────────────────────
    env.W.analyzer.print_simple_stats(force_print=True)
    env.W.analyzer.macroscopic_fundamental_diagram()

    env.W.analyzer.time_space_diagram_traj_links(
        [["W1I1", "I1I2", "I2E1"],
         ["N1I1", "I1I3", "I3S1"]],
        figsize=(12, 12))

    for t in range(0, env.W.TMAX, int(env.W.TMAX / 4)):
        env.W.analyzer.network(t, detailed=1,
                               network_font_size=0, figsize=(3, 3))

    plt.figure(figsize=(4, 3))
    if log_epi_average_delay:       # not empty if you trained this run
        plt.plot(log_epi_average_delay, "r.")
    plt.xlabel("episode")
    plt.ylabel("average delay (s)")
    plt.grid()
    plt.tight_layout()
    plt.show()
