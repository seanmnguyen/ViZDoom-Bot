#!/usr/bin/env python3

"""
DQN (Deep Q-Network) for ViZDoom - deadly_corridor scenario.
RGB input | CNN backbone + MLP Q-network.
Includes HEALTH game variable as auxiliary input to the MLP.
"""

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from utils import SCENARIO_PATH


# ── Hyperparameters ───────────────────────────────────────────────────────────

learning_rate      = 0.00025
discount_factor    = 0.99
train_epochs       = 20
steps_per_epoch    = 4000
replay_memory_size = 20000
batch_size         = 64

# Epsilon-greedy exploration
epsilon_start = 1.0
epsilon_decay = 0.9996
epsilon_min   = 0.1

# Testing
test_episodes_per_epoch = 10

# Environment
frame_repeat      = 4
resolution        = (60, 90)   # (H, W) — larger res for corridor detail
episodes_to_watch = 10

# Persistence
model_savefile = "../models/deadly_corridor/q_cnn_deadly_corridor_rgb.pth"
save_model     = True
load_model     = True
skip_learning  = True

config_file_path = os.path.join(SCENARIO_PATH, "deadly_corridor.cfg")
print(config_file_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE}")


# ── Actions ───────────────────────────────────────────────────────────────────
# Buttons: [MOVE_LEFT, MOVE_RIGHT, ATTACK, MOVE_FORWARD, MOVE_BACKWARD, TURN_LEFT, TURN_RIGHT]

actions = [list(a) for a in it.product([0, 1], repeat=7)]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_rgb(frame, resolution):
    """
    Resize RGB frame and move channels to front.
    Input:  (H, W, 3)  uint8
    Output: (3, H, W)  float32 in [0, 1]
    """
    frame = skimage.transform.resize(frame, resolution, anti_aliasing=True)
    frame = np.transpose(frame, (2, 0, 1))  # (H, W, 3) → (3, H, W)
    return frame.astype(np.float32)


# ── Game setup ────────────────────────────────────────────────────────────────

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.init()
    print("Doom initialized.")
    return game


# ── Network ───────────────────────────────────────────────────────────────────

class DuelQNet(nn.Module):
    """
    Dueling DQN with RGB CNN backbone + auxiliary health input.

    Architecture:
        RGB frame (3 x H x W)
              │
         CNN Backbone
              │
         Flatten + concat health value
              │
         Shared MLP
          ↙       ↘
    Value stream  Advantage stream
         ↘       ↙
          Q values
    """

    def __init__(self, action_size: int):
        super().__init__()

        # CNN backbone — 3 input channels for RGB
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        # Calculate CNN output size for resolution (60, 90)
        # After conv1 (k=8,s=4): (60-8)/4+1=14, (90-8)/4+1=21 → (32,14,21)
        # After conv2 (k=4,s=2): (14-4)/2+1=6,  (21-4)/2+1=9  → (64, 6, 9)
        # After conv3 (k=3,s=1): (6-3)/1+1=4,   (9-3)/1+1=7   → (64, 4, 7)
        cnn_out = 64 * 4 * 7  # 1792

        # +1 for health variable
        combined = cnn_out + 1

        # Shared MLP layer
        self.shared = nn.Sequential(
            nn.Linear(combined, 512),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frame: torch.Tensor, health: torch.Tensor) -> torch.Tensor:
        x = self.cnn(frame)
        x = x.view(x.size(0), -1)

        # Normalise health to [0, 1] and concatenate
        health = health.unsqueeze(1).float() / 100.0
        x = torch.cat([x, health], dim=1)

        x = self.shared(x)

        value     = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Dueling aggregation: Q = V + (A - mean(A))
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, health, action, reward, next_state, next_health, done):
        self.buffer.append((state, health, action, reward, next_state, next_health, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, healths, actions, rewards, next_states, next_healths, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(healths, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(next_healths, dtype=np.float32),
            np.array(dones, dtype=bool),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent:

    def __init__(
        self,
        action_size:     int,
        lr:              float = 0.00025,
        batch_size:      int   = 64,
        memory_size:     int   = 20000,
        discount_factor: float = 0.99,
        load_model:      bool  = False,
        model_weights:   str   = None,
        epsilon:         float = 1.0,
        epsilon_decay:   float = 0.9996,
        epsilon_min:     float = 0.1,
    ):
        self.action_size   = action_size
        self.batch_size    = batch_size
        self.discount      = discount_factor
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min

        self.memory    = ReplayBuffer(memory_size)
        self.criterion = nn.SmoothL1Loss()  # Huber loss — more stable than MSE

        self.q_net      = DuelQNet(action_size).to(DEVICE)
        self.target_net = DuelQNet(action_size).to(DEVICE)

        if load_model and model_weights:
            print(f"Loading model from: {model_weights}")
            sd = torch.load(model_weights, map_location=DEVICE)
            self.q_net.load_state_dict(sd)
            self.target_net.load_state_dict(sd)
            self.epsilon = epsilon_min

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.q_net.eval()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    # ── Inference ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, state: np.ndarray, state_vars=None, eval_mode: bool = False) -> int:
        """
        state_vars: ignored (health is read from state_vars[0] if provided,
                    otherwise defaults to 100). Kept for eval_models.py compatibility.
        """
        # Extract health from state_vars if provided by eval script
        if state_vars is not None and len(state_vars) > 0:
            health = float(state_vars[0])
        else:
            health = 100.0

        if not eval_mode and np.random.uniform() < self.epsilon:
            return random.randrange(self.action_size)

        frame_t  = torch.from_numpy(state).unsqueeze(0).float().to(DEVICE)
        health_t = torch.tensor([health]).float().to(DEVICE)
        return self.q_net(frame_t, health_t).argmax(dim=1).item()

    def store(self, state, health, action, reward, next_state, next_health, done):
        self.memory.push(state, health, action, reward, next_state, next_health, done)

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    # ── Training step ──────────────────────────────────────────────────────

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        states, healths, actions, rewards, next_states, next_healths, dones = \
            self.memory.sample(self.batch_size)

        states_t       = torch.from_numpy(states).float().to(DEVICE)
        healths_t      = torch.from_numpy(healths).to(DEVICE)
        actions_t      = torch.from_numpy(actions).to(DEVICE)
        rewards_t      = torch.from_numpy(rewards).to(DEVICE)
        next_states_t  = torch.from_numpy(next_states).float().to(DEVICE)
        next_healths_t = torch.from_numpy(next_healths).to(DEVICE)
        not_dones      = torch.from_numpy(~dones).to(DEVICE)

        # Double DQN: select action with q_net, evaluate with target_net
        with torch.no_grad():
            next_actions = self.q_net(next_states_t, next_healths_t).argmax(dim=1)
            next_q       = self.target_net(next_states_t, next_healths_t)
            next_q_vals  = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets      = rewards_t + self.discount * next_q_vals * not_dones.float()

        self.q_net.train()
        current_q = self.q_net(states_t, healths_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss      = self.criterion(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.q_net.eval()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_state(game):
    """Return (frame, health) for the current game state."""
    gs     = game.get_state()
    frame  = preprocess_rgb(gs.screen_buffer, resolution)
    health = gs.game_variables[0]  # HEALTH
    return frame, health


# ── Test loop ─────────────────────────────────────────────────────────────────

def test(game, agent, num_episodes: int = 10):
    print("\nTesting...")
    scores = []
    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            frame, health = get_state(game)
            action = agent.get_action(frame, state_vars=[health], eval_mode=True)
            game.make_action(actions[action], frame_repeat)
        scores.append(game.get_total_reward())

    scores = np.array(scores)
    print(
        f"Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}, "
        f"min: {scores.min():.1f}, max: {scores.max():.1f}"
    )
    return scores.mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def run(game, agent, num_epochs, steps_per_epoch, frame_repeat):
    start_time       = time()
    best_mean_reward = float("-inf")

    for epoch in range(num_epochs):
        print(f"\n{'='*50}\nEpoch #{epoch + 1}\n{'='*50}")

        game.new_episode()
        train_scores   = []
        episode_reward = 0.0
        global_step    = 0

        for _ in trange(steps_per_epoch, desc="Training", leave=False):
            frame, health = get_state(game)
            action = agent.get_action(frame, state_vars=[health])

            reward = game.make_action(actions[action], frame_repeat)
            done   = game.is_episode_finished()
            episode_reward += reward

            if not done:
                next_frame, next_health = get_state(game)
            else:
                next_frame  = np.zeros((3, *resolution), dtype=np.float32)
                next_health = 0.0

            agent.store(frame, health, action, reward, next_frame, next_health, done)
            agent.train()

            if done:
                train_scores.append(episode_reward)
                episode_reward = 0.0
                game.new_episode()

            global_step += 1

        agent.update_target_net()

        if train_scores:
            s = np.array(train_scores)
            print(
                f"\nTrain ({len(s)} eps): "
                f"mean {s.mean():.1f} ± {s.std():.1f}  "
                f"min {s.min():.1f}  max {s.max():.1f}  "
                f"epsilon {agent.epsilon:.3f}"
            )

        mean_reward = test(game, agent, test_episodes_per_epoch)

        if save_model and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print(f"New best ({mean_reward:.1f})! Saving → {model_savefile}")
            agent.save(model_savefile)

        print(f"Elapsed: {(time() - start_time) / 60:.2f} min")

    game.close()
    return agent, game


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game = create_simple_game()
    print(f"Action space size: {len(actions)}")

    agent = DQNAgent(
        action_size     = len(actions),
        memory_size     = replay_memory_size,
        batch_size      = batch_size,
        discount_factor = discount_factor,
        lr              = learning_rate,
        epsilon         = epsilon_start,
        epsilon_decay   = epsilon_decay,
        epsilon_min     = epsilon_min,
        load_model      = load_model,
        model_weights   = model_savefile if load_model else None,
    )

    if not skip_learning:
        agent, game = run(
            game, agent,
            num_epochs      = train_epochs,
            steps_per_epoch = steps_per_epoch,
            frame_repeat    = frame_repeat,
        )
        print("\n" + "="*50 + "\nTraining finished. It's time to watch!\n" + "="*50)

    # Watch the agent play
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    total_score = 0
    for ep in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            gs = game.get_state()
            assert gs is not None
            frame  = preprocess_rgb(gs.screen_buffer, resolution)
            health = gs.game_variables[0]
            action = agent.get_action(frame, state_vars=[health], eval_mode=True)
            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score        = game.get_total_reward()
        total_score += score
        print(f"Episode {ep + 1} Total Score: {score}")

    print(f"\n----- Average Score: {total_score / episodes_to_watch:.1f} -----")
    game.close()