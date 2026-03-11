#!/usr/bin/env python3

"""
test run 1:
==================================================
Episode 1 Total Score: 305.91168212890625
Episode 2 Total Score: 51.98722839355469
Episode 3 Total Score: 67.29446411132812
Episode 4 Total Score: 416.0734100341797
Episode 5 Total Score: 28.7291259765625
Episode 6 Total Score: 50.899810791015625
Episode 7 Total Score: 177.4473876953125
Episode 8 Total Score: 32.54685974121094
Episode 9 Total Score: -31.660614013671875
Episode 10 Total Score: 70.52247619628906

-----Average Score: 116.97518310546874-----
"""

"""
test run 2:
==================================================
Episode 1 Total Score: 419.87193298339844
Episode 2 Total Score: 460.5805206298828
Episode 3 Total Score: 393.7855987548828
Episode 4 Total Score: 469.3984832763672
Episode 5 Total Score: 440.26678466796875
Episode 6 Total Score: 310.6322479248047
Episode 7 Total Score: 392.44020080566406
Episode 8 Total Score: 467.6341857910156
Episode 9 Total Score: 389.23020935058594
Episode 10 Total Score: 380.39549255371094

-----Average Score: 412.42356567382814-----
"""

import itertools as it
import math
import os
from collections import deque
from time import sleep, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

import vizdoom as vzd
from utils import preprocess_rgb, SCENARIO_PATH, RESOLUTION, get_num_game_variables

NET_NUM_VARS = 2

learning_rate = 2.5e-4
gamma = 0.99            # discount factor
gae_lambda = 0.95       # GAE lambda for advantage estimation
clip_epsilon = 0.2      # PPO clip parameter
entropy_coef_start = 0.02  # initial entropy bonus
entropy_coef_end = 0.002   # final entropy bonus
value_coef = 0.5        # value loss coefficient
max_grad_norm = 0.5     # gradient clipping
value_clip_range = 0.2  # value function clipping range

FRAME_STACK_SIZE = 4
RGB_CHANNELS = 3
INPUT_CHANNELS = FRAME_STACK_SIZE * RGB_CHANNELS  # 12

train_epochs = 60
steps_per_epoch = 8192    # larger rollouts for more stable updates
ppo_epochs = 6            # more PPO passes per batch
mini_batch_size = 128     # larger mini-batch for stability

HEALTH_LOSS_PENALTY = 0.5    # penalty per health point lost (higher = cautious play)
SURVIVAL_BONUS = 0.02        # small reward per step alive
KILL_REWARD = 20.0           # large reward per enemy killed
DISTANCE_REWARD_SCALE = 0.3  # scale down the raw distance reward so kills dominate
ADVANCE_WHILE_ENEMIES_PENALTY = 0.5  # penalty for moving forward when recently taking damage
ENEMIES_PER_ROOM = 2         # expected enemies per room section

test_episodes_per_epoch = 100

frame_repeat = 4          # lower = finer movement/aiming control
resolution = RESOLUTION   # (96, 128)
episodes_to_watch = 10

SCENARIO_NAME = "deadly_corridor"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

model_savefile = f"../models/{SCENARIO_NAME}/ppo_late_fusion_rgb.pth"
save_model = True
load_model = True
skip_learning = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

NUM_VARS = get_num_game_variables(config_file_path)

print(config_file_path)
print(f"Using device: {DEVICE}")
print(f"Number of game variables (cfg): {NUM_VARS}")
print(f"Number of network input vars: {NET_NUM_VARS}")

# total enemies in deadly_corridor
TOTAL_ENEMIES = 6


def preprocess_vars_corridor(game_variables, kills=0):
    health = float(game_variables[0]) if len(game_variables) >= 1 else 0.0
    health_norm = np.clip(health, 0.0, 100.0) / 100.0
    kills_norm = np.clip(float(kills), 0.0, float(TOTAL_ENEMIES)) / float(TOTAL_ENEMIES)
    return np.array([health_norm, kills_norm], dtype=np.float32)

def get_deadly_corridor_actions():
    actions = [

        [0, 0, 0, 1, 0, 0, 0],  # forward
        [0, 0, 0, 0, 1, 0, 0],  # backward
        [1, 0, 0, 0, 0, 0, 0],  # strafe left
        [0, 1, 0, 0, 0, 0, 0],  # strafe right
        [0, 0, 0, 0, 0, 1, 0],  # turn left
        [0, 0, 0, 0, 0, 0, 1],  # turn right

        [1, 0, 0, 1, 0, 0, 0],  # forward + strafe left
        [0, 1, 0, 1, 0, 0, 0],  # forward + strafe right

        [0, 0, 0, 1, 0, 1, 0],  # forward + turn left
        [0, 0, 0, 1, 0, 0, 1],  # forward + turn right

        [0, 0, 1, 0, 0, 0, 0],  # shoot (stationary)
        [0, 0, 1, 0, 0, 1, 0],  # shoot + turn left
        [0, 0, 1, 0, 0, 0, 1],  # shoot + turn right

        [1, 0, 1, 0, 0, 0, 0],  # strafe left + shoot
        [0, 1, 1, 0, 0, 0, 0],  # strafe right + shoot
        [1, 0, 1, 0, 0, 1, 0],  # strafe left + shoot + turn left
        [0, 1, 1, 0, 0, 0, 1],  # strafe right + shoot + turn right

        [0, 0, 1, 1, 0, 0, 0],  # shoot + forward
        [1, 0, 1, 1, 0, 0, 0],  # shoot + forward + strafe left
        [0, 1, 1, 1, 0, 0, 0],  # shoot + forward + strafe right

        [0, 0, 1, 0, 1, 0, 0],  # shoot + backward

        [0, 0, 0, 0, 0, 0, 0],
    ]
    return actions


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.init()
    print("Doom initialized.")
    return game


def preprocess_rgb_normalized(img, res=resolution):
    x = preprocess_rgb(img, res)
    x /= 255.0
    return x


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv(combined))
        return x * attn


class SEBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)
        return x * s


class ResBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        return out


class StrongCNN(nn.Module):

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ResBlock(32, 32, stride=1, use_se=True)
        self.stage2 = nn.Sequential(
            ResBlock(32, 64, stride=2, use_se=True),
            ResBlock(64, 64, stride=1, use_se=True),
        )
        self.stage3 = nn.Sequential(
            ResBlock(64, 96, stride=2, use_se=True),
            ResBlock(96, 96, stride=1, use_se=True),
        )

        self.context = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.spatial_attn = SpatialAttention(kernel_size=7)

        self.pool = nn.AdaptiveAvgPool2d((4, 8))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.context(x)
        x = self.spatial_attn(x) 
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ActorCriticLateFusion(nn.Module):

    def __init__(
        self,
        action_size: int,
        num_vars: int,
        in_channels: int = 12,
        img_h: int = 96,
        img_w: int = 128,
    ):
        super().__init__()

        self.cnn = StrongCNN(in_channels=in_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            cnn_dim = self.cnn(dummy).shape[1]

        self.img_fc = nn.Sequential(
            nn.Linear(cnn_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.vars_mlp = nn.Sequential(
            nn.LayerNorm(num_vars),
            nn.Linear(num_vars, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = 256 + 64  # 320

        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
        )

        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)
        self._initialize_heads()

    def _initialize_heads(self):
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, img, vars_):
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = torch.cat([img_feat, vars_feat], dim=1)
        features = self.shared_fc(fused)
        return features

    def get_action_and_value(self, img, vars_, action=None):
        features = self.forward(img, vars_)

        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, img, vars_):
        features = self.forward(img, vars_)
        return self.critic(features).squeeze(-1)


class FrameStackRGB:
    def __init__(self, stack_size, frame_shape, channels=RGB_CHANNELS):
        self.stack_size = stack_size
        self.channels = channels
        self.frame_shape = frame_shape  # (H, W)
        self.frames = deque(maxlen=stack_size)
        self.reset()

    def reset(self):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(
                np.zeros((self.channels, *self.frame_shape), dtype=np.float32)
            )

    def push(self, frame):
        self.frames.append(frame)

    def get(self):
        return np.concatenate(list(self.frames), axis=0)


class RolloutBuffer:
    def __init__(self):
        self.states_img = []
        self.states_vars = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state_img, state_vars, action, reward, done, log_prob, value):
        self.states_img.append(state_img)
        self.states_vars.append(state_vars)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states_img.clear()
        self.states_vars.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * \
                next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * \
                gae_lambda * next_non_terminal * last_gae

        returns = advantages + np.array(self.values)
        return returns, advantages

    def get_batches(self, batch_size, returns, advantages):
        n_samples = len(self.states_img)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                np.array([self.states_img[i] for i in batch_indices]),
                np.array([self.states_vars[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                returns[batch_indices],
                advantages[batch_indices],
            )


class PPOAgent:
    def __init__(
        self,
        action_size,
        num_vars=NET_NUM_VARS,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=64,
        load_model_path=None,
    ):
        self.action_size = action_size
        self.num_vars = num_vars
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        # Initialize network
        if load_model_path:
            print(f"Loading PPO Late Fusion model from: {load_model_path}")
            self.network = ActorCriticLateFusion(
                action_size, num_vars,
                in_channels=INPUT_CHANNELS,
                img_h=resolution[0], img_w=resolution[1],
            ).to(DEVICE)
            self.network.load_state_dict(torch.load(
                load_model_path, map_location=DEVICE))
            self.network.eval()
        else:
            print("Initializing new PPO Late Fusion model")
            self.network = ActorCriticLateFusion(
                action_size, num_vars,
                in_channels=INPUT_CHANNELS,
                img_h=resolution[0], img_w=resolution[1],
            ).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def get_action(self, state_img, state_vars, deterministic=False):
        img = np.expand_dims(state_img, axis=0)
        vars_ = np.expand_dims(state_vars, axis=0)
        img_t = torch.from_numpy(img).float().to(DEVICE)
        vars_t = torch.from_numpy(vars_).float().to(DEVICE)

        with torch.no_grad():
            if deterministic:
                features = self.network(img_t, vars_t)
                logits = self.network.actor(features)
                action = torch.argmax(logits, dim=-1).item()
                return action
            else:
                action, log_prob, _, value = self.network.get_action_and_value(
                    img_t, vars_t)
                return action.item(), log_prob.item(), value.item()

    def store_transition(self, state_img, state_vars, action, reward, done, log_prob, value):
        self.buffer.add(state_img, state_vars, action, reward, done, log_prob, value)

    def get_last_value(self, state_img, state_vars):
        img = np.expand_dims(state_img, axis=0)
        vars_ = np.expand_dims(state_vars, axis=0)
        img_t = torch.from_numpy(img).float().to(DEVICE)
        vars_t = torch.from_numpy(vars_).float().to(DEVICE)
        with torch.no_grad():
            return self.network.get_value(img_t, vars_t).item()

    def update_entropy_coef(self, progress):
        self.entropy_coef = entropy_coef_start + \
            (entropy_coef_end - entropy_coef_start) * progress

    def update_lr(self, progress):
        lr = learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def train(self):
        # Get last value for GAE
        last_img = self.buffer.states_img[-1] if self.buffer.states_img else np.zeros(
            (INPUT_CHANNELS, *resolution))
        last_vars = self.buffer.states_vars[-1] if self.buffer.states_vars else np.zeros(
            (NET_NUM_VARS,))
        last_value = self.get_last_value(
            last_img, last_vars) if not self.buffer.dones[-1] else 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

        # Store old values for value clipping
        old_values = np.array(self.buffer.values)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, returns, advantages):
                imgs, vars_, actions, old_log_probs, batch_returns, batch_advantages = batch

                imgs_t = torch.from_numpy(imgs).float().to(DEVICE)
                vars_t = torch.from_numpy(vars_).float().to(DEVICE)
                actions_t = torch.from_numpy(actions).long().to(DEVICE)
                old_log_probs_t = torch.from_numpy(
                    old_log_probs).float().to(DEVICE)
                batch_returns_t = torch.from_numpy(
                    batch_returns).float().to(DEVICE)
                batch_advantages_t = torch.from_numpy(
                    batch_advantages).float().to(DEVICE)

                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    imgs_t, vars_t, actions_t)

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * batch_advantages_t
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss_unclipped = (values - batch_returns_t) ** 2
                value_loss = 0.5 * value_loss_unclipped.mean()

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * \
                    value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=DEVICE))


def test(game, agent, actions, num_episodes=100):
    print("\nTesting...")
    test_scores = []
    frame_stack = FrameStackRGB(FRAME_STACK_SIZE, resolution, RGB_CHANNELS)

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        frame_stack.reset()
        test_kills = 0
        while not game.is_episode_finished():
            game_state = game.get_state()
            frame = preprocess_rgb_normalized(game_state.screen_buffer)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_corridor(game_state.game_variables, kills=test_kills)
            action = agent.get_action(state_img, state_vars, deterministic=True)
            game.make_action(actions[action], frame_repeat)

            if not game.is_episode_finished():
                gv = game.get_state().game_variables
                test_kills = gv[1] if len(gv) > 1 else 0
        test_scores.append(game.get_total_reward())

    test_scores = np.array(test_scores)
    print(
        f"Results: mean: {test_scores.mean():.1f} +/- {test_scores.std():.1f}, "
        f"min: {test_scores.min():.1f}, max: {test_scores.max():.1f}"
    )
    return test_scores.mean()


def run(game, agent, actions, num_epochs, steps_per_epoch, frame_repeat):
    start_time = time()
    best_mean_reward = float("-inf")

    frame_stack = FrameStackRGB(FRAME_STACK_SIZE, resolution, RGB_CHANNELS)

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch #{epoch + 1}")
        print(f"{'='*50}")

        progress = epoch / max(num_epochs - 1, 1)
        current_lr = agent.update_lr(progress)
        agent.update_entropy_coef(progress)
        print(f"  LR: {current_lr:.6f}  Entropy coef: {agent.entropy_coef:.4f}")

        game.new_episode()
        frame_stack.reset()
        train_scores = []
        episode_reward = 0
        prev_health = 100.0   # starting health
        prev_kills = 0        # starting kill count
        taking_damage = False # track if agent is under fire

        for step in trange(steps_per_epoch, desc="Collecting rollout", leave=False):
            game_state = game.get_state()
            frame = preprocess_rgb_normalized(game_state.screen_buffer)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_corridor(game_state.game_variables, kills=prev_kills)

            action, log_prob, value = agent.get_action(state_img, state_vars)

            raw_reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            shaped_reward = raw_reward * DISTANCE_REWARD_SCALE

            if not done:
                gv = game.get_state().game_variables
                current_health = gv[0]
                current_kills = gv[1]  # killcount

                # penalize health loss (encourages taking cover)
                health_delta = current_health - prev_health
                if health_delta < 0:
                    shaped_reward += health_delta * HEALTH_LOSS_PENALTY
                    taking_damage = True
                else:
                    taking_damage = False

                # larger reward for kills (incentive to clear rooms)
                new_kills = current_kills - prev_kills
                if new_kills > 0:
                    shaped_reward += new_kills * KILL_REWARD

                # penalize if room not cleared
                action_vec = actions[action]
                is_moving_forward = action_vec[3] == 1
                room_kills = current_kills % ENEMIES_PER_ROOM  # kills within current room
                if is_moving_forward and taking_damage and room_kills < ENEMIES_PER_ROOM:
                    shaped_reward -= ADVANCE_WHILE_ENEMIES_PENALTY

                shaped_reward += SURVIVAL_BONUS

                prev_health = current_health
                prev_kills = current_kills

            episode_reward += raw_reward  # track raw reward for logging
            agent.store_transition(state_img, state_vars, action, shaped_reward,
                                   done, log_prob, value)

            if done:
                train_scores.append(episode_reward)
                episode_reward = 0
                game.new_episode()
                frame_stack.reset()
                prev_health = 100.0
                prev_kills = 0
                taking_damage = False

        # PPO update
        stats = agent.train()

        if train_scores:
            train_scores = np.array(train_scores)
            print(f"\nTraining episodes: {len(train_scores)}")
            print(
                f"Episode rewards: mean: {train_scores.mean():.1f} +/- {train_scores.std():.1f}, "
                f"min: {train_scores.min():.1f}, max: {train_scores.max():.1f}"
            )

        print(f"\nPPO Update Stats:")
        print(f"  Policy Loss: {stats['policy_loss']:.4f}")
        print(f"  Value Loss: {stats['value_loss']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")

        mean_test_reward = test(game, agent, actions, test_episodes_per_epoch)

        if save_model and mean_test_reward > best_mean_reward:
            best_mean_reward = mean_test_reward
            print(f"New best model! Saving to: {model_savefile}")
            agent.save(model_savefile)

        print(
            f"Total elapsed time: {(time() - start_time) / 60.0:.2f} minutes")

    game.close()
    return agent, game


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()

    actions = get_deadly_corridor_actions()

    print(f"Number of actions: {len(actions)}")

    # Initialize PPO agent
    agent = PPOAgent(
        action_size=len(actions),
        num_vars=NET_NUM_VARS,
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef_start,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        load_model_path=model_savefile if load_model else None,
    )

    # Run training
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            steps_per_epoch=steps_per_epoch,
            frame_repeat=frame_repeat,
        )

        print("\n" + "=" * 50)
        print("Training finished. It's time to watch!")
        print("=" * 50)

    # Watch the trained agent play
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    frame_stack = FrameStackRGB(FRAME_STACK_SIZE, resolution, RGB_CHANNELS)
    total_score = 0
    for episode_num in range(episodes_to_watch):
        game.new_episode()
        frame_stack.reset()
        watch_kills = 0
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            frame = preprocess_rgb_normalized(game_state.screen_buffer)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_corridor(game_state.game_variables,
                                                  kills=watch_kills)
            action = agent.get_action(state_img, state_vars, deterministic=True)

            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

            if not game.is_episode_finished():
                gv = game.get_state().game_variables
                watch_kills = gv[1] if len(gv) > 1 else 0

        sleep(1.0)
        score = game.get_total_reward()
        total_score += score
        print(f"Episode {episode_num + 1} Total Score: {score}")

    print(f"\n-----Average Score: {total_score / episodes_to_watch}-----")
    game.close()
