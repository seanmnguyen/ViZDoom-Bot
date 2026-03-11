#!/usr/bin/env python3
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
from utils import preprocess, SCENARIO_PATH, RESOLUTION


learning_rate = 2.5e-4
gamma = 0.99                 # discount factor
gae_lambda = 0.95            # GAE lambda for advantage estimation
clip_epsilon = 0.2           # PPO clip parameter
entropy_coef_start = 0.02    # initial entropy bonus (encourages exploration)
entropy_coef_end = 0.002     # final entropy bonus (exploitation)
value_coef = 0.5             # value loss coefficient
max_grad_norm = 0.5          # gradient clipping
value_clip_range = 0.2       # value function clipping range

FRAME_STACK_SIZE = 4

train_epochs = 80
steps_per_epoch = 8192       # larger rollouts for more stable updates
ppo_epochs = 6               # PPO passes per rollout batch
mini_batch_size = 256        # mini-batch size for PPO updates

test_episodes_per_epoch = 50

frame_repeat = 4             # low = finer control for fast-paced combat
resolution = RESOLUTION      # (96, 128)
episodes_to_watch = 10

KILL_BONUS = 5.0             # bonus per enemy killed
HEALTH_LOSS_PENALTY = 0.01   # penalty per health point lost
HEALTH_GAIN_BONUS = 0.02     # reward per health point picked up
ARMOR_GAIN_BONUS = 0.01      # reward per armor point picked up
DEATH_PENALTY = -1.0         # penalty for dying
SURVIVAL_BONUS = 0.01        # small reward per step alive
AMMO_PICKUP_BONUS = 0.005   # reward per ammo picked up

SCENARIO_NAME = "deathmatch"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

model_savefile = "../models/deathmatch/ppo_cnn_gray.pth"
save_model = True
load_model = True
skip_learning = True

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


print(config_file_path)
print(f"Using device: {DEVICE}")


def get_deathmatch_actions():
    """
    Curated discrete action set for deathmatch.

    Buttons (set in create_game):
      0: ATTACK
      1: SPEED
      2: MOVE_FORWARD
      3: MOVE_BACKWARD
      4: MOVE_LEFT   (strafe)
      5: MOVE_RIGHT  (strafe)
      6: TURN_LEFT
      7: TURN_RIGHT
      8: SELECT_NEXT_WEAPON
    """
    #                               ATK SPD FWD BWD  ML  MR  TL  TR WPN
    actions = [
        [0,  0,  1,  0,  0,  0,  0,  0,  0],   # forward
        [0,  0,  0,  1,  0,  0,  0,  0,  0],   # backward
        [0,  0,  0,  0,  1,  0,  0,  0,  0],   # strafe left
        [0,  0,  0,  0,  0,  1,  0,  0,  0],   # strafe right
        [0,  0,  0,  0,  0,  0,  1,  0,  0],   # turn left
        [0,  0,  0,  0,  0,  0,  0,  1,  0],   # turn right

        [0,  1,  1,  0,  0,  0,  0,  0,  0],   # sprint forward
        [0,  1,  1,  0,  1,  0,  0,  0,  0],   # sprint forward + strafe left
        [0,  1,  1,  0,  0,  1,  0,  0,  0],   # sprint forward + strafe right

        [0,  0,  1,  0,  1,  0,  0,  0,  0],   # forward + strafe left
        [0,  0,  1,  0,  0,  1,  0,  0,  0],   # forward + strafe right
        [0,  0,  1,  0,  0,  0,  1,  0,  0],   # forward + turn left
        [0,  0,  1,  0,  0,  0,  0,  1,  0],   # forward + turn right

        [1,  0,  0,  0,  0,  0,  0,  0,  0],   # attack

        [1,  0,  0,  0,  0,  0,  1,  0,  0],   # attack + turn left
        [1,  0,  0,  0,  0,  0,  0,  1,  0],   # attack + turn right

        [1,  0,  0,  0,  1,  0,  0,  0,  0],   # attack + strafe left
        [1,  0,  0,  0,  0,  1,  0,  0,  0],   # attack + strafe right

        [1,  0,  1,  0,  0,  0,  0,  0,  0],   # attack + forward
        [1,  0,  1,  0,  0,  0,  1,  0,  0],   # attack + forward + turn left
        [1,  0,  1,  0,  0,  0,  0,  1,  0],   # attack + forward + turn right
        [1,  0,  1,  0,  1,  0,  0,  0,  0],   # attack + forward + strafe left
        [1,  0,  1,  0,  0,  1,  0,  0,  0],   # attack + forward + strafe right

        [1,  0,  0,  0,  1,  0,  0,  1,  0],   # attack + strafe left  + turn right
        [1,  0,  0,  0,  0,  1,  1,  0,  0],   # attack + strafe right + turn left

        [1,  0,  0,  1,  0,  0,  0,  0,  0],   # attack + backward
        [1,  0,  0,  1,  0,  0,  1,  0,  0],   # attack + backward + turn left
        [1,  0,  0,  1,  0,  0,  0,  1,  0],   # attack + backward + turn right

        [1,  1,  1,  0,  0,  0,  0,  0,  0],   # sprint + attack + forward

        [0,  0,  0,  0,  0,  0,  0,  0,  1],   # next weapon

        [0,  0,  0,  0,  0,  0,  0,  0,  0],   # do nothing
    ]
    return actions


def create_game():
    """Initialize ViZDoom deathmatch game with grayscale rendering."""
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    game.clear_available_buttons()
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.SPEED)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)
    game.add_available_button(vzd.Button.SELECT_NEXT_WEAPON)

    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.ARMOR)
    game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON)
    game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)

    game.init()
    print("Doom initialized.")
    print(f"  Buttons:        {game.get_available_buttons_size()}")
    print(f"  Game variables: {game.get_available_game_variables_size()}")
    return game


class RewardShaper:
    """Track game variable deltas and compute shaped rewards each step."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_kills = 0
        self.prev_health = 100.0
        self.prev_armor = 0.0
        self.prev_ammo = 0.0

    def shape_reward(self, game_reward, game_vars, done):
        """
        Compute shaped reward from game variable deltas.

        Game variable indices (from create_game):
          0: KILLCOUNT  1: HEALTH  2: ARMOR  3: SELECTED_WEAPON  4: SELECTED_WEAPON_AMMO
        """
        if game_vars is None or len(game_vars) < 5:
            return game_reward

        kills = game_vars[0]
        health = game_vars[1]
        armor = game_vars[2]
        ammo = game_vars[4]

        shaped = game_reward

        # Kill bonus
        new_kills = kills - self.prev_kills
        if new_kills > 0:
            shaped += new_kills * KILL_BONUS

        # Health change
        health_delta = health - self.prev_health
        if health_delta < 0:
            shaped += health_delta * HEALTH_LOSS_PENALTY   # negative → penalty
        elif health_delta > 0:
            shaped += health_delta * HEALTH_GAIN_BONUS     # picked up medkit

        # Armor change
        armor_delta = armor - self.prev_armor
        if armor_delta > 0:
            shaped += armor_delta * ARMOR_GAIN_BONUS

        # Ammo pickup
        ammo_delta = ammo - self.prev_ammo
        if ammo_delta > 0:
            shaped += ammo_delta * AMMO_PICKUP_BONUS

        # Survival bonus
        shaped += SURVIVAL_BONUS

        # Death penalty
        if done and health <= 0:
            shaped += DEATH_PENALTY

        # Update tracking
        self.prev_kills = kills
        self.prev_health = health
        self.prev_armor = armor
        self.prev_ammo = ammo

        return shaped


class ResidualBlock(nn.Module):
    """Pre-activation residual block (no BN, matches IMPALA paper)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x


class IMPALABlock(nn.Module):
    """Conv → MaxPool(stride 2) → ResBlock → ResBlock."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class DeathmatchCNN(nn.Module):
    """
    IMPALA-style Actor-Critic CNN for deathmatch.

    Input : (batch, FRAME_STACK_SIZE, 96, 128)
    Blocks: [32, 64, 64] channels with stride-2 max-pooling each
            → feature map (64, 12, 16)  → 12 288 features
    FC    : 12 288 → 512 (shared) → actor / critic heads
    """

    def __init__(self, action_size, frame_stack=FRAME_STACK_SIZE):
        super().__init__()

        self.blocks = nn.Sequential(
            IMPALABlock(frame_stack, 32),    # → (32, 48, 64)
            IMPALABlock(32, 64),             # → (64, 24, 32)
            IMPALABlock(64, 64),             # → (64, 12, 16)
        )

        self.feature_size = 64 * 12 * 16    # 12 288

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Linear(512, action_size)
        self.critic = nn.Linear(512, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        """Orthogonal init for conv/fc; small init for actor head."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        x = self.blocks(x)
        x = F.relu(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_action_and_value(self, x, action=None):
        features = self.forward(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(features)
        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, x):
        features = self.forward(x)
        return self.critic(features).squeeze(-1)


class FrameStack:
    """FIFO stack of recent grayscale frames for temporal context."""

    def __init__(self, stack_size, frame_shape):
        self.stack_size = stack_size
        self.frame_shape = frame_shape  # (H, W)
        self.frames = deque(maxlen=stack_size)
        self.reset()

    def reset(self):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(np.zeros(self.frame_shape, dtype=np.float32))

    def push(self, frame):
        """Add a new frame.  `frame` shape: (1, H, W) from preprocess."""
        self.frames.append(frame[0])  # strip channel dim → (H, W)

    def get(self):
        """Return stacked frames as (FRAME_STACK_SIZE, H, W)."""
        return np.array(self.frames, dtype=np.float32)


class RolloutBuffer:
    """Stores one epoch of rollout data for PPO training."""

    def __init__(self):
        self.clear()

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """GAE(λ) computation."""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = (
                rewards[t]
                + gamma * values[t + 1] * next_non_terminal
                - values[t]
            )
            advantages[t] = last_gae = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae
            )

        returns = advantages + np.array(self.values)
        return returns, advantages

    def get_batches(self, batch_size, returns, advantages):
        """Yield shuffled mini-batches (includes old values for value clipping)."""
        n = len(self.states)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield (
                np.array([self.states[i] for i in idx]),
                np.array([self.actions[i] for i in idx]),
                np.array([self.log_probs[i] for i in idx]),
                np.array([self.values[i] for i in idx]),
                returns[idx],
                advantages[idx],
            )


class PPOAgent:
    """Proximal Policy Optimization agent with IMPALA CNN backbone."""

    def __init__(
        self,
        action_size,
        lr=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.02,
        value_coef=0.5,
        max_grad_norm=0.5,
        ppo_epochs=6,
        mini_batch_size=256,
        value_clip_range=0.2,
        load_model_path=None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.value_clip_range = value_clip_range

        # Network
        if load_model_path and os.path.exists(load_model_path):
            print(f"Loading model from: {load_model_path}")
            self.network = DeathmatchCNN(action_size).to(DEVICE)
            self.network.load_state_dict(
                torch.load(load_model_path, map_location=DEVICE)
            )
            self.network.eval()
        else:
            print("Initializing new DeathmatchCNN model")
            self.network = DeathmatchCNN(action_size).to(DEVICE)

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=lr, eps=1e-5
        )
        self.buffer = RolloutBuffer()


    def get_action(self, state, deterministic=False):
        state_t = torch.from_numpy(np.expand_dims(state, 0)).float().to(DEVICE)
        with torch.no_grad():
            if deterministic:
                features = self.network(state_t)
                logits = self.network.actor(features)
                return torch.argmax(logits, dim=-1).item()
            else:
                action, log_prob, _, value = self.network.get_action_and_value(
                    state_t
                )
                return action.item(), log_prob.item(), value.item()

    def store_transition(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob, value)

    def get_last_value(self, state):
        state_t = torch.from_numpy(np.expand_dims(state, 0)).float().to(DEVICE)
        with torch.no_grad():
            return self.network.get_value(state_t).item()


    def update_entropy_coef(self, progress):
        """Linear anneal: entropy_coef_start → entropy_coef_end."""
        self.entropy_coef = (
            entropy_coef_start
            + (entropy_coef_end - entropy_coef_start) * progress
        )

    def update_lr(self, progress):
        """Linear anneal: learning_rate → ~0."""
        new_lr = max(learning_rate * (1.0 - progress), 1e-6)
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr


    def train(self):
        self.network.train()

        last_state = (
            self.buffer.states[-1]
            if self.buffer.states
            else np.zeros((FRAME_STACK_SIZE, *resolution))
        )
        last_value = (
            self.get_last_value(last_state)
            if not self.buffer.dones[-1]
            else 0.0
        )

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(
                self.mini_batch_size, returns, advantages
            ):
                (
                    b_states,
                    b_actions,
                    b_old_log_probs,
                    b_old_values,
                    b_returns,
                    b_advantages,
                ) = batch

                b_states = torch.from_numpy(b_states).float().to(DEVICE)
                b_actions = torch.from_numpy(b_actions).long().to(DEVICE)
                b_old_log_probs = (
                    torch.from_numpy(b_old_log_probs).float().to(DEVICE)
                )
                b_old_values = (
                    torch.from_numpy(b_old_values).float().to(DEVICE)
                )
                b_returns = (
                    torch.from_numpy(b_returns).float().to(DEVICE)
                )
                b_advantages = (
                    torch.from_numpy(b_advantages).float().to(DEVICE)
                )

                _, new_log_probs, entropy, values = (
                    self.network.get_action_and_value(b_states, b_actions)
                )

                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1 - self.clip_epsilon,
                        1 + self.clip_epsilon,
                    )
                    * b_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                values_clipped = b_old_values + torch.clamp(
                    values - b_old_values,
                    -self.value_clip_range,
                    self.value_clip_range,
                )
                v_loss1 = F.mse_loss(values, b_returns)
                v_loss2 = F.mse_loss(values_clipped, b_returns)
                value_loss = torch.max(v_loss1, v_loss2)

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Stats
                with torch.no_grad():
                    clip_frac = (
                        ((ratio - 1.0).abs() > self.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_clip_frac += clip_frac
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "clip_fraction": total_clip_frac / max(n_updates, 1),
        }


    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(
            torch.load(path, map_location=DEVICE)
        )


def test(game, agent, actions, num_episodes=50):
    """Run evaluation episodes and report mean score."""
    print("\nTesting...")
    agent.network.eval()
    test_scores = []
    frame_stack = FrameStack(FRAME_STACK_SIZE, resolution)

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        frame_stack.reset()
        while not game.is_episode_finished():
            state = game.get_state()
            frame = preprocess(state.screen_buffer, resolution)
            frame_stack.push(frame)
            obs = frame_stack.get()
            action = agent.get_action(obs, deterministic=True)
            game.make_action(actions[action], frame_repeat)
        test_scores.append(game.get_total_reward())

    test_scores = np.array(test_scores)
    print(
        f"Results: mean: {test_scores.mean():.1f} +/- {test_scores.std():.1f}, "
        f"min: {test_scores.min():.1f}, max: {test_scores.max():.1f}"
    )
    return test_scores.mean()


def run(game, agent, actions, num_epochs, steps_per_epoch, frame_repeat):
    """Main PPO training loop with reward shaping and schedule annealing."""
    start_time = time()
    best_mean_reward = float("-inf")

    frame_stack = FrameStack(FRAME_STACK_SIZE, resolution)
    reward_shaper = RewardShaper()

    for epoch in range(num_epochs):
        progress = epoch / max(num_epochs - 1, 1)
        agent.update_entropy_coef(progress)
        agent.update_lr(progress)

        current_lr = agent.optimizer.param_groups[0]["lr"]
        print(f"\n{'=' * 60}")
        print(
            f"Epoch #{epoch + 1}/{num_epochs}  |  "
            f"Ent coef: {agent.entropy_coef:.4f}  |  "
            f"LR: {current_lr:.2e}"
        )
        print(f"{'=' * 60}")

        game.new_episode()
        frame_stack.reset()
        reward_shaper.reset()

        train_scores = []
        train_kills = []
        episode_reward = 0.0

        agent.network.train()

        for _ in trange(steps_per_epoch, desc="Collecting rollout", leave=False):
            state = game.get_state()
            frame = preprocess(state.screen_buffer, resolution)
            frame_stack.push(frame)
            obs = frame_stack.get()

            action, log_prob, value = agent.get_action(obs)
            game_reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            # Reward shaping
            if not done:
                game_vars = game.get_state().game_variables
                shaped_reward = reward_shaper.shape_reward(
                    game_reward, game_vars, done
                )
            else:
                shaped_reward = game_reward + DEATH_PENALTY

            episode_reward += shaped_reward
            agent.store_transition(
                obs, action, shaped_reward, done, log_prob, value
            )

            if done:
                train_scores.append(episode_reward)
                train_kills.append(reward_shaper.prev_kills)
                episode_reward = 0.0
                game.new_episode()
                frame_stack.reset()
                reward_shaper.reset()

        stats = agent.train()

        if train_scores:
            scores = np.array(train_scores)
            kills = np.array(train_kills)
            print(f"\nTraining episodes: {len(scores)}")
            print(
                f"  Rewards:  mean={scores.mean():.1f} +/- {scores.std():.1f},  "
                f"min={scores.min():.1f},  max={scores.max():.1f}"
            )
            print(
                f"  Kills:    mean={kills.mean():.1f},  max={kills.max()}"
            )

        print(f"\nPPO Stats:")
        print(f"  Policy Loss:  {stats['policy_loss']:.4f}")
        print(f"  Value Loss:   {stats['value_loss']:.4f}")
        print(f"  Entropy:      {stats['entropy']:.4f}")
        print(f"  Clip Frac:    {stats['clip_fraction']:.3f}")

        mean_test_reward = test(game, agent, actions, test_episodes_per_epoch)

        if save_model and mean_test_reward > best_mean_reward:
            best_mean_reward = mean_test_reward
            print(
                f"New best model! Score: {mean_test_reward:.1f} → "
                f"Saving to: {model_savefile}"
            )
            agent.save(model_savefile)

        print(f"Elapsed: {(time() - start_time) / 60.0:.1f} min")

    game.close()
    return agent, game


if __name__ == "__main__":
    game = create_game()
    actions = get_deathmatch_actions()
    print(f"Number of actions: {len(actions)}")

    os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

    agent = PPOAgent(
        action_size=len(actions),
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef_start,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        value_clip_range=value_clip_range,
        load_model_path=model_savefile if load_model else None,
    )

    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            steps_per_epoch=steps_per_epoch,
            frame_repeat=frame_repeat,
        )

        print("\n" + "=" * 60)
        print("Training finished. Time to watch!")
        print("=" * 60)

    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    frame_stack = FrameStack(FRAME_STACK_SIZE, resolution)
    total_score = 0
    for ep in range(episodes_to_watch):
        game.new_episode()
        frame_stack.reset()
        while not game.is_episode_finished():
            state = game.get_state()
            assert state is not None
            frame = preprocess(state.screen_buffer, resolution)
            frame_stack.push(frame)
            obs = frame_stack.get()
            action = agent.get_action(obs, deterministic=True)

            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        total_score += score
        print(f"Episode {ep + 1} Total Score: {score}")

    print(f"\n-----Average Score: {total_score / episodes_to_watch}-----")
    game.close()
