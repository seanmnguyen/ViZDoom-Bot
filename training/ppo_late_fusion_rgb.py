#!/usr/bin/env python3

"""
Episode 1 Total Score: 9.0
Episode 2 Total Score: 10.0
Episode 3 Total Score: 6.0
Episode 4 Total Score: 7.0
Episode 5 Total Score: 8.0
Episode 6 Total Score: 11.0
Episode 7 Total Score: 8.0
Episode 8 Total Score: 12.0
Episode 9 Total Score: 4.0
Episode 10 Total Score: 8.0

-----Average Score: 8.3-----

with:
    frame_repeat = 12
    entropy_coef = 0.01
    train_epochs = 30
    steps_per_epoch = 4096

    // not much difference with current settings either
"""

import itertools as it
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
from utils import preprocess_rgb, preprocess_vars, SCENARIO_PATH, RESOLUTION, get_num_game_variables


# PPO Hyperparameters
learning_rate = 3e-4
gamma = 0.99  # discount factor
gae_lambda = 0.95  # GAE lambda for advantage estimation
clip_epsilon = 0.2  # PPO clip parameter
entropy_coef = 0.005  # entropy bonus coefficient (lower = less random spraying) (originally 0.01)
value_coef = 0.5  # value loss coefficient
max_grad_norm = 0.5  # gradient clipping

# Frame stacking
FRAME_STACK_SIZE = 4  # number of consecutive frames stacked as input channels
RGB_CHANNELS = 3
INPUT_CHANNELS = FRAME_STACK_SIZE * RGB_CHANNELS  # 4 * 3 = 12

# Training settings
train_epochs = 50 # (increased from 30)
steps_per_epoch = 8192  # steps to collect before each update (increased from 4096)
ppo_epochs = 4  # number of PPO update epochs per batch
mini_batch_size = 64  # mini-batch size for PPO updates

# Testing
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 4  # lower = finer aiming control (originally 12)
resolution = RESOLUTION  # (96, 128)
episodes_to_watch = 10

# Scenario
SCENARIO_NAME = "defend_the_center"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

model_savefile = f"../models/{SCENARIO_NAME}/ppo_late_fusion_rgb.pth"
save_model = True
load_model = False
skip_learning = False

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

NUM_VARS = get_num_game_variables(config_file_path)

print(config_file_path)
print(f"Using device: {DEVICE}")
print(f"Number of game variables: {NUM_VARS}")


def create_simple_game():
    """Initialize and configure the ViZDoom game."""
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

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
    """
    Stronger CNN for distant target detection:
    - less aggressive early downsampling
    - deeper residual stages
    - optional SE attention
    """
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

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.context(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # (B, 2048)
        return x


class ActorCriticLateFusion(nn.Module):
    """
    Late Fusion Actor-Critic for PPO:
    StrongCNN(img) + MLP(vars) -> concat -> shared FC -> actor + critic heads
    """
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

        # Infer CNN output dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            cnn_dim = self.cnn(dummy).shape[1]

        self.img_fc = nn.Sequential(
            nn.Linear(cnn_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.vars_mlp = nn.Sequential(
            nn.LayerNorm(num_vars),
            nn.Linear(num_vars, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = 128 + 64  # 192

        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
        )

        # Actor head (policy)
        self.actor = nn.Linear(256, action_size)

        # Critic head (value function)
        self.critic = nn.Linear(256, 1)

        # Initialize actor/critic heads
        self._initialize_heads()

    def _initialize_heads(self):
        """Initialize actor/critic head weights."""
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, img, vars_):
        """Forward pass through shared backbone."""
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = torch.cat([img_feat, vars_feat], dim=1)
        features = self.shared_fc(fused)
        return features

    def get_action_and_value(self, img, vars_, action=None):
        """
        Get action, log prob, entropy, and value.

        Args:
            img: image observation tensor
            vars_: game variables tensor
            action: optional action to compute log prob for (used in training)
        """
        features = self.forward(img, vars_)

        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, img, vars_):
        """Get only the value estimate (used for GAE computation)."""
        features = self.forward(img, vars_)
        return self.critic(features).squeeze(-1)

class FrameStackRGB:
    """Maintains a stack of recent RGB frames for temporal context."""

    def __init__(self, stack_size, frame_shape, channels=3):
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
        """Add a new frame. `frame` shape: (C, H, W) from preprocess_rgb."""
        self.frames.append(frame)

    def get(self):
        """Return stacked frames as (FRAME_STACK_SIZE * C, H, W)."""
        return np.concatenate(list(self.frames), axis=0)

class RolloutBuffer:
    """Buffer to store rollout data for PPO training."""

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
    """PPO Agent with Late Fusion architecture for ViZDoom."""

    def __init__(
        self,
        action_size,
        num_vars=NUM_VARS,
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
        """
        Get action for the given state.

        Args:
            state_img: preprocessed image observation (frame-stacked)
            state_vars: preprocessed game variables
            deterministic: if True, return most probable action
        """
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

    def train(self):
        """Perform PPO update using collected rollout data."""
        # Get last value for GAE
        last_img = self.buffer.states_img[-1] if self.buffer.states_img else np.zeros(
            (INPUT_CHANNELS, *resolution))
        last_vars = self.buffer.states_vars[-1] if self.buffer.states_vars else np.zeros(
            (NUM_VARS,))
        last_value = self.get_last_value(
            last_img, last_vars) if not self.buffer.dones[-1] else 0.0

        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        advantages = (advantages - advantages.mean()) / \
            (advantages.std() + 1e-8)

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

                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * batch_advantages_t
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, batch_returns_t)
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
    """Run test episodes and report results."""
    print("\nTesting...")
    test_scores = []
    frame_stack = FrameStackRGB(FRAME_STACK_SIZE, resolution, RGB_CHANNELS)

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        frame_stack.reset()
        while not game.is_episode_finished():
            game_state = game.get_state()
            frame = preprocess_rgb(game_state.screen_buffer, resolution)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)
            action = agent.get_action(state_img, state_vars, deterministic=True)
            game.make_action(actions[action], frame_repeat)
        test_scores.append(game.get_total_reward())

    test_scores = np.array(test_scores)
    print(
        f"Results: mean: {test_scores.mean():.1f} +/- {test_scores.std():.1f}, "
        f"min: {test_scores.min():.1f}, max: {test_scores.max():.1f}"
    )
    return test_scores.mean()


def run(game, agent, actions, num_epochs, steps_per_epoch, frame_repeat):
    """Main training loop using PPO with late fusion."""
    start_time = time()
    best_mean_reward = float("-inf")

    frame_stack = FrameStackRGB(FRAME_STACK_SIZE, resolution, RGB_CHANNELS)

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch #{epoch + 1}")
        print(f"{'='*50}")

        game.new_episode()
        frame_stack.reset()
        train_scores = []
        episode_reward = 0

        for step in trange(steps_per_epoch, desc="Collecting rollout", leave=False):
            game_state = game.get_state()
            frame = preprocess_rgb(game_state.screen_buffer, resolution)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)

            action, log_prob, value = agent.get_action(state_img, state_vars)

            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            episode_reward += reward
            agent.store_transition(state_img, state_vars, action, reward,
                                   done, log_prob, value)

            if done:
                train_scores.append(episode_reward)
                episode_reward = 0
                game.new_episode()
                frame_stack.reset()

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
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    print(f"Number of actions: {len(actions)}")

    # Initialize PPO agent
    agent = PPOAgent(
        action_size=len(actions),
        num_vars=NUM_VARS,
        lr=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_coef,
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
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            frame = preprocess_rgb(game_state.screen_buffer, resolution)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)
            action = agent.get_action(state_img, state_vars, deterministic=True)

            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score = game.get_total_reward()
        total_score += score
        print(f"Episode {episode_num + 1} Total Score: {score}")

    print(f"\n-----Average Score: {total_score / episodes_to_watch}-----")
    game.close()
