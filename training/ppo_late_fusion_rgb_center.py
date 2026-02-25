#!/usr/bin/env python3

import itertools as it
import os
from time import sleep, time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

import vizdoom as vzd
from utils import preprocess_rgb, preprocess_vars, SCENARIO_PATH


# PPO Hyperparameters
learning_rate = 3e-4
gamma = 0.99  # discount factor
gae_lambda = 0.95  # GAE lambda for advantage estimation
clip_epsilon = 0.2  # PPO clip parameter
entropy_coef = 0.01  # entropy bonus coefficient
value_coef = 0.5  # value loss coefficient
max_grad_norm = 0.5  # gradient clipping

# Training settings
train_epochs = 10
steps_per_epoch = 2048  # steps to collect before each update
ppo_epochs = 4  # number of PPO update epochs per batch
mini_batch_size = 64  # mini-batch size for PPO updates

# Testing
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 10

#model_savefile = "../models/ppo_cnn.pth"
model_savefile = "../models/ppo_late_fusion_rgb_center.pth"
save_model = True
#load_model = False
# finish running 
load_model = True
skip_learning = False

# Configuration file path
#config_file_path = os.path.join(SCENARIO_PATH, "defend_the_line.cfg")
config_file_path = os.path.join(SCENARIO_PATH, "defend_the_center.cfg")

# Device setup
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

# Game variables for late fusion (MLP input)
GAME_VARS = [
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.HEALTH,
]
NUM_VARS = len(GAME_VARS)

def create_simple_game():
    """Initialize and configure the ViZDoom game."""
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    #game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    for gv in GAME_VARS:
        game.add_available_game_variable(gv)
    game.init()
    print("Doom initialized.")
    return game


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic CNN architecture for PPO.

    Uses a shared CNN backbone with separate heads for:
    - Actor (policy): outputs action probabilities
    - Critic (value): outputs state value estimate
    """

    def __init__(self, action_size):
        super().__init__()

        # Shared convolutional backbone
        self.conv1 = nn.Sequential(
            #nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate the size of flattened features
        # Input: (1, 30, 45) -> after convs: approximately (64, 4, 6)
        self.feature_size = 64 * 4 * 6

        # Shared fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 256),
            nn.ReLU(),
        )
        self.vars_mlp = nn.Sequential(
            nn.LayerNorm(NUM_VARS),
            nn.Linear(NUM_VARS, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            )

        # Actor head (policy)
        #self.actor = nn.Linear(256, action_size)

        # Critic head (value function)
        #self.critic = nn.Linear(256, 1)

        fused_dim = 256 + 64
        self.actor = nn.Linear(fused_dim, action_size)
        self.critic = nn.Linear(fused_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Actor head with smaller initial weights for exploration
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    
    def forward(self, x, vars_):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        img_feat = self.fc(x)                 # (B,256)
        
        vars_feat = self.vars_mlp(vars_)      # (B,64)
        fused = torch.cat([img_feat, vars_feat], dim=1)
        return fused

    def get_action_and_value(self, x, vars_, action=None):
        features = self.forward(x, vars_)
        """
        Get action probabilities, sampled action, log prob, entropy, and value.

        Args:
            x: observation tensor
            action: optional action to compute log prob for (used in training)

        Returns:
            action: sampled or provided action
            log_prob: log probability of the action
            entropy: entropy of the action distribution
            value: estimated state value
        """

        # Actor: action probabilities
        logits = self.actor(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        # Critic: state value
        value = self.critic(features)

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)

    def get_value(self, x, vars_):
        """Get only the value estimate (used for GAE computation)."""
        features = self.forward(x, vars_)
        return self.critic(features).squeeze(-1)


class RolloutBuffer:
    """Buffer to store rollout data for PPO training."""

    def __init__(self):
        self.states = []
        self.vars = []       
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, state, vars_, action, reward, done, log_prob, value):
        self.states.append(state)
        self.vars.append(vars_)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear()
        self.vars.clear()     
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Compute returns and GAE advantages.

        Args:
            last_value: value estimate for the last state
            gamma: discount factor
            gae_lambda: GAE lambda parameter

        Returns:
            returns: discounted returns
            advantages: GAE advantages
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [last_value])

        # GAE computation
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
        """Generate mini-batches for PPO update."""
        n_samples = len(self.states)
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]

            yield (
                np.array([self.states[i] for i in batch_indices]),
                np.array([self.vars[i] for i in batch_indices]),
                np.array([self.actions[i] for i in batch_indices]),
                np.array([self.log_probs[i] for i in batch_indices]),
                returns[batch_indices],
                advantages[batch_indices],
            )


class PPOAgent:
    """PPO Agent for ViZDoom."""

    def __init__(
        self,
        action_size,
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
            print(f"Loading PPO model from: {load_model_path}")
            self.network = ActorCriticCNN(action_size).to(DEVICE)
            self.network.load_state_dict(torch.load(
                load_model_path, map_location=DEVICE))
            self.network.eval()
        else:
            print("Initializing new PPO model")
            self.network = ActorCriticCNN(action_size).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()
        
    def get_action(self, state_img, state_vars, deterministic=False):
        """Get action for the given state."""
        state_img = np.expand_dims(state_img, axis=0)     # (1,3,H,W)
        state_vars = np.expand_dims(state_vars, axis=0)   # (1,NUM_VARS)

        img_t = torch.from_numpy(state_img).float().to(DEVICE)
        vars_t = torch.from_numpy(state_vars).float().to(DEVICE)

        with torch.no_grad():
            if deterministic:
                features = self.network(img_t, vars_t)
                logits = self.network.actor(features)
                action = torch.argmax(logits, dim=-1).item()
                return action
            else:
                action, log_prob, _, value = self.network.get_action_and_value(img_t, vars_t)
                return action.item(), log_prob.item(), value.item()


    def store_transition(self, state_img, state_vars, action, reward, done, log_prob, value):
        """Store a transition in the rollout buffer."""
        self.buffer.add(state_img, state_vars, action, reward, done, log_prob, value)


    def get_last_value(self, state_img, state_vars):
        """Get value estimate for the last state (for GAE computation)."""
        state_img = np.expand_dims(state_img, axis=0)
        state_vars = np.expand_dims(state_vars, axis=0)

        img_t = torch.from_numpy(state_img).float().to(DEVICE)
        vars_t = torch.from_numpy(state_vars).float().to(DEVICE)

        with torch.no_grad():
            return self.network.get_value(img_t, vars_t).item()

    def train(self):
        """
        Perform PPO update using collected rollout data (image + vars).

        Returns:
            dict with training statistics
        """
        # If buffer is empty, nothing to train on
        if not self.buffer.states:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # ----- Get last value for GAE -----
        # last_state: (3, H, W), last_vars: (NUM_VARS,)
        last_state_img = self.buffer.states[-1]
        last_state_vars = self.buffer.vars[-1]

        # If the last transition ended the episode, bootstrap value = 0
        if self.buffer.dones and self.buffer.dones[-1]:
            last_value = 0.0
        else:
            last_value = self.get_last_value(last_state_img, last_state_vars)

        # ----- Compute returns and advantages -----
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ----- Training statistics -----
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        # ----- PPO update epochs -----
        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, returns, advantages):
                states, vars_, actions, old_log_probs, batch_returns, batch_advantages = batch

                # Convert to tensors
                states = torch.from_numpy(states).float().to(DEVICE)            # (B,3,H,W)
                vars_ = torch.from_numpy(vars_).float().to(DEVICE)              # (B,NUM_VARS)
                actions = torch.from_numpy(actions).long().to(DEVICE)           # (B,)
                old_log_probs = torch.from_numpy(old_log_probs).float().to(DEVICE)  # (B,)
                batch_returns = torch.from_numpy(batch_returns).float().to(DEVICE)  # (B,)
                batch_advantages = torch.from_numpy(batch_advantages).float().to(DEVICE)  # (B,)

                # Current policy predictions
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    states, vars_, actions
                )

                # PPO clipped objective
                ratio = torch.exp(new_log_probs - old_log_probs)  # (B,)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss).item()  # convert back to +entropy
                n_updates += 1

        # Clear buffer after update
        self.buffer.clear()

        # Avoid divide-by-zero just in case
        if n_updates == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        """Load model weights."""
        self.network.load_state_dict(torch.load(path, map_location=DEVICE))

def test(game, agent, actions, num_episodes=100):
    """Run test episodes and report results (RGB + vars)."""
    print("\nTesting...")
    test_scores = []

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            gs = game.get_state()
            state_img = preprocess_rgb(gs.screen_buffer, resolution)
            state_vars = preprocess_vars(gs.game_variables, NUM_VARS)

            # If preprocess_rgb returns HWC, convert to CHW for PyTorch Conv2d
            if state_img.ndim == 3 and state_img.shape[-1] == 3:
                state_img = np.transpose(state_img, (2, 0, 1))

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
    """
    Main training loop using PPO (RGB + vars).

    Collects rollouts for steps_per_epoch steps, then performs PPO update.
    """
    start_time = time()
    best_mean_reward = float("-inf")

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch #{epoch + 1}")
        print(f"{'='*50}")

        game.new_episode()
        train_scores = []
        episode_reward = 0

        # Collect rollout
        for step in trange(steps_per_epoch, desc="Collecting rollout", leave=False):
            gs = game.get_state()
            state_img = preprocess_rgb(gs.screen_buffer, resolution)
            state_vars = preprocess_vars(gs.game_variables, NUM_VARS)

            # If preprocess_rgb returns HWC, convert to CHW for PyTorch Conv2d
            if state_img.ndim == 3 and state_img.shape[-1] == 3:
                state_img = np.transpose(state_img, (2, 0, 1))

            action, log_prob, value = agent.get_action(state_img, state_vars)

            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            episode_reward += reward
            agent.store_transition(state_img, state_vars, action, reward, done, log_prob, value)

            if done:
                train_scores.append(episode_reward)
                episode_reward = 0
                game.new_episode()

        # PPO update
        stats = agent.train()

        # Report training results
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

        # Test the agent
        mean_test_reward = test(game, agent, actions, test_episodes_per_epoch)

        # Save best model
        if save_model and mean_test_reward > best_mean_reward:
            best_mean_reward = mean_test_reward
            print(f"New best model! Saving to: {model_savefile}")
            agent.save(model_savefile)

        print(f"Total elapsed time: {(time() - start_time) / 60.0:.2f} minutes")

    game.close()
    return agent, game


if __name__ == "__main__":
    print(config_file_path)
    print(f"Using device: {DEVICE}")

    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    print(f"Number of actions: {len(actions)}")

    # Initialize PPO agent
    agent = PPOAgent(
        action_size=len(actions),
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

    total_score = 0
    for episode_num in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            state_img = preprocess_rgb(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)
            
            if state_img.ndim == 3 and state_img.shape[-1] == 3:
                state_img = np.transpose(state_img, (2, 0, 1))

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
