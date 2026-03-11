#!/usr/bin/env python3

"""
PPO (Proximal Policy Optimization) for ViZDoom - defend_the_line scenario.
Greyscale input | CNN backbone + MLP Actor/Critic heads.
"""

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
from utils import preprocess, SCENARIO_PATH


# ── Hyperparameters ───────────────────────────────────────────────────────────

learning_rate   = 3e-4
gamma           = 0.99    # discount factor
gae_lambda      = 0.95    # GAE-λ for advantage estimation
clip_epsilon    = 0.2     # PPO clipping parameter
entropy_coef    = 0.05    # entropy bonus (exploration)
value_coef      = 0.5     # value-loss weight
max_grad_norm   = 0.5     # gradient clipping

# Training regime
train_epochs            = 20
steps_per_epoch         = 4096   # rollout length before each PPO update
ppo_epochs              = 4      # optimisation passes per rollout
mini_batch_size         = 64
test_episodes_per_epoch = 100

# Environment
frame_repeat      = 12
resolution        = (30, 45)   # (H, W) after preprocessing
episodes_to_watch = 10

# Persistence
model_savefile = "../models/ppo_late_fusion_gray.pth"
save_model     = True
load_model     = True
skip_learning  = True

config_file_path = os.path.join(SCENARIO_PATH, "defend_the_line.cfg")
print(config_file_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE}")


# ── Game setup ────────────────────────────────────────────────────────────────

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)         # 1-channel greyscale
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game


# ── Network ───────────────────────────────────────────────────────────────────

class ActorCriticNet(nn.Module):
    """
    Shared CNN backbone → shared FC layer → separate MLP Actor and Critic heads.

    Input:  (B, 1, 30, 45)  greyscale frame
    CNN:    Conv(1→32) → Conv(32→64) → Conv(64→64)  (stride-2, padding-1 each)
    Shared: Linear(1536 → 256) + ReLU
    Actor:  256 → 128 → 64 → n_actions  (policy logits)
    Critic: 256 → 128 → 64 → 1          (state value)
    """

    def __init__(self, action_size: int):
        super().__init__()

        # Shared CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),   # → (B,32,15,23)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (B,64, 8,12)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # → (B,64, 4, 6)
            nn.ReLU(),
        )
        cnn_out = 64 * 4 * 6  # 1536

        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
        )

        # Actor MLP head
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

        # Critic MLP head
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialisation — standard best practice for PPO."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Smaller gain on final actor layer promotes early exploration
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.shared_fc(x)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """
        Full forward pass used during rollout collection and PPO updates.
        Returns: action, log_prob, entropy, value
        """
        features = self._features(x)
        dist     = Categorical(logits=self.actor(features))
        value    = self.critic(features).squeeze(-1)

        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Critic-only pass — used for bootstrapping the final rollout value."""
        return self.critic(self._features(x)).squeeze(-1)

    def get_action_deterministic(self, x: torch.Tensor) -> int:
        """Greedy action for evaluation (no sampling)."""
        return self.actor(self._features(x)).argmax(dim=-1).item()


# ── Rollout buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores one epoch of on-policy experience."""

    def __init__(self):
        self.states:    list = []
        self.actions:   list = []
        self.rewards:   list = []
        self.dones:     list = []
        self.log_probs: list = []
        self.values:    list = []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.states.clear();    self.actions.clear();   self.rewards.clear()
        self.dones.clear();     self.log_probs.clear(); self.values.clear()

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """
        Generalised Advantage Estimation (GAE-λ).
        Returns returns and advantages as float32 numpy arrays.
        """
        rewards    = np.array(self.rewards, dtype=np.float32)
        dones      = np.array(self.dones,   dtype=np.float32)
        values     = np.array(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae        = 0.0

        for t in reversed(range(len(rewards))):
            delta         = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae           = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return returns, advantages

    def get_mini_batches(self, batch_size: int, returns: np.ndarray, advantages: np.ndarray):
        """Yield shuffled mini-batches for the PPO update loop."""
        n   = len(self.states)
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            yield (
                np.stack([self.states[i]    for i in b]),
                np.array([self.actions[i]   for i in b]),
                np.array([self.log_probs[i] for i in b], dtype=np.float32),
                returns[b],
                advantages[b],
            )


# ── PPO Agent ─────────────────────────────────────────────────────────────────

class PPOAgent:

    def __init__(
        self,
        action_size:     int,
        lr:              float = 3e-4,
        gamma:           float = 0.99,
        gae_lambda:      float = 0.95,
        clip_epsilon:    float = 0.2,
        entropy_coef:    float = 0.01,
        value_coef:      float = 0.5,
        max_grad_norm:   float = 0.5,
        ppo_epochs:      int   = 4,
        mini_batch_size: int   = 64,
        load_model_path: str   = None,
    ):
        self.gamma           = gamma
        self.gae_lambda      = gae_lambda
        self.clip_epsilon    = clip_epsilon
        self.entropy_coef    = entropy_coef
        self.value_coef      = value_coef
        self.max_grad_norm   = max_grad_norm
        self.ppo_epochs      = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.net = ActorCriticNet(action_size).to(DEVICE)

        if load_model_path:
            print(f"Loading model from: {load_model_path}")
            self.net.load_state_dict(torch.load(load_model_path, map_location=DEVICE))

        self.net.eval()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer    = RolloutBuffer()

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_action(self, state: np.ndarray, deterministic: bool = False):
        """
        Rollout mode  → returns (action_idx, log_prob, value).
        Eval mode     → returns action_idx only.
        """
        t = torch.from_numpy(np.expand_dims(state, 0)).float().to(DEVICE)
        if deterministic:
            return self.net.get_action_deterministic(t)
        action, log_prob, _, value = self.net.get_action_and_value(t)
        return action.item(), log_prob.item(), value.item()

    @torch.no_grad()
    def get_last_value(self, state: np.ndarray) -> float:
        t = torch.from_numpy(np.expand_dims(state, 0)).float().to(DEVICE)
        return self.net.get_value(t).item()

    def store(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob, value)

    # ── PPO update ────────────────────────────────────────────────────────

    def train(self) -> dict:
        """Run ppo_epochs of mini-batch updates on the current rollout buffer."""
        last_done  = self.buffer.dones[-1]
        last_value = 0.0 if last_done else self.get_last_value(self.buffer.states[-1])

        returns, advantages = self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

        # Normalise advantages across the full rollout
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_pl, total_vl, total_ent, n_updates = 0.0, 0.0, 0.0, 0

        self.net.train()
        for _ in range(self.ppo_epochs):
            for states, actions, old_lps, rets, advs in \
                    self.buffer.get_mini_batches(self.mini_batch_size, returns, advantages):

                states_t  = torch.from_numpy(states).float().to(DEVICE)
                actions_t = torch.from_numpy(actions).long().to(DEVICE)
                old_lps_t = torch.from_numpy(old_lps).float().to(DEVICE)
                rets_t    = torch.from_numpy(rets).float().to(DEVICE)
                advs_t    = torch.from_numpy(advs).float().to(DEVICE)

                _, new_lps, entropy, values = self.net.get_action_and_value(states_t, actions_t)

                # Clipped surrogate objective
                ratio  = torch.exp(new_lps - old_lps_t)
                surr1  = ratio * advs_t
                surr2  = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advs_t
                p_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_loss = nn.functional.mse_loss(values, rets_t)

                # Total loss (entropy subtracted to maximise it)
                loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pl  += p_loss.item()
                total_vl  += v_loss.item()
                total_ent += entropy.mean().item()
                n_updates += 1

        self.net.eval()
        self.buffer.clear()

        return {
            "policy_loss": total_pl  / n_updates,
            "value_loss":  total_vl  / n_updates,
            "entropy":     total_ent / n_updates,
        }

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=DEVICE))


# ── Test loop ─────────────────────────────────────────────────────────────────

def test(game, agent, actions, num_episodes: int = 100):
    print("\nTesting...")
    scores = []
    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            state  = preprocess(game.get_state().screen_buffer, resolution)
            action = agent.get_action(state, deterministic=True)
            game.make_action(actions[action], frame_repeat)
        scores.append(game.get_total_reward())

    scores = np.array(scores)
    print(
        f"Results: mean: {scores.mean():.1f} +/- {scores.std():.1f}, "
        f"min: {scores.min():.1f}, max: {scores.max():.1f}"
    )
    return scores.mean()


# ── Training loop ─────────────────────────────────────────────────────────────

def run(game, agent, actions, num_epochs, steps_per_epoch, frame_repeat):
    start_time       = time()
    best_mean_reward = float("-inf")

    for epoch in range(num_epochs):
        print(f"\n{'='*50}\nEpoch #{epoch + 1}\n{'='*50}")

        game.new_episode()
        train_scores   = []
        episode_reward = 0.0

        # Rollout collection
        for _ in trange(steps_per_epoch, desc="Collecting rollout", leave=False):
            state  = preprocess(game.get_state().screen_buffer, resolution)
            action, log_prob, value = agent.get_action(state)

            reward = game.make_action(actions[action], frame_repeat)
            done   = game.is_episode_finished()
            episode_reward += reward

            agent.store(state, action, reward, done, log_prob, value)

            if done:
                train_scores.append(episode_reward)
                episode_reward = 0.0
                game.new_episode()

        # PPO update
        stats = agent.train()

        # Logging
        if train_scores:
            s = np.array(train_scores)
            print(
                f"\nTrain ({len(s)} eps): "
                f"mean {s.mean():.1f} ± {s.std():.1f}  "
                f"min {s.min():.1f}  max {s.max():.1f}"
            )
        print(
            f"PPO stats — policy: {stats['policy_loss']:.4f}  "
            f"value: {stats['value_loss']:.4f}  "
            f"entropy: {stats['entropy']:.4f}"
        )

        mean_reward = test(game, agent, actions, test_episodes_per_epoch)

        # Save only when a new best is reached
        if save_model and mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print(f"New best ({mean_reward:.1f})! Saving → {model_savefile}")
            agent.save(model_savefile)

        print(f"Elapsed: {(time() - start_time) / 60:.2f} min")

    game.close()
    return agent, game


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game    = create_simple_game()
    n       = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print(f"Action space size: {len(actions)}")

    agent = PPOAgent(
        action_size     = len(actions),
        lr              = learning_rate,
        gamma           = gamma,
        gae_lambda      = gae_lambda,
        clip_epsilon    = clip_epsilon,
        entropy_coef    = entropy_coef,
        value_coef      = value_coef,
        max_grad_norm   = max_grad_norm,
        ppo_epochs      = ppo_epochs,
        mini_batch_size = mini_batch_size,
        load_model_path = model_savefile if load_model else None,
    )

    if not skip_learning:
        agent, game = run(
            game, agent, actions,
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
            state  = preprocess(gs.screen_buffer, resolution)
            action = agent.get_action(state, deterministic=True)
            game.set_action(actions[action])
            for _ in range(frame_repeat):
                game.advance_action()

        sleep(1.0)
        score        = game.get_total_reward()
        total_score += score
        print(f"Episode {ep + 1} Total Score: {score}")

    print(f"\n----- Average Score: {total_score / episodes_to_watch:.1f} -----")
    game.close()