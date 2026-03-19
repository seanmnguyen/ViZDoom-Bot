#!/usr/bin/env python3
"""
ppo_late_fusion_gray.py

PPO + Late-Fusion (CNN(image) + MLP(game vars)) for ViZDoom, using GRAYSCALE
screen buffer input and (optionally) frame stacking.

Key features:
- Actor-Critic PPO with GAE(λ) + clipped objective
- Late-fusion architecture:
    image -> CNN -> img_fc
    vars  -> MLP -> vars_fc
    concat -> fused_fc -> actor + critic heads
- Grayscale preprocessing (GRAY8) via utils.preprocess()
- Full checkpointing (epoch-level resume): model, optimizer, AMP scaler (if used),
  RNG states, training counters, and best test score.

Notes on "full" checkpointing:
- PPO is on-policy; we collect fresh rollouts each epoch. Saving the rollout buffer
  to resume mid-epoch would make checkpoints huge (states are big). This file resumes
  cleanly at the start of an epoch (same idea as many PPO codebases).

After training:
- weights:   ../models/<scenario>/ppo_late_fusion_gray.pth
- checkpoint:../checkpoints/<scenario>/ppo_late_fusion_gray.pt

To integrate with demo/eval:
- Add this agent to model_registry.py and include it in LATE_FUSION_PPO_MODELS / PPO_MODELS.
"""

from __future__ import annotations

import itertools as it
import os
import random
from collections import deque
from time import sleep, time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

import vizdoom as vzd
from utils import *  # shared constants + preprocess fns


# -----------------------------------------------------------------------------
# Scenario / save naming
# -----------------------------------------------------------------------------
SCENARIO_NAME = "deadly_corridor"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

MODEL_TYPE = os.path.splitext(os.path.basename(__file__))[0]
model_savefile = f"../models/{SCENARIO_NAME}/{MODEL_TYPE}.pth"
os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

checkpoint_savefile = f"../checkpoints/{SCENARIO_NAME}/{MODEL_TYPE}.pt"
os.makedirs(os.path.dirname(checkpoint_savefile), exist_ok=True)

save_model = True
save_checkpoint = True
load_checkpoint = True  # resume if checkpoint exists
load_model_weights = False  # fallback: load model_savefile weights (no optimizer state)
skip_learning = False

checkpoint_interval_epochs = 1


# -----------------------------------------------------------------------------
# PPO Hyperparameters
# -----------------------------------------------------------------------------
# (We keep these as PPO-specific; utils.py standardizes DQN hyperparams.)
LEARNING_RATE_PPO = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

PPO_EPOCHS = 4
MINI_BATCH_SIZE = 64

# Rollout collection
TRAIN_EPOCHS_PPO = 150
STEPS_PER_EPOCH = 4096  # number of env steps to collect per PPO update

# Testing
TEST_EPISODES = TEST_EPISODES_PER_EPOCH  # from utils.py (default 100)

# Frame stacking (temporal context)
FRAME_STACK_SIZE = 4

# Use standardized params from utils.py
FRAME_REPEAT_EFFECTIVE = FRAME_REPEAT  # default 12
RESOLUTION_EFFECTIVE = RESOLUTION      # default (96, 128)

# Game vars
NUM_VARS = get_num_game_variables(config_file_path)


# -----------------------------------------------------------------------------
# Device + AMP
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

USE_AMP = (DEVICE.type == "cuda")
SCALER = torch.amp.GradScaler("cuda", enabled=USE_AMP)


# -----------------------------------------------------------------------------
# Checkpoint helpers (epoch-level full resume)
# -----------------------------------------------------------------------------
def _get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda"] = None
    else:
        state["torch_cuda"] = None
    return state


def _set_rng_state(state: dict):
    try:
        if state is None:
            return
        if "python" in state and state["python"] is not None:
            random.setstate(state["python"])
        if "numpy" in state and state["numpy"] is not None:
            np.random.set_state(state["numpy"])
        if "torch" in state and state["torch"] is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        print("Warning: failed to restore RNG state:", e)


def save_full_checkpoint(
    path: str,
    agent: "PPOAgent",
    *,
    epoch: int,
    global_step: int,
    best_mean_reward: float,
):
    ckpt = {
        "meta": {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "best_mean_reward": float(best_mean_reward),
            "saved_at": float(time()),
        },
        "config": {
            "MODEL_TYPE": MODEL_TYPE,
            "SCENARIO_NAME": SCENARIO_NAME,
            "RESOLUTION": tuple(RESOLUTION_EFFECTIVE),
            "FRAME_REPEAT": int(FRAME_REPEAT_EFFECTIVE),
            "FRAME_STACK_SIZE": int(FRAME_STACK_SIZE),
            "NUM_VARS": int(NUM_VARS),
            "action_size": int(agent.action_size),
            "USE_AMP": bool(agent.use_amp),
        },
        "agent": agent.state_dict(),
        "rng": _get_rng_state(),
    }
    torch.save(ckpt, path)


def load_full_checkpoint(path: str, agent: "PPOAgent"):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt.get("config", {})

    # Guard against accidental mismatched resumes (easy when changing stack size, vars, etc.)
    def _assert_eq(key, cur):
        if key in cfg and cfg[key] != cur:
            raise ValueError(f"Checkpoint mismatch for {key}: ckpt={cfg[key]} current={cur}")

    _assert_eq("RESOLUTION", tuple(RESOLUTION_EFFECTIVE))
    _assert_eq("FRAME_REPEAT", int(FRAME_REPEAT_EFFECTIVE))
    _assert_eq("FRAME_STACK_SIZE", int(FRAME_STACK_SIZE))
    _assert_eq("NUM_VARS", int(NUM_VARS))
    _assert_eq("action_size", int(agent.action_size))

    agent.load_state_dict(ckpt["agent"])
    _set_rng_state(ckpt.get("rng"))

    meta = ckpt.get("meta", {})
    start_epoch = int(meta.get("epoch", 0))
    global_step = int(meta.get("global_step", 0))
    best_mean_reward = float(meta.get("best_mean_reward", float("-inf")))
    return start_epoch, global_step, best_mean_reward


# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def create_simple_game(visible: bool = False, async_player: bool = False):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if async_player else vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)  # grayscale
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Keep behavior consistent with your other files (some configs need explicit adds)
    for gv in game.get_available_game_variables():
        game.add_available_game_variable(gv)

    game.init()
    print("Doom initialized.")
    return game


# -----------------------------------------------------------------------------
# Frame stack (grayscale)
# -----------------------------------------------------------------------------
class FrameStack:
    """Maintains a stack of recent frames for temporal context (grayscale)."""

    def __init__(self, stack_size: int, frame_shape_hw: Tuple[int, int]):
        self.stack_size = int(stack_size)
        self.frame_shape_hw = frame_shape_hw  # (H, W)
        self.frames = deque(maxlen=self.stack_size)
        self.reset()

    def reset(self):
        self.frames.clear()
        z = np.zeros(self.frame_shape_hw, dtype=np.float32)
        for _ in range(self.stack_size):
            self.frames.append(z.copy())

    def push(self, frame_chw: np.ndarray):
        """
        frame_chw: (1,H,W) float32 from utils.preprocess()
        stores (H,W) internally
        """
        self.frames.append(frame_chw[0])

    def get(self) -> np.ndarray:
        """Return stacked frames as (K, H, W) float32."""
        return np.asarray(self.frames, dtype=np.float32)


# -----------------------------------------------------------------------------
# Network: Strong CNN backbone (adapted from your StrongCNN family)
# -----------------------------------------------------------------------------
NORM_KIND = "group"
GROUP_NORM_GROUPS = 8


def norm2d(channels: int) -> nn.Module:
    if NORM_KIND == "group":
        g = min(GROUP_NORM_GROUPS, channels)
        while g > 1 and channels % g != 0:
            g -= 1
        return nn.GroupNorm(g, channels)
    return nn.BatchNorm2d(channels)


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
        self.bn1 = norm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                norm2d(out_ch),
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
    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm2d(32),
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
            norm2d(128),
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
        return torch.flatten(x, 1)


class ActorCriticLateFusion(nn.Module):
    """
    Actor-Critic with Late Fusion.

    Inputs:
      img:  (B, K, H, W)  (grayscale stacked)
      vars: (B, V)
    """

    def __init__(self, action_size: int, num_vars: int, img_hw: Tuple[int, int], in_channels: int):
        super().__init__()
        self.action_size = int(action_size)
        self.num_vars = int(num_vars)

        self.cnn = StrongCNN(in_channels=in_channels)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_hw[0], img_hw[1])
            cnn_dim = int(self.cnn(dummy).shape[1])

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

        fused_dim = 128 + 64
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
        )

        self.actor = nn.Linear(256, action_size)
        self.critic = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Orthogonal init is standard for PPO.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        # Smaller actor init encourages early exploration.
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        # Critic head.
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward_features(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = self.fuse(torch.cat([img_feat, vars_feat], dim=1))
        return fused

    def get_action_and_value(self, img: torch.Tensor, vars_: torch.Tensor, action: Optional[torch.Tensor] = None):
        features = self.forward_features(img, vars_)
        logits = self.actor(features)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        value = self.critic(features).squeeze(-1)
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, img: torch.Tensor, vars_: torch.Tensor):
        features = self.forward_features(img, vars_)
        return self.critic(features).squeeze(-1)


# -----------------------------------------------------------------------------
# PPO Rollout Buffer (stores both img + vars)
# -----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.img_states = []
        self.var_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, img_state, var_state, action, reward, done, log_prob, value):
        self.img_states.append(img_state)
        self.var_states.append(var_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.img_states.clear()
        self.var_states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        values = np.asarray(self.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return returns, advantages

    def get_batches(self, batch_size: int, returns: np.ndarray, advantages: np.ndarray):
        n = len(self.actions)
        idxs = np.random.permutation(n)
        for start in range(0, n, batch_size):
            bi = idxs[start:start + batch_size]
            yield (
                np.asarray([self.img_states[i] for i in bi], dtype=np.float32),
                np.asarray([self.var_states[i] for i in bi], dtype=np.float32),
                np.asarray([self.actions[i] for i in bi], dtype=np.int64),
                np.asarray([self.log_probs[i] for i in bi], dtype=np.float32),
                returns[bi].astype(np.float32),
                advantages[bi].astype(np.float32),
            )


# -----------------------------------------------------------------------------
# PPO Agent
# -----------------------------------------------------------------------------
class PPOAgent:
    def __init__(
        self,
        action_size: int,
        *,
        lr: float = LEARNING_RATE_PPO,
        gamma: float = GAMMA,
        gae_lambda: float = GAE_LAMBDA,
        clip_epsilon: float = CLIP_EPSILON,
        entropy_coef: float = ENTROPY_COEF,
        value_coef: float = VALUE_COEF,
        max_grad_norm: float = MAX_GRAD_NORM,
        ppo_epochs: int = PPO_EPOCHS,
        mini_batch_size: int = MINI_BATCH_SIZE,
        load_model_path: Optional[str] = None,
    ):
        self.action_size = int(action_size)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)

        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)

        self.use_amp = USE_AMP

        self.network = ActorCriticLateFusion(
            action_size=self.action_size,
            num_vars=NUM_VARS,
            img_hw=RESOLUTION_EFFECTIVE,
            in_channels=FRAME_STACK_SIZE,  # grayscale: 1 channel per frame => K channels when stacked
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scaler = SCALER  # shared global scaler
        self.buffer = RolloutBuffer()

        if load_model_path:
            print(f"Loading PPO weights from: {load_model_path}")
            sd = torch.load(load_model_path, map_location=DEVICE)
            # accept both pure state_dict and {"state_dict": ...}
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            self.network.load_state_dict(sd)
            self.network.eval()

    @torch.no_grad()
    def get_action(self, state_img: np.ndarray, state_vars: np.ndarray, deterministic: bool = False):
        """
        state_img: (K,H,W) float32
        state_vars:(V,) float32
        """
        img_t = torch.from_numpy(state_img[None, ...]).float().to(DEVICE)
        vars_t = torch.from_numpy(state_vars[None, ...]).float().to(DEVICE)

        if deterministic:
            features = self.network.forward_features(img_t, vars_t)
            logits = self.network.actor(features)
            action = int(torch.argmax(logits, dim=-1).item())
            return action

        action_t, logp_t, _, value_t = self.network.get_action_and_value(img_t, vars_t)
        return int(action_t.item()), float(logp_t.item()), float(value_t.item())

    @torch.no_grad()
    def get_last_value(self, state_img: np.ndarray, state_vars: np.ndarray) -> float:
        img_t = torch.from_numpy(state_img[None, ...]).float().to(DEVICE)
        vars_t = torch.from_numpy(state_vars[None, ...]).float().to(DEVICE)
        v = self.network.get_value(img_t, vars_t)
        return float(v.item())

    def store_transition(self, img_state, var_state, action, reward, done, log_prob, value):
        self.buffer.add(img_state, var_state, action, reward, done, log_prob, value)

    def state_dict(self) -> dict:
        # Save everything needed to resume training
        sd = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.use_amp else None),
        }
        return sd

    def load_state_dict(self, sd: dict):
        self.network.load_state_dict(sd["network"])
        self.optimizer.load_state_dict(sd["optimizer"])

        # Move optimizer state tensors to current device (CPU-saved checkpoints)
        for st in self.optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(DEVICE)

        if self.use_amp and sd.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(sd["scaler"])
            except Exception as e:
                print("Warning: failed to load AMP scaler state:", e)

    def save_weights(self, path: str):
        torch.save(self.network.state_dict(), path)

    def train_update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected rollout data.
        Returns dict with training statistics.
        """
        if not self.buffer.actions:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # For GAE: need V(s_T). Use last transition's state; if episode ended there, last_value=0.
        last_img = self.buffer.img_states[-1]
        last_vars = self.buffer.var_states[-1]
        last_value = 0.0 if self.buffer.dones[-1] else self.get_last_value(last_img, last_vars)

        returns, advantages = self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)

        # Normalize advantages (helps stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        self.network.train()

        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp)

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size, returns, advantages):
                img_states, var_states, actions, old_logp, batch_returns, batch_adv = batch

                img_t = torch.from_numpy(img_states).float().to(DEVICE)
                vars_t = torch.from_numpy(var_states).float().to(DEVICE)
                actions_t = torch.from_numpy(actions).long().to(DEVICE)
                old_logp_t = torch.from_numpy(old_logp).float().to(DEVICE)
                returns_t = torch.from_numpy(batch_returns).float().to(DEVICE)
                adv_t = torch.from_numpy(batch_adv).float().to(DEVICE)

                self.optimizer.zero_grad(set_to_none=True)

                with autocast_ctx:
                    _, new_logp_t, entropy_t, values_t = self.network.get_action_and_value(img_t, vars_t, actions_t)

                    ratio = torch.exp(new_logp_t - old_logp_t)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values_t, returns_t)

                    entropy_bonus = entropy_t.mean()

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_bonus.item())
                n_updates += 1

        self.buffer.clear()
        self.network.eval()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def set_eval_mode(self):
        self.network.eval()

    def set_train_mode(self):
        self.network.train()


# -----------------------------------------------------------------------------
# Evaluation + training loops
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(game: vzd.DoomGame, agent: PPOAgent, actions, num_episodes: int) -> float:
    agent.set_eval_mode()
    scores = []
    frame_stack = FrameStack(FRAME_STACK_SIZE, RESOLUTION_EFFECTIVE)

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        frame_stack.reset()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_safe(gs.game_variables, NUM_VARS)

            a = agent.get_action(state_img, state_vars, deterministic=True)
            game.make_action(actions[a], FRAME_REPEAT_EFFECTIVE)

        scores.append(float(game.get_total_reward()))

    scores = np.asarray(scores, dtype=np.float32)
    print(
        "Results: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
            float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
        )
    )
    return float(scores.mean())


def train(game: vzd.DoomGame, agent: PPOAgent, actions, *, start_epoch: int = 0, start_global_step: int = 0, best_mean_reward: float = float("-inf")):
    start_time = time()
    global_step = int(start_global_step)

    frame_stack = FrameStack(FRAME_STACK_SIZE, RESOLUTION_EFFECTIVE)

    for epoch in range(start_epoch, TRAIN_EPOCHS_PPO):
        print(f"\n{'=' * 60}")
        print(f"Epoch #{epoch + 1} / {TRAIN_EPOCHS_PPO}")
        print(f"{'=' * 60}")

        # Start a fresh episode for rollout collection
        game.new_episode()
        frame_stack.reset()

        train_episode_rewards = []
        ep_reward = 0.0

        # Collect rollout
        for _ in trange(STEPS_PER_EPOCH, desc="Collecting rollout", leave=False):
            gs = game.get_state()
            if gs is None:
                # Treat as terminal
                train_episode_rewards.append(ep_reward)
                ep_reward = 0.0
                game.new_episode()
                frame_stack.reset()
                continue

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_safe(gs.game_variables, NUM_VARS)

            action, logp, value = agent.get_action(state_img, state_vars, deterministic=False)

            r = float(game.make_action(actions[action], FRAME_REPEAT_EFFECTIVE))
            done = bool(game.is_episode_finished())

            ep_reward += r
            agent.store_transition(state_img, state_vars, action, r, done, logp, value)

            global_step += 1

            if done:
                train_episode_rewards.append(ep_reward)
                ep_reward = 0.0
                game.new_episode()
                frame_stack.reset()

        # PPO update
        stats = agent.train_update()
        print(f"\nPPO update stats:")
        print(f"  policy_loss: {stats['policy_loss']:.4f}")
        print(f"  value_loss : {stats['value_loss']:.4f}")
        print(f"  entropy    : {stats['entropy']:.4f}")

        if train_episode_rewards:
            tr = np.asarray(train_episode_rewards, dtype=np.float32)
            print(
                "Train episodes: {} | mean {:.2f} +/- {:.2f} | min {:.2f} | max {:.2f}".format(
                    len(tr), float(tr.mean()), float(tr.std()), float(tr.min()), float(tr.max())
                )
            )
        else:
            print("Train episodes: none completed during rollout collection.")

        print("\nTesting...")
        mean_test_reward = evaluate(game, agent, actions, num_episodes=TEST_EPISODES)

        if save_model and mean_test_reward > best_mean_reward:
            best_mean_reward = mean_test_reward
            print(f"New best model! Saving weights to: {model_savefile}")
            agent.save_weights(model_savefile)

        if save_checkpoint and ((epoch + 1) % checkpoint_interval_epochs == 0):
            print(f"Saving checkpoint to: {checkpoint_savefile}")
            save_full_checkpoint(
                checkpoint_savefile,
                agent,
                epoch=epoch + 1,  # resume will start at this epoch index
                global_step=global_step,
                best_mean_reward=best_mean_reward,
            )

        elapsed_min = (time() - start_time) / 60.0
        print(f"Total elapsed time: {elapsed_min:.2f} minutes")

    game.close()
    return best_mean_reward


def watch_trained(agent: PPOAgent, actions, episodes: int = EPISODES_TO_WATCH):
    game = create_simple_game(visible=True, async_player=True)
    agent.set_eval_mode()

    frame_stack = FrameStack(FRAME_STACK_SIZE, RESOLUTION_EFFECTIVE)

    total = 0.0
    for ep in range(episodes):
        game.new_episode()
        frame_stack.reset()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)
            frame_stack.push(frame)
            state_img = frame_stack.get()
            state_vars = preprocess_vars_safe(gs.game_variables, NUM_VARS)

            a = agent.get_action(state_img, state_vars, deterministic=True)

            game.set_action(actions[a])
            for _ in range(FRAME_REPEAT_EFFECTIVE):
                game.advance_action()

        sleep(1.0)
        score = float(game.get_total_reward())
        total += score
        print(f"Episode {ep + 1} Total score: {score}")

    print(f"-----Average Score: {total / max(1, episodes):.2f}-----")
    game.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("----------MODEL CONFIGURATION----------")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("SCENARIO_NAME:", SCENARIO_NAME)
    print("DEVICE:", DEVICE)
    print("USE_AMP:", USE_AMP)
    print("RESOLUTION:", RESOLUTION_EFFECTIVE)
    print("FRAME_REPEAT:", FRAME_REPEAT_EFFECTIVE)
    print("FRAME_STACK_SIZE:", FRAME_STACK_SIZE)
    print("NUM_VARS:", NUM_VARS)
    print("LEARNING_RATE_PPO:", LEARNING_RATE_PPO)
    print("TRAIN_EPOCHS_PPO:", TRAIN_EPOCHS_PPO)
    print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH)
    print("PPO_EPOCHS:", PPO_EPOCHS)
    print("MINI_BATCH_SIZE:", MINI_BATCH_SIZE)
    print("TEST_EPISODES:", TEST_EPISODES)
    print("MODEL_SAVEFILE:", model_savefile)
    print("CHECKPOINT_SAVEFILE:", checkpoint_savefile)
    print("load_checkpoint:", load_checkpoint)

    # Init game and action space
    game = create_simple_game(visible=False, async_player=False)
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print("ACTIONS:", len(actions))

    agent = PPOAgent(action_size=len(actions))

    # Resume training (full) or load weights (inference only)
    start_epoch = 0
    start_global_step = 0
    best_mean_reward = float("-inf")

    if load_checkpoint and os.path.exists(checkpoint_savefile):
        print("Loading checkpoint from:", checkpoint_savefile)
        start_epoch, start_global_step, best_mean_reward = load_full_checkpoint(checkpoint_savefile, agent)
        print(f"Resuming from epoch={start_epoch}, global_step={start_global_step}, best_mean_reward={best_mean_reward:.2f}")
    elif load_model_weights and os.path.exists(model_savefile):
        print("Loading model weights from:", model_savefile)
        agent.network.load_state_dict(torch.load(model_savefile, map_location=DEVICE))
        agent.set_eval_mode()

    if not skip_learning:
        best_mean_reward = train(
            game,
            agent,
            actions,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            best_mean_reward=best_mean_reward,
        )
        print("\n" + "=" * 60)
        print("Training finished. It's time to watch!")
        print("=" * 60)
    else:
        game.close()

    watch_trained(agent, actions, episodes=EPISODES_TO_WATCH)
