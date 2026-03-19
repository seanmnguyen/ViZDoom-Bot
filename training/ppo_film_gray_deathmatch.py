#!/usr/bin/env python3
"""
ppo_film_gray.py

PPO + FiLM-conditioned vision backbone for ViZDoom using GRAYSCALE screen buffer.

Why FiLM?
- Late fusion (concat CNN features + vars MLP) works, but it forces the network to learn
  "how to use vars" *after* vision features are already computed.
- FiLM (Feature-wise Linear Modulation) uses vars to *modulate* intermediate vision feature maps:
      x = (1 + gamma(vars)) * x + beta(vars)
  which often helps when vars represent "context" (ammo, health, armor, frag count, etc.).

Key features:
- Actor-Critic PPO with GAE(λ) + clipped objective
- Grayscale preprocessing (GRAY8) via utils.preprocess()
- Frame stacking (default K=4)
- Full checkpointing (epoch-level resume): model, optimizer, AMP scaler (if used), RNG states,
  training counters, best test score, and (optionally) vars normalizer state.

Notes on "full" checkpointing for PPO:
- PPO is on-policy; we collect fresh rollouts each epoch. Saving the rollout buffer to resume
  mid-epoch would make checkpoints huge (storing stacked frames). This script resumes cleanly
  at epoch boundaries.

To integrate with demo/eval:
- Add this agent to model_registry.py and include it in PPO_MODELS and LATE_FUSION_PPO_MODELS-like
  sets (you may want a new FILM_PPO_MODELS set). If you keep the same PPOAgent interface
  (get_action(img, vars, deterministic)), demo/eval can treat it like late-fusion PPO.
"""

from __future__ import annotations

import itertools as it
import os
import random
from collections import deque
from time import sleep, time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

import vizdoom as vzd
from utils import *  # shared constants + preprocess fns (preprocess for images)


# -----------------------------------------------------------------------------
# Scenario / save naming
# -----------------------------------------------------------------------------
SCENARIO_NAME = "deathmatch"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

MODEL_TYPE = os.path.splitext(os.path.basename(__file__))[0]
model_savefile = f"../models/{SCENARIO_NAME}/{MODEL_TYPE}.pth"
os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

checkpoint_savefile = f"../checkpoints/{SCENARIO_NAME}/{MODEL_TYPE}.pt"
os.makedirs(os.path.dirname(checkpoint_savefile), exist_ok=True)

save_model = True
save_checkpoint = True
load_checkpoint = True # resume if checkpoint exists
load_model_weights = False   # fallback: load model_savefile weights (no optimizer state)
skip_learning = False

checkpoint_interval_epochs = 1


# -----------------------------------------------------------------------------
# PPO Hyperparameters
# -----------------------------------------------------------------------------
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
TRAIN_EPOCHS_PPO = 251
STEPS_PER_EPOCH = 10000

# Testing
TEST_EPISODES = TEST_EPISODES_PER_EPOCH  # from utils.py (default 100)

# Frame stacking
FRAME_STACK_SIZE = 4

# Use standardized params from utils.py
FRAME_REPEAT_EFFECTIVE = FRAME_REPEAT
RESOLUTION_EFFECTIVE = RESOLUTION

# Game vars (scenario-specific)
NUM_VARS = get_num_game_variables(config_file_path)


# -----------------------------------------------------------------------------
# Device + AMP
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

USE_AMP = torch.cuda.is_available()
SCALER = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)


# -----------------------------------------------------------------------------
# Vars normalization (generalizes preprocess_vars_safe for deathmatch-like scenarios)
# -----------------------------------------------------------------------------
class RunningMeanStd:
    """Running mean/var for normalizing game vars vectors."""
    def __init__(self, shape, epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = float(epsilon)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count: int):
        batch_count = float(batch_count)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / tot_count)

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * batch_count / tot_count)
        new_var = M2 / tot_count

        self.mean = new_mean.astype(np.float32)
        self.var = np.maximum(new_var, 1e-8).astype(np.float32)
        self.count = tot_count

    def normalize(self, x: np.ndarray, clip: float = 5.0, eps: float = 1e-8) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = (x - self.mean) / np.sqrt(self.var + eps)
        if clip is not None:
            y = np.clip(y, -clip, clip)
        return y.astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": float(self.count)}

    def load_state_dict(self, sd: dict):
        self.mean = np.asarray(sd.get("mean", self.mean), dtype=np.float32)
        self.var = np.asarray(sd.get("var", self.var), dtype=np.float32)
        self.count = float(sd.get("count", self.count))


def preprocess_vars_safe_general(
    raw_vars: np.ndarray,
    expected: int,
    *,
    normalizer: RunningMeanStd | None = None,
    update: bool = False,
    clip: float = 5.0,
) -> np.ndarray:
    """
    Compatibility layer:
    - If expected <= 2, preserves legacy defend_the_center mapping (health/ammo scaling).
    - Else: pad/truncate and (optionally) running-mean-std normalize.
    """
    if expected <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Keep old behavior for DC (and any 2-var scenario in your codebase)
    if expected <= 2:
        return preprocess_vars_safe(raw_vars, expected)

    v = np.asarray(raw_vars, dtype=np.float32).reshape(-1)
    out = np.zeros((expected,), dtype=np.float32)
    n = min(expected, v.size)
    out[:n] = v[:n]

    if normalizer is not None:
        if update:
            normalizer.update(out)
        return normalizer.normalize(out, clip=clip)

    # Stateless fallback (bounded, robust)
    denom = np.log1p(1000.0).astype(np.float32)
    y = np.sign(out) * (np.log1p(np.abs(out)) / denom)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


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
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Keep behavior consistent with other scripts (some configs need explicit adds)
    for gv in game.get_available_game_variables():
        game.add_available_game_variable(gv)

    game.init()
    print("Doom initialized.")
    return game


class FrameStack:
    """Maintains a stack of recent frames for temporal context."""
    def __init__(self, stack_size: int, frame_shape: Tuple[int, int]):
        self.stack_size = int(stack_size)
        self.frame_shape = tuple(frame_shape)  # (H, W)
        self.frames = deque(maxlen=self.stack_size)
        self.reset()

    def reset(self):
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(np.zeros(self.frame_shape, dtype=np.float32))

    def push(self, frame_chw: np.ndarray):
        # frame_chw from utils.preprocess is (1, H, W)
        self.frames.append(frame_chw[0])

    def get(self) -> np.ndarray:
        # (K, H, W)
        return np.asarray(self.frames, dtype=np.float32)


# -----------------------------------------------------------------------------
# FiLM Actor-Critic Network
# -----------------------------------------------------------------------------
def norm2d(c: int) -> nn.Module:
    # BatchNorm2d tends to be unstable for RL with small batch sizes; GroupNorm is safer.
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n1 = norm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.n2 = norm2d(out_c)

        self.skip = None
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, bias=False),
                norm2d(out_c),
            )

    def forward(self, x):
        h = F.relu(self.n1(self.conv1(x)), inplace=True)
        h = self.n2(self.conv2(h))
        s = x if self.skip is None else self.skip(x)
        return F.relu(h + s, inplace=True)


class FiLM(nn.Module):
    """
    Produces (gamma, beta) for a given channel count C from a vars embedding.
    Initialize to identity (gamma=0, beta=0) so early training matches unconditioned CNN.
    """
    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.channels = int(channels)
        self.fc = nn.Linear(embed_dim, 2 * self.channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        gb = self.fc(emb)  # (B, 2C)
        gamma, beta = torch.chunk(gb, 2, dim=1)
        gamma = gamma.view(-1, self.channels, 1, 1)
        beta = beta.view(-1, self.channels, 1, 1)
        return x * (1.0 + gamma) + beta


class FiLMCNN(nn.Module):
    """
    CNN backbone with FiLM modulation at multiple feature-map stages.
    Input is stacked grayscale frames: (B, K, H, W).
    """
    def __init__(self, in_channels: int, vars_dim: int, embed_dim: int = 128):
        super().__init__()
        self.in_channels = int(in_channels)
        self.vars_dim = int(vars_dim)
        self.embed_dim = int(embed_dim)

        # Vars embed used for all FiLM blocks
        self.vars_embed = nn.Sequential(
            nn.LayerNorm(self.vars_dim),
            nn.Linear(self.vars_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embed_dim),
            nn.ReLU(inplace=True),
        )

        # Vision trunk (strong-ish but not huge)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            norm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = ResBlock(32, 32, stride=1)
        self.block2 = nn.Sequential(
            ResBlock(32, 64, stride=2),
            ResBlock(64, 64, stride=1),
        )
        self.block3 = nn.Sequential(
            ResBlock(64, 96, stride=2),
            ResBlock(96, 96, stride=1),
        )
        self.context = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # FiLM modulations after major stages
        self.film1 = FiLM(self.embed_dim, 32)
        self.film2 = FiLM(self.embed_dim, 64)
        self.film3 = FiLM(self.embed_dim, 96)
        self.film4 = FiLM(self.embed_dim, 128)

    def forward(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        emb = self.vars_embed(vars_)
        x = self.stem(img)
        x = self.block1(x)
        x = self.film1(x, emb)

        x = self.block2(x)
        x = self.film2(x, emb)

        x = self.block3(x)
        x = self.film3(x, emb)

        x = self.context(x)
        x = self.film4(x, emb)

        x = self.pool(x)
        return torch.flatten(x, 1)


class ActorCriticFiLM(nn.Module):
    """
    Actor-Critic head on top of FiLM-conditioned CNN.

    Inputs:
      img:  (B, K, H, W) stacked grayscale
      vars: (B, V) normalized vars vector
    """
    def __init__(self, action_size: int, num_vars: int, img_hw: Tuple[int, int], in_channels: int):
        super().__init__()
        self.action_size = int(action_size)
        self.num_vars = int(num_vars)

        self.cnn = FiLMCNN(in_channels=in_channels, vars_dim=self.num_vars, embed_dim=128)

        # Determine CNN output dim dynamically
        with torch.no_grad():
            dummy_img = torch.zeros(1, in_channels, img_hw[0], img_hw[1])
            dummy_vars = torch.zeros(1, self.num_vars)
            cnn_dim = int(self.cnn(dummy_img, dummy_vars).shape[1])

        self.fc = nn.Sequential(
            nn.Linear(cnn_dim, 256),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Linear(256, self.action_size)
        self.critic = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # PPO orthogonal init
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # FiLM layers already zero-initialized intentionally
                if isinstance(m, nn.Linear) and m is self.actor:
                    continue
                if isinstance(m, nn.Linear) and m is self.critic:
                    continue
                if isinstance(m, nn.Linear) and hasattr(m, "weight"):
                    # Don't override FiLM init (weights are zero)
                    pass

        # Orthogonal init for non-FiLM convs/linears (best effort)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            elif isinstance(m, nn.Linear):
                if m in [self.actor, self.critic]:
                    continue
                # Skip FiLM internal fc, which should remain zeros
                if m in [self.cnn.film1.fc, self.cnn.film2.fc, self.cnn.film3.fc, self.cnn.film4.fc]:
                    continue
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward_features(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(img, vars_))

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
# PPO Rollout Buffer
# -----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.img_states = []
        self.var_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.img_states.clear()
        self.var_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, img_state, var_state, action, log_prob, reward, done, value):
        self.img_states.append(img_state)
        self.var_states.append(var_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.actions)

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

        self.network = ActorCriticFiLM(
            action_size=self.action_size,
            num_vars=NUM_VARS,
            img_hw=RESOLUTION_EFFECTIVE,
            in_channels=FRAME_STACK_SIZE,
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scaler = SCALER
        self.buffer = RolloutBuffer()

        # Running normalization for vars (mainly useful for deathmatch / long var vectors)
        self.vars_rms = RunningMeanStd((NUM_VARS,)) if NUM_VARS > 2 else None

        if load_model_path:
            print(f"Loading PPO weights from: {load_model_path}")
            self.network.load_state_dict(torch.load(load_model_path, map_location=DEVICE))

        self.set_train_mode()

    def set_train_mode(self):
        self.network.train()

    def set_eval_mode(self):
        self.network.eval()

    @torch.no_grad()
    def get_action(self, img_state: np.ndarray, var_state: np.ndarray, deterministic: bool = False):
        img_t = torch.from_numpy(img_state).unsqueeze(0).to(DEVICE)   # (1,K,H,W)
        vars_t = torch.from_numpy(var_state).unsqueeze(0).to(DEVICE)  # (1,V)

        action, logp, _, _ = self.network.get_action_and_value(img_t, vars_t)
        if deterministic:
            # Greedy action from logits
            features = self.network.forward_features(img_t, vars_t)
            logits = self.network.actor(features)
            action = torch.argmax(logits, dim=1)
            logp = Categorical(logits=logits).log_prob(action)
            return action

        return int(action.item()), float(logp.item())

    def compute_gae(self, next_value: float):
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_nonterminal = 1.0 - (dones[t])
            next_values = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def update(self, next_value: float):
        returns, advantages = self.compute_gae(next_value)

        # Advantage normalization is standard for PPO
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for img_b, var_b, act_b, oldlog_b, ret_b, adv_b in self.buffer.get_batches(
                self.mini_batch_size, returns, advantages
            ):
                img_t = torch.from_numpy(img_b).to(DEVICE)
                vars_t = torch.from_numpy(var_b).to(DEVICE)
                act_t = torch.from_numpy(act_b).to(DEVICE)
                oldlog_t = torch.from_numpy(oldlog_b).to(DEVICE)
                ret_t = torch.from_numpy(ret_b).to(DEVICE)
                adv_t = torch.from_numpy(adv_b).to(DEVICE)

                with torch.amp.autocast(str(DEVICE), enabled=self.use_amp):
                    _, logp, entropy, value = self.network.get_action_and_value(img_t, vars_t, act_t)

                    ratio = torch.exp(logp - oldlog_t)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(value, ret_t)
                    entropy_loss = entropy.mean()

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
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

                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy_loss.item())
                n_updates += 1

        self.buffer.clear()
        return {
            "policy_loss": total_policy / max(1, n_updates),
            "value_loss": total_value / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def state_dict(self) -> dict:
        sd = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.use_amp else None),
            "vars_rms": (self.vars_rms.state_dict() if self.vars_rms is not None else None),
        }
        return sd

    def load_state_dict(self, sd: dict):
        self.network.load_state_dict(sd["network"])
        self.optimizer.load_state_dict(sd["optimizer"])

        # Move optimizer state tensors to current device
        for st in self.optimizer.state.values():
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(DEVICE)

        if self.use_amp and sd.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(sd["scaler"])
            except Exception as e:
                print("Warning: failed to restore GradScaler:", e)

        if self.vars_rms is not None and sd.get("vars_rms") is not None:
            try:
                self.vars_rms.load_state_dict(sd["vars_rms"])
            except Exception as e:
                print("Warning: failed to restore vars_rms:", e)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Evaluation + training loops (progress bars match ppo_late_fusion_gray.py)
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

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)  # (1,H,W)
            frame_stack.push(frame)
            state_img = frame_stack.get()  # (K,H,W)

            state_vars = preprocess_vars_safe_general(
                gs.game_variables,
                NUM_VARS,
                normalizer=agent.vars_rms,
                update=False,
                clip=5.0,
            )

            a_idx = agent.get_action(state_img, state_vars, deterministic=True)
            game.make_action(actions[a_idx], FRAME_REPEAT_EFFECTIVE)

        scores.append(float(game.get_total_reward()))

    scores = np.asarray(scores, dtype=np.float32)
    print(
        "Results: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
            float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
        )
    )
    return float(scores.mean())


def train(
    game: vzd.DoomGame,
    agent: PPOAgent,
    actions,
    *,
    start_epoch: int = 0,
    start_global_step: int = 0,
    best_mean_reward: float = float("-inf"),
):
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

        # Collect rollout (progress bar shows steps remaining)
        for _ in trange(STEPS_PER_EPOCH, desc="Collecting rollout", leave=False):
            gs = game.get_state()
            if gs is None:
                # Treat as terminal and restart
                train_episode_rewards.append(ep_reward)
                ep_reward = 0.0
                game.new_episode()
                frame_stack.reset()
                continue

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)  # (1,H,W)
            frame_stack.push(frame)
            state_img = frame_stack.get()  # (K,H,W)

            state_vars = preprocess_vars_safe_general(
                gs.game_variables,
                NUM_VARS,
                normalizer=agent.vars_rms,
                update=True,  # update running stats during training
                clip=5.0,
            )

            img_t = torch.from_numpy(state_img).unsqueeze(0).to(DEVICE)
            vars_t = torch.from_numpy(state_vars).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                action, logp, _, value = agent.network.get_action_and_value(img_t, vars_t)

            a_idx = int(action.item())
            r = float(game.make_action(actions[a_idx], FRAME_REPEAT_EFFECTIVE))
            done = bool(game.is_episode_finished())

            ep_reward += r

            agent.buffer.add(
                img_state=state_img,
                var_state=state_vars,
                action=a_idx,
                log_prob=float(logp.item()),
                reward=r,
                done=float(done),
                value=float(value.item()),
            )
            global_step += 1

            if done:
                train_episode_rewards.append(ep_reward)
                ep_reward = 0.0
                game.new_episode()
                frame_stack.reset()

        # Bootstrap value for last state
        with torch.no_grad():
            if game.is_episode_finished():
                next_value = 0.0
            else:
                gs = game.get_state()
                if gs is None:
                    next_value = 0.0
                else:
                    frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)
                    frame_stack.push(frame)
                    state_img = frame_stack.get()

                    state_vars = preprocess_vars_safe_general(
                        gs.game_variables,
                        NUM_VARS,
                        normalizer=agent.vars_rms,
                        update=False,
                        clip=5.0,
                    )

                    img_t = torch.from_numpy(state_img).unsqueeze(0).to(DEVICE)
                    vars_t = torch.from_numpy(state_vars).unsqueeze(0).to(DEVICE)
                    next_value = float(agent.network.get_value(img_t, vars_t).item())

        # PPO update (prints like late-fusion)
        stats = agent.update(next_value)
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
            torch.save(agent.network.state_dict(), model_savefile)

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
    n_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
    print("ACTIONS:", len(actions))

    agent = PPOAgent(action_size=len(actions))

    # Resume training (full) or load weights (inference only)
    start_epoch = 0
    start_global_step = 0
    best_mean_reward = float("-inf")

    if load_checkpoint and os.path.exists(checkpoint_savefile):
        print("Loading checkpoint from:", checkpoint_savefile)
        start_epoch, start_global_step, best_mean_reward = load_full_checkpoint(checkpoint_savefile, agent)
        print(
            f"Resuming from epoch={start_epoch}, global_step={start_global_step}, "
            f"best_mean_reward={best_mean_reward:.2f}"
        )
    elif load_model_weights and os.path.exists(model_savefile):
        print("Loading model weights from:", model_savefile)
        agent.network.load_state_dict(torch.load(model_savefile, map_location=DEVICE))
        agent.set_eval_mode()

    if skip_learning:
        print("\nTesting...")
        mean_r = evaluate(game, agent, actions, num_episodes=TEST_EPISODES)
        print("Mean reward:", mean_r)
    else:
        train(
            game,
            agent,
            actions,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            best_mean_reward=best_mean_reward,
        )
