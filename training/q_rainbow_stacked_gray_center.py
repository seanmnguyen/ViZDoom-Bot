#!/usr/bin/env python3
"""
Rainbow (no recurrence) + Late-Fusion (Image + game vars) for ViZDoom.

This version focuses on **performance + memory** by fixing the main replay issue in
q_rainbow_stacked_rgb.py:

    Old: store (stacked_state, stacked_next_state) per transition  -> huge duplication
    New: store ONE raw frame per step, and **reconstruct K-frame stacks on sampling**

Key features kept:
- Double DQN
- Dueling network
- Distributional RL (C51)
- Noisy Nets exploration
- Prioritized Experience Replay (SumTree)
- N-step returns (computed on-the-fly at sampling)

One-switch RGB <-> Grayscale:
    Set USE_GRAYSCALE = True/False

Notes on memory:
- RGB, 96x128, uint8 frames: ~36KB/frame. 10k frames ~360MB (+ small arrays).
- Old approach stored stacked frames AND next stacked frames -> >1GB+ and swap.
"""

import itertools as it
import os
import random
import argparse
from collections import deque
from time import sleep, time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from utils import *  # shared constants + preprocess functions

# -----------------------------------------------------------------------------
# Scenario / save naming
# -----------------------------------------------------------------------------
SCENARIO_NAME = "defend_the_center"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

MODEL_TYPE = os.path.splitext(os.path.basename(__file__))[0]
model_savefile = f"../models/{SCENARIO_NAME}/{MODEL_TYPE}.pth"
os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

# -----------------------------------------------------------------------------
# Checkpointing (full training resume)
# -----------------------------------------------------------------------------
# We keep model_savefile (.pth) for lightweight inference/eval.
# Checkpoints (.pt) include: networks, optimizer, scaler (AMP), replay buffer, and training counters.
checkpoint_savefile = f"../checkpoints/{SCENARIO_NAME}/{MODEL_TYPE}.pt"
os.makedirs(os.path.dirname(checkpoint_savefile), exist_ok=True)

save_checkpoint = True          # save full training state for resume
load_checkpoint = True         # resume training from checkpoint_savefile
checkpoint_save_replay = True  # include replay buffer in checkpoint (true full resume, larger files) 
checkpoint_interval_epochs = 1  # save every N epochs (replay checkpoints can be large)

save_model = True
load_model = True     # TRUE: Resume Training; FALSE: new training
skip_learning = False

# ----------------------------------------------------------------------------
# Checkpoint helpers (full resume)
# ----------------------------------------------------------------------------
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


def save_full_checkpoint(path: str, agent: "DQNAgent", epoch: int, global_step: int):
    ckpt = {
        "meta": {"epoch": int(epoch), "global_step": int(global_step), "saved_at": float(time())},
        "config": {
            "USE_GRAYSCALE": bool(USE_GRAYSCALE),
            "RESOLUTION": tuple(RESOLUTION),
            "FRAME_STACK_SIZE": int(FRAME_STACK_SIZE),
            "FRAME_REPEAT": int(FRAME_REPEAT),
            "NUM_VARS": int(NUM_VARS),
            "action_size": int(agent.action_size),
            "ATOMS": int(ATOMS),
            "V_MIN": float(V_MIN),
            "V_MAX": float(V_MAX),
            "N_STEP": int(N_STEP),
        },
        "agent": agent.state_dict(include_replay=checkpoint_save_replay),
        "rng": _get_rng_state(),
    }
    torch.save(ckpt, path)


def load_full_checkpoint(path: str, agent: "DQNAgent"):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # For older torch versions that don't have weights_only
        ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {})

    # Guard against accidental mismatched resumes (easy to do when toggling RGB/Gray etc.)
    def _assert_eq(key, cur):
        if key in cfg and cfg[key] != cur:
            raise ValueError(f"Checkpoint mismatch for {key}: ckpt={cfg[key]} current={cur}")

    _assert_eq("USE_GRAYSCALE", bool(USE_GRAYSCALE))
    _assert_eq("RESOLUTION", tuple(RESOLUTION))
    _assert_eq("FRAME_STACK_SIZE", int(FRAME_STACK_SIZE))
    _assert_eq("NUM_VARS", int(NUM_VARS))
    _assert_eq("action_size", int(agent.action_size))

    agent_sd = ckpt["agent"]
    agent.load_state_dict(agent_sd, load_replay=("replay" in agent_sd and agent_sd["replay"] is not None))

    _set_rng_state(ckpt.get("rng"))

    meta = ckpt.get("meta", {})
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0))

# -----------------------------------------------------------------------------
# 🔁 ONE SWITCH: RGB <-> Grayscale
# -----------------------------------------------------------------------------
USE_GRAYSCALE = True

# -----------------------------------------------------------------------------
# Device
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------------
# Rainbow hyperparameters
# -----------------------------------------------------------------------------
ATOMS = 51
V_MIN = -10.0
V_MAX = 30.0

N_STEP = 5

PER_ALPHA = 0.4
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPS = 1e-6

TARGET_UPDATE_EVERY = 3000  # optimizer steps
GRAD_CLIP_NORM = 10.0

BATCH_SIZE = 128
# LEARNING_STARTS = 2 * BATCH_SIZE
LEARNING_STARTS = 5000
TRAIN_EVERY = 1
UPDATES_PER_TRAIN = 1

NUM_VARS = get_num_game_variables(config_file_path)

# Frame stacking
FRAME_STACK_SIZE = 4
FRAME_C = 1 if USE_GRAYSCALE else 3
STACKED_CHANNELS = FRAME_C * FRAME_STACK_SIZE

# Training knobs
RESOLUTION = (96, 128)
TRAIN_EPOCHS = 75
LEARNING_RATE = 0.00025

# With lazy stacking, you can usually go back to 12 safely.
FRAME_REPEAT = 12

# With lazy stacking, 10k is typically safe even for RGB.
REPLAY_MEMORY_SIZE = 50000 if USE_GRAYSCALE else 20000

# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def create_simple_game(visible: bool = False, async_player: bool = False):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if async_player else vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8 if USE_GRAYSCALE else vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Keep your existing behavior (some configs don't expose vars unless explicitly added)
    for gv in game.get_available_game_variables():
        game.add_available_game_variable(gv)

    game.init()
    print("Doom initialized.")
    return game


def preprocess_frame_u8(screen_buf: np.ndarray) -> np.ndarray:
    """Return a single frame as uint8 CHW (C,H,W) where C is 1 or 3."""
    if USE_GRAYSCALE:
        # utils.preprocess returns (1,H,W) float32 in ~[0,1]
        x = preprocess(screen_buf, RESOLUTION)
        return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        # utils.preprocess_rgb returns (3,H,W) float32 in ~[0,255]
        x = preprocess_rgb(screen_buf, RESOLUTION)
        return np.clip(x + 0.5, 0, 255).astype(np.uint8)


class FrameStack:
    """Stores last K frames shaped (C,H,W) and returns concatenated (C*K,H,W)."""

    def __init__(self, k: int, c: int, hw: Tuple[int, int]):
        self.k = k
        self.c = c
        self.h, self.w = hw
        self.frames = deque(maxlen=k)

    def reset(self, first_frame: Optional[np.ndarray] = None):
        self.frames.clear()
        if first_frame is None:
            z = np.zeros((self.c, self.h, self.w), dtype=np.uint8)
            for _ in range(self.k):
                self.frames.append(z)
        else:
            for _ in range(self.k):
                self.frames.append(first_frame.copy())

    def append(self, frame: np.ndarray):
        self.frames.append(frame.copy())

    def get(self) -> np.ndarray:
        return np.concatenate(tuple(self.frames), axis=0)  # uint8


# -----------------------------------------------------------------------------
# Noisy Linear (factorized Gaussian)
# -----------------------------------------------------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# -----------------------------------------------------------------------------
# CNN backbone (same spirit as your StrongCNN)
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
    def __init__(self, in_channels: int = STACKED_CHANNELS):
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


# -----------------------------------------------------------------------------
# Rainbow late-fusion dueling C51 network
# -----------------------------------------------------------------------------
class RainbowLateFusionC51(nn.Module):
    def __init__(
        self,
        action_size: int,
        num_vars: int,
        atoms: int,
        vmin: float,
        vmax: float,
        in_channels: int = STACKED_CHANNELS,
        img_h: int = RESOLUTION[0],
        img_w: int = RESOLUTION[1],
    ):
        super().__init__()
        self.action_size = action_size
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax

        self.cnn = StrongCNN(in_channels=in_channels)
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

        fused_dim = 128 + 64
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
        )

        self.val_fc1 = NoisyLinear(256, 256)
        self.val_fc2 = NoisyLinear(256, atoms)
        self.adv_fc1 = NoisyLinear(256, 256)
        self.adv_fc2 = NoisyLinear(256, action_size * atoms)

        support = torch.linspace(vmin, vmax, atoms)
        self.register_buffer("support", support)

    def reset_noise(self):
        self.val_fc1.reset_noise()
        self.val_fc2.reset_noise()
        self.adv_fc1.reset_noise()
        self.adv_fc2.reset_noise()

    def forward(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = self.fuse(torch.cat([img_feat, vars_feat], dim=1))

        v = self.val_fc2(F.relu(self.val_fc1(fused), inplace=True)).view(-1, 1, self.atoms)
        a = self.adv_fc2(F.relu(self.adv_fc1(fused), inplace=True)).view(-1, self.action_size, self.atoms)
        return v + (a - a.mean(dim=1, keepdim=True))

    def probs(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(img, vars_), dim=-1)

    def q_values(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        p = self.probs(img, vars_)
        return (p * self.support.view(1, 1, -1)).sum(dim=-1)


# -----------------------------------------------------------------------------
# PER SumTree
# -----------------------------------------------------------------------------
class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, p: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, p: float):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s: float):
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            left_sum = float(self.tree[left])
            if s < left_sum:
                idx = left
            else:
                s -= left_sum
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]


    def state_dict(self) -> dict:
        # Note: data is a Python list (ints or None). It's small relative to frames.
        return {
            "capacity": self.capacity,
            "tree": self.tree,
            "data": self.data,
            "write": self.write,
            "n_entries": self.n_entries,
        }

    def load_state_dict(self, sd: dict):
        if int(sd["capacity"]) != self.capacity:
            raise ValueError(f"SumTree capacity mismatch: ckpt={sd['capacity']} current={self.capacity}")
        self.tree = np.asarray(sd["tree"], dtype=np.float32)
        self.data = list(sd["data"])
        self.write = int(sd["write"])
        self.n_entries = int(sd["n_entries"])


# -----------------------------------------------------------------------------
# Lazy frame-stacking PER replay (stores ONE frame per step)
# -----------------------------------------------------------------------------
class LazyFrameStackPERReplay:
    def __init__(
        self,
        capacity: int,
        alpha: float,
        frame_shape: Tuple[int, int, int],
        num_vars: int,
        stack_k: int,
        gamma: float,
        n_step: int,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.max_priority = 1.0

        self.C, self.H, self.W = frame_shape
        self.num_vars = num_vars
        self.k = stack_k
        self.gamma = gamma
        self.n_step = n_step

        self.frames = np.zeros((capacity, self.C, self.H, self.W), dtype=np.uint8)
        self.vars = np.zeros((capacity, num_vars), dtype=np.float16)
        self.actions = np.zeros((capacity,), dtype=np.int16)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

        self.tree = SumTree(capacity)

    def __len__(self):
        return self.tree.n_entries

    def add(self, frame_u8: np.ndarray, vars_f32: np.ndarray, action: int, reward: float, done: bool, priority: Optional[float] = None):
        buf_idx = int(self.tree.write)
        self.frames[buf_idx] = frame_u8
        self.vars[buf_idx] = vars_f32.astype(np.float16)
        self.actions[buf_idx] = int(action)
        self.rewards[buf_idx] = float(reward)
        self.dones[buf_idx] = bool(done)

        # Track max priority (raw TD-error space) but do NOT immediately make
        # the newest transition sample-able for n-step. If we did, stratified PER
        # sampling can get stuck in the last priority segments that correspond to
        # too-new indices with no valid n-step target yet.
        p_raw = self.max_priority if priority is None else float(priority)
        p_raw = max(p_raw, PER_EPS)
        self.max_priority = max(self.max_priority, p_raw)

        # Add the new transition with ZERO priority initially (not sample-able yet).
        # It becomes sample-able once it is at least n_step old.
        self.tree.add(0.0, buf_idx)

        # Activate the transition that just became n_step old.
        # For n_step=3: when we add at t, index (t-3) now has a full 3-step future.
        if self.tree.n_entries > self.n_step:
            valid_idx = (buf_idx - self.n_step) % self.capacity
            leaf_idx = valid_idx + self.capacity - 1
            self.tree.update(leaf_idx, (self.max_priority ** self.alpha))


    def update_priorities(self, tree_idxs: np.ndarray, new_priorities: np.ndarray):
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        for idx, p in zip(tree_idxs, new_priorities):
            p = float(max(float(p), PER_EPS))
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), (p ** self.alpha))

    def _is_valid_buf_idx(self, i: int) -> bool:
        n = self.tree.n_entries
        if n < (self.n_step + 1):
            return False
        if n < self.capacity:
            return i <= (n - self.n_step - 1)
        w = int(self.tree.write)
        for t in range(1, self.n_step + 1):
            if i == ((w - t) % self.capacity):
                return False
        return True

    def _stack_obs(self, idx: int) -> np.ndarray:
        out = np.zeros((self.C * self.k, self.H, self.W), dtype=np.uint8)
        n = self.tree.n_entries
        if n == 0:
            return out

        frames = []  # newest -> oldest
        for t in range(self.k):
            j = idx - t
            if n < self.capacity and j < 0:
                break
            jm = j % self.capacity
            frames.append(self.frames[jm])

            if n < self.capacity:
                if j == 0:
                    break
            else:
                prev = (jm - 1) % self.capacity
                if self.dones[prev]:
                    break

        if not frames:
            return out

        frames = frames[::-1]  # oldest -> newest
        while len(frames) < self.k:
            frames.insert(0, frames[0])

        for s, fr in enumerate(frames):
            out[s * self.C:(s + 1) * self.C] = fr
        return out

    def _n_step_info(self, idx: int):
        R = 0.0
        steps = 0
        done_n = False
        n = self.tree.n_entries
        for t in range(self.n_step):
            j = idx + t
            if n < self.capacity and j >= n:
                break
            jm = j % self.capacity
            R += (self.gamma ** t) * float(self.rewards[jm])
            steps += 1
            if self.dones[jm]:
                done_n = True
                break

        next_idx = (idx + steps) % self.capacity
        discount = float(self.gamma ** steps)
        return R, done_n, next_idx, discount

    def sample(self, batch_size: int, beta: float):
        total = self.tree.total()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"PER: invalid total priority sum: {total}")

        segment = total / batch_size
        eps = np.nextafter(0.0, 1.0)

        batch_buf_idx = []
        tree_idxs = np.empty((batch_size,), dtype=np.int64)
        priorities = np.empty((batch_size,), dtype=np.float32)

        for b in range(batch_size):
            a = segment * b
            bb = segment * (b + 1)

            picked = False
            for _ in range(64):
                s = a + (bb - a) * random.random()
                s = max(s, eps)
                s = min(s, np.nextafter(total, 0.0))

                tidx, p, data = self.tree.get(s)
                if data is None or (not np.isfinite(p)) or p <= 0.0:
                    continue

                buf_idx = int(data)
                if not self._is_valid_buf_idx(buf_idx):
                    continue

                tree_idxs[b] = tidx
                priorities[b] = p
                batch_buf_idx.append(buf_idx)
                picked = True
                break

            if not picked:
                # Fallback: global rejection sampling (helps if a stratified segment
                # lands entirely on invalid/zero-priority leaves).
                for _ in range(256):
                    s = total * random.random()
                    s = max(s, eps)
                    s = min(s, np.nextafter(total, 0.0))
                    tidx, p, data = self.tree.get(s)
                    if data is None or (not np.isfinite(p)) or p <= 0.0:
                        continue
                    buf_idx = int(data)
                    if not self._is_valid_buf_idx(buf_idx):
                        continue
                    tree_idxs[b] = tidx
                    priorities[b] = p
                    batch_buf_idx.append(buf_idx)
                    picked = True
                    break

                if not picked:
                    raise RuntimeError("PER: failed to sample a valid index (buffer may be too small / too new).")

        probs = priorities / (total + 1e-12)
        probs = np.clip(probs, 1e-12, 1.0)
        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= (weights.max() + 1e-8)
        weights = weights.astype(np.float32)

        B = batch_size
        obs_u8 = np.zeros((B, self.C * self.k, self.H, self.W), dtype=np.uint8)
        next_obs_u8 = np.zeros_like(obs_u8)
        vars_f32 = np.zeros((B, self.num_vars), dtype=np.float32)
        next_vars_f32 = np.zeros_like(vars_f32)
        actions = np.zeros((B,), dtype=np.int64)
        rewards_n = np.zeros((B,), dtype=np.float32)
        dones_n = np.zeros((B,), dtype=np.float32)
        discounts = np.zeros((B,), dtype=np.float32)

        for b, i in enumerate(batch_buf_idx):
            obs_u8[b] = self._stack_obs(i)
            vars_f32[b] = self.vars[i].astype(np.float32)
            actions[b] = int(self.actions[i])

            Rn, dn, ni, disc = self._n_step_info(i)
            rewards_n[b] = Rn
            dones_n[b] = 1.0 if dn else 0.0
            discounts[b] = disc

            if dn:
                next_obs_u8[b] = 0
                next_vars_f32[b] = 0
            else:
                next_obs_u8[b] = self._stack_obs(ni)
                next_vars_f32[b] = self.vars[ni].astype(np.float32)

        return obs_u8, vars_f32, actions, rewards_n, next_obs_u8, next_vars_f32, dones_n, discounts, tree_idxs, weights


    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "max_priority": self.max_priority,
            "C": self.C,
            "H": self.H,
            "W": self.W,
            "num_vars": self.num_vars,
            "k": self.k,
            "gamma": self.gamma,
            "n_step": self.n_step,
            "frames": self.frames,
            "vars": self.vars,
            "actions": self.actions,
            "rewards": self.rewards,
            "dones": self.dones,
            "tree": self.tree.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        if int(sd["capacity"]) != self.capacity:
            raise ValueError(f"Replay capacity mismatch: ckpt={sd['capacity']} current={self.capacity}")
        # Validate shapes/config
        if int(sd["C"]) != self.C or int(sd["H"]) != self.H or int(sd["W"]) != self.W:
            raise ValueError(f"Replay frame_shape mismatch: ckpt={(sd['C'], sd['H'], sd['W'])} current={(self.C, self.H, self.W)}")
        if int(sd["num_vars"]) != self.num_vars:
            raise ValueError(f"Replay num_vars mismatch: ckpt={sd['num_vars']} current={self.num_vars}")
        if int(sd["k"]) != self.k:
            raise ValueError(f"Replay stack_k mismatch: ckpt={sd['k']} current={self.k}")
        if int(sd["n_step"]) != self.n_step:
            raise ValueError(f"Replay n_step mismatch: ckpt={sd['n_step']} current={self.n_step}")

        self.alpha = float(sd.get("alpha", self.alpha))
        self.max_priority = float(sd.get("max_priority", 1.0))
        self.gamma = float(sd.get("gamma", self.gamma))

        self.frames = np.asarray(sd["frames"], dtype=np.uint8)
        self.vars = np.asarray(sd["vars"], dtype=np.float16)
        self.actions = np.asarray(sd["actions"], dtype=np.int16)
        self.rewards = np.asarray(sd["rewards"], dtype=np.float32)
        self.dones = np.asarray(sd["dones"], dtype=np.bool_)

        self.tree.load_state_dict(sd["tree"])



# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, action_size: int, lr: float, discount_factor: float, memory_size: int, batch_size: int = BATCH_SIZE):
        self.action_size = action_size
        self.gamma = discount_factor
        self.n_step = N_STEP
        self.batch_size = batch_size

        self.q_net = RainbowLateFusionC51(
            action_size=action_size,
            num_vars=NUM_VARS,
            atoms=ATOMS,
            vmin=V_MIN,
            vmax=V_MAX,
            in_channels=STACKED_CHANNELS,
            img_h=RESOLUTION[0],
            img_w=RESOLUTION[1],
        ).to(DEVICE)

        self.target_net = RainbowLateFusionC51(
            action_size=action_size,
            num_vars=NUM_VARS,
            atoms=ATOMS,
            vmin=V_MIN,
            vmax=V_MAX,
            in_channels=STACKED_CHANNELS,
            img_h=RESOLUTION[0],
            img_w=RESOLUTION[1],
        ).to(DEVICE)

        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)

        frame_shape = (FRAME_C, RESOLUTION[0], RESOLUTION[1])
        self.replay = LazyFrameStackPERReplay(
            capacity=memory_size,
            alpha=PER_ALPHA,
            frame_shape=frame_shape,
            num_vars=NUM_VARS,
            stack_k=FRAME_STACK_SIZE,
            gamma=self.gamma,
            n_step=self.n_step,
        )

        self.beta = PER_BETA_START
        self.learn_step = 0

        self.use_amp = (DEVICE.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.update_target(hard=True)

    def set_train_mode(self):
        self.q_net.train()
        self.target_net.eval()

    def set_eval_mode(self):
        self.q_net.eval()
        self.target_net.eval()

    def update_target(self, hard: bool = True, tau: float = 1.0):
        if hard:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

    def state_dict(self, include_replay: bool = True) -> dict:
        sd = {
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": (self.scaler.state_dict() if self.use_amp else None),
            "beta": float(self.beta),
            "learn_step": int(self.learn_step),
        }
        if include_replay:
            sd["replay"] = self.replay.state_dict()
        return sd

    def load_state_dict(self, sd: dict, load_replay: bool = True):
        self.q_net.load_state_dict(sd["q_net"])
        self.target_net.load_state_dict(sd.get("target_net", sd["q_net"]))
        self.opt.load_state_dict(sd["opt"])

        # Move optimizer state tensors to current device (important if checkpoint loaded on CPU)
        for state in self.opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(DEVICE)

        if self.use_amp and sd.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(sd["scaler"])
            except Exception as e:
                print("Warning: failed to load AMP scaler state:", e)

        self.beta = float(sd.get("beta", PER_BETA_START))
        self.learn_step = int(sd.get("learn_step", 0))

        if load_replay and ("replay" in sd) and (sd["replay"] is not None):
            self.replay.load_state_dict(sd["replay"])

    @staticmethod
    def _project_distribution(next_prob, rewards, dones, discounts, support, vmin, vmax):
        B, atoms = next_prob.shape
        delta_z = (vmax - vmin) / (atoms - 1)

        support = support.view(1, atoms)
        tz = rewards.unsqueeze(-1) + (1.0 - dones.unsqueeze(-1)) * discounts.unsqueeze(-1) * support
        tz = tz.clamp(vmin, vmax)

        b = (tz - vmin) / delta_z
        l = b.floor().long().clamp(0, atoms - 1)
        u = b.ceil().long().clamp(0, atoms - 1)

        m = torch.zeros_like(next_prob)
        neq = (u != l).float()
        m.scatter_add_(dim=-1, index=l, src=next_prob * (u.float() - b) * neq)
        m.scatter_add_(dim=-1, index=u, src=next_prob * (b - l.float()) * neq)

        eq = (u == l).float()
        m.scatter_add_(dim=-1, index=l, src=next_prob * eq)

        m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        return m

    def get_action(self, state_img_u8: np.ndarray, state_vars: np.ndarray, eval_mode: bool = False) -> int:
        img_t = torch.from_numpy(np.expand_dims(state_img_u8, axis=0)).to(DEVICE).float().div_(255.0)
        vars_t = torch.from_numpy(np.expand_dims(state_vars, axis=0)).float().to(DEVICE)

        if eval_mode:
            self.q_net.eval()
        else:
            self.q_net.train()
            self.q_net.reset_noise()

        with torch.no_grad():
            q = self.q_net.q_values(img_t, vars_t)
            return int(torch.argmax(q, dim=1).item())

    def store_step(self, frame_u8: np.ndarray, vars_: np.ndarray, action: int, reward: float, done: bool):
        self.replay.add(frame_u8, vars_, action, reward, done)

    def train_step(self) -> Optional[float]:
        if len(self.replay) < max(self.batch_size, LEARNING_STARTS):
            return None

        self.set_train_mode()

        total_steps = max(1, TRAIN_EPOCHS * LEARNING_STEPS_PER_EPOCH // max(1, TRAIN_EVERY) * max(1, UPDATES_PER_TRAIN))
        frac = min(1.0, self.learn_step / total_steps)
        self.beta = PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

        imgs_u8, vars_, actions, rewards, next_imgs_u8, next_vars, dones, discounts, tree_idxs, is_w = self.replay.sample(
            self.batch_size, beta=self.beta
        )

        imgs_t = torch.from_numpy(imgs_u8).to(DEVICE).float()
        next_imgs_t = torch.from_numpy(next_imgs_u8).to(DEVICE).float()
        # dequant noise: (x + U[0,1)) / 255
        imgs_t = (imgs_t + torch.rand_like(imgs_t)) / 255.0
        next_imgs_t = (next_imgs_t + torch.rand_like(next_imgs_t)) / 255.0
        vars_t = torch.from_numpy(vars_).to(DEVICE)
        actions_t = torch.from_numpy(actions).to(DEVICE)
        rewards_t = torch.from_numpy(rewards).to(DEVICE)
        next_vars_t = torch.from_numpy(next_vars).to(DEVICE)
        dones_t = torch.from_numpy(dones).to(DEVICE)
        discounts_t = torch.from_numpy(discounts).to(DEVICE)
        is_w_t = torch.from_numpy(is_w).to(DEVICE)

        self.opt.zero_grad(set_to_none=True)

        self.q_net.reset_noise()
        self.target_net.reset_noise()

        use_amp = self.use_amp
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

        with autocast_ctx:
            logits = self.q_net(imgs_t, vars_t)
            atoms = logits.size(-1)
            act_idx = actions_t.view(-1, 1, 1).expand(-1, 1, atoms)
            logits_a = logits.gather(1, act_idx).squeeze(1)
            log_prob_a = F.log_softmax(logits_a, dim=-1)

            with torch.no_grad():
                next_logits_online = self.q_net(next_imgs_t, next_vars_t)
                next_prob_online = torch.softmax(next_logits_online, dim=-1)
                support = self.q_net.support.view(1, 1, -1)
                next_q_online = (next_prob_online * support).sum(dim=-1)
                next_actions = next_q_online.argmax(dim=1)

                next_logits_target = self.target_net(next_imgs_t, next_vars_t)
                next_prob_target = torch.softmax(next_logits_target, dim=-1)
                next_act_idx = next_actions.view(-1, 1, 1).expand(-1, 1, atoms)
                next_prob_a = next_prob_target.gather(1, next_act_idx).squeeze(1)

                target_dist = self._project_distribution(
                    next_prob=next_prob_a,
                    rewards=rewards_t,
                    dones=dones_t,
                    discounts=discounts_t,
                    support=self.q_net.support,
                    vmin=V_MIN,
                    vmax=V_MAX,
                )

            per_sample_loss = -(target_dist * log_prob_a).sum(dim=-1)
            loss = (per_sample_loss * is_w_t).mean()

        if use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP_NORM)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP_NORM)
            self.opt.step()

        new_prios = (per_sample_loss.detach().abs().clamp_min(PER_EPS) + PER_EPS).cpu().numpy()
        self.replay.update_priorities(tree_idxs, new_prios)

        self.learn_step += 1
        if (self.learn_step % TARGET_UPDATE_EVERY) == 0:
            self.update_target(hard=True)

        return float(loss.item())


# -----------------------------------------------------------------------------
# Eval / train / watch loops
# -----------------------------------------------------------------------------
def test(game, agent: DQNAgent, actions):
    print("\nTesting...")
    agent.set_eval_mode()

    scores = []
    for _ in trange(TEST_EPISODES_PER_EPOCH, leave=False):
        game.new_episode()
        stacker = FrameStack(FRAME_STACK_SIZE, FRAME_C, RESOLUTION)

        gs = game.get_state()
        if gs is None:
            scores.append(0.0)
            continue

        frame = preprocess_frame_u8(gs.screen_buffer)
        stacker.reset(frame)
        obs = stacker.get()
        vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)

        while not game.is_episode_finished():
            a = agent.get_action(obs, vars_, eval_mode=True)
            game.make_action(actions[a], FRAME_REPEAT)

            if game.is_episode_finished():
                break

            gs = game.get_state()
            if gs is None:
                break

            frame = preprocess_frame_u8(gs.screen_buffer)
            stacker.append(frame)
            obs = stacker.get()
            vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)

        scores.append(game.get_total_reward())

    scores = np.array(scores, dtype=np.float32)
    print(
        "Results: mean: {:.1f} +/- {:.1f}, min: {:.1f}, max: {:.1f}".format(
            scores.mean(), scores.std(), scores.min(), scores.max()
        )
    )



def run(game, agent: DQNAgent, actions, start_epoch: int = 0, start_global_step: int = 0):
    start_time = time()
    global_step = int(start_global_step)

    for epoch in range(start_epoch, TRAIN_EPOCHS):
        print(f"\nEpoch #{epoch + 1}")

        game.new_episode()
        stacker = FrameStack(FRAME_STACK_SIZE, FRAME_C, RESOLUTION)

        gs = game.get_state()
        if gs is None:
            print("Warning: gs is None at episode start; skipping epoch.")
            continue

        frame = preprocess_frame_u8(gs.screen_buffer)
        stacker.reset(frame)
        obs = stacker.get()
        vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)

        train_scores = []
        losses = []

        for _ in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
            a = agent.get_action(obs, vars_, eval_mode=False)
            r = game.make_action(actions[a], FRAME_REPEAT)
            done = game.is_episode_finished()

            agent.store_step(frame, vars_, a, r, done)

            if (global_step % TRAIN_EVERY) == 0:
                for _ in range(UPDATES_PER_TRAIN):
                    l = agent.train_step()
                    if l is not None:
                        losses.append(l)

            if done:
                train_scores.append(game.get_total_reward())

                game.new_episode()
                gs = game.get_state()
                if gs is None:
                    global_step += 1
                    continue

                frame = preprocess_frame_u8(gs.screen_buffer)
                stacker.reset(frame)
                obs = stacker.get()
                vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)
            else:
                ngs = game.get_state()
                if ngs is None:
                    train_scores.append(game.get_total_reward())
                    game.new_episode()
                    gs = game.get_state()
                    if gs is None:
                        global_step += 1
                        continue
                    frame = preprocess_frame_u8(gs.screen_buffer)
                    stacker.reset(frame)
                    obs = stacker.get()
                    vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)
                else:
                    frame = preprocess_frame_u8(ngs.screen_buffer)
                    stacker.append(frame)
                    obs = stacker.get()
                    vars_ = preprocess_vars_safe(ngs.game_variables, NUM_VARS)

            global_step += 1

        if len(train_scores) > 0:
            ts = np.array(train_scores, dtype=np.float32)
            print(
                "Train: mean: {:.1f} +/- {:.1f}, min: {:.1f}, max: {:.1f}".format(
                    ts.mean(), ts.std(), ts.min(), ts.max()
                )
            )
        else:
            print("Train: no completed episodes this epoch.")

        if len(losses) > 0:
            print(f"Loss: mean={np.mean(losses):.4f}, last={losses[-1]:.4f}")

        test(game, agent, actions)

        if save_model:
            print("Saving model to:", model_savefile)
            torch.save(agent.q_net.state_dict(), model_savefile)

        if save_checkpoint and ((epoch + 1) % checkpoint_interval_epochs == 0):
            print("Saving checkpoint to:", checkpoint_savefile, f"(include_replay={checkpoint_save_replay})")
            save_full_checkpoint(checkpoint_savefile, agent, epoch=epoch + 1, global_step=global_step)

        elapsed = (time() - start_time) / 60.0
        print(f"Total elapsed time: {elapsed:.2f} minutes")

    game.close()



def watch_trained(agent: DQNAgent, actions):
    game = create_simple_game(visible=True, async_player=True)
    agent.set_eval_mode()

    total = 0.0
    for ep in range(EPISODES_TO_WATCH):
        game.new_episode()
        stacker = FrameStack(FRAME_STACK_SIZE, FRAME_C, RESOLUTION)

        gs = game.get_state()
        if gs is None:
            print(f"Episode {ep + 1}: gs is None at start.")
            continue

        frame = preprocess_frame_u8(gs.screen_buffer)
        stacker.reset(frame)
        obs = stacker.get()
        vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)

        while not game.is_episode_finished():
            a = agent.get_action(obs, vars_, eval_mode=True)
            game.set_action(actions[a])
            for _ in range(FRAME_REPEAT):
                game.advance_action()

            if game.is_episode_finished():
                break

            gs = game.get_state()
            if gs is None:
                break

            frame = preprocess_frame_u8(gs.screen_buffer)
            stacker.append(frame)
            obs = stacker.get()
            vars_ = preprocess_vars_safe(gs.game_variables, NUM_VARS)

        score = game.get_total_reward()
        total += score
        print(f"Episode {ep + 1} Total score: {score}")
        sleep(1.0)

    print(f"-----Average Score: {total / EPISODES_TO_WATCH:.2f}-----")
    game.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("----------MODEL CONFIGURATION----------")
    print("MODEL_TYPE:", MODEL_TYPE)
    print("SCENARIO_NAME:", SCENARIO_NAME)
    print("DEVICE:", DEVICE)
    print("USE_GRAYSCALE:", USE_GRAYSCALE)
    print("FRAME_STACK_SIZE:", FRAME_STACK_SIZE)
    print("STACKED_CHANNELS:", STACKED_CHANNELS)
    print("LEARNING_RATE:", LEARNING_RATE)
    print("DISCOUNT_FACTOR:", DISCOUNT_FACTOR)
    print("TRAIN_EPOCHS:", TRAIN_EPOCHS)
    print("LEARNING_STEPS_PER_EPOCH:", LEARNING_STEPS_PER_EPOCH)
    print("TEST_EPISODES_PER_EPOCH:", TEST_EPISODES_PER_EPOCH)
    print("REPLAY_MEMORY_SIZE:", REPLAY_MEMORY_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("FRAME_REPEAT:", FRAME_REPEAT)
    print("RESOLUTION:", RESOLUTION)
    print("MODEL_SAVEFILE:", model_savefile)
    print("CHECKPOINT_SAVEFILE:", checkpoint_savefile)
    print("load_checkpoint:", load_checkpoint)
    print("save_checkpoint:", save_checkpoint)
    print("checkpoint_save_replay:", checkpoint_save_replay)
    print("checkpoint_interval_epochs:", checkpoint_interval_epochs)
    print("Rainbow: ATOMS=", ATOMS, "V_MIN=", V_MIN, "V_MAX=", V_MAX, "N_STEP=", N_STEP)

    game = create_simple_game(visible=False, async_player=False)

    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print("ACTIONS:", len(actions))

    agent = DQNAgent(
        action_size=len(actions),
        lr=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        memory_size=REPLAY_MEMORY_SIZE,
    )


    # Resume training (full) or load weights (inference only)
    start_epoch = 0
    start_global_step = 0
    if load_checkpoint and os.path.exists(checkpoint_savefile):
        print("Loading checkpoint from:", checkpoint_savefile)
        start_epoch, start_global_step = load_full_checkpoint(checkpoint_savefile, agent)
        print(f"Resuming from epoch={start_epoch}, global_step={start_global_step}")
    elif load_model and os.path.exists(model_savefile):
        print("Loading model weights from:", model_savefile)
        agent.q_net.load_state_dict(torch.load(model_savefile, map_location=DEVICE))
        agent.update_target(hard=True)

    if not skip_learning:
        run(game, agent, actions, start_epoch=start_epoch, start_global_step=start_global_step)
        print("======================================")
        print("Training finished. Time to watch!")
    else:
        game.close()

    watch_trained(agent, actions)
