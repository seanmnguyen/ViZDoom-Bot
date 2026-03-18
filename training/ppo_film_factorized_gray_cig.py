#!/usr/bin/env python3
"""
ppo_film_factorized_gray.py

PPO + FiLM-conditioned vision backbone for ViZDoom using GRAYSCALE screen buffer,
with a FACTORIZED action policy (MultiDiscrete-style heads) instead of enumerating
all 2^N button combinations.

Why factorized heads?
- deathmatch.cfg exposes 20 buttons, but enumerating all binary combinations yields 2^20
  discrete actions (~1,048,576). A single softmax over that many actions is extremely
  inefficient and (often) untrainable.
- Factorizing the policy into multiple small categorical heads lets PPO learn each
  control dimension with a small output layer and combines log-probabilities additively.

This script keeps the FiLM backbone from ppo_film_gray.py, but replaces the actor head.

Default factorization for deathmatch.cfg buttons:
- move_fb:  {back, none, forward}                 -> MOVE_BACKWARD / MOVE_FORWARD
- attack:   {no, yes}                             -> ATTACK
- speed:    {walk, run} (optional; default always run = 1)
- weapon:   {none, w1..w6, next, prev} (optional)

You can tune bins at ACTION_* constants below.

Checkpointing:
- Same "epoch-boundary full checkpointing" style as your other PPO scripts:
  model + optimizer + scaler + RNG + counters (+ vars RMS stats).

Notes:
- We deliberately set FiLM layers to identity at init (gamma=0,beta=0) so the network
  starts as a plain CNN and can learn conditioning gradually.
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import trange

import vizdoom as vzd
from utils import *  # shared constants + preprocess(img) + get_num_game_variables


# -----------------------------------------------------------------------------
# Scenario / save naming
# -----------------------------------------------------------------------------
# Set this to "deathmatch" for your deathmatch.cfg run.
SCENARIO_NAME = "cig_learning"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

MODEL_TYPE = os.path.splitext(os.path.basename(__file__))[0] + "_slow3"
model_savefile = f"../models/{SCENARIO_NAME}/{MODEL_TYPE}.pth"
os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

checkpoint_savefile = f"../checkpoints/{SCENARIO_NAME}/{MODEL_TYPE}.pt"
os.makedirs(os.path.dirname(checkpoint_savefile), exist_ok=True)

save_model = True
save_checkpoint = True
load_checkpoint = True       # resume if checkpoint exists
load_model_weights = False   # fallback: load model_savefile weights (no optimizer state)
skip_learning = False

checkpoint_interval_epochs = 1


# -----------------------------------------------------------------------------
# PPO Hyperparameters (start here for tuning)
# -----------------------------------------------------------------------------
LEARNING_RATE_PPO = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
ENTROPY_COEF = 0.015 / 4     # deathmatch often benefits from a bit more exploration
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

PPO_EPOCHS = 4
MINI_BATCH_SIZE = 128        # slightly bigger minibatches help in noisy deathmatch

# Rollout collection
TRAIN_EPOCHS_PPO = 200
STEPS_PER_EPOCH = 8192       # bigger rollout reduces variance in PPO updates

# Testing
TEST_EPISODES = 30           # Quick evaluate calls, only 10 episodes
EVAL_EVERY_EPOCHS = 1        # Run quick evaluate every epoch (unless full evaluate)
FULL_TEST_EPISODES = 90      # Full evaluate calls, all 100 episodes
FULL_EVAL_EVERY = 15         # Run full evaluate every 10 epochs
MAX_DEATHS = 50              # Effectively allow infinite respawns until episode end
TEST_DEATH_CAP = 5           # Only 5 respawns for quick evaluate calls

# Frame stacking
FRAME_STACK_SIZE = 4

# For deathmatch you often want lower frame_repeat than defend_the_center
FRAME_REPEAT_EFFECTIVE = 6   # try 4 or 6 for smoother aiming
RESOLUTION_EFFECTIVE = RESOLUTION

# Game vars (scenario-specific)
NUM_VARS = get_num_game_variables(config_file_path)

# Reward Structure
FRAG_REWARD = 5.0
DEATH_PENALTY = 1.0
TIME_PENALTY = 1e-4

BOTS = 20

# -----------------------------------------------------------------------------
# Device + AMP
# -----------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

USE_AMP = (DEVICE.type == "cuda")
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
    - If expected <= 2, preserves legacy defend_the_center mapping via utils.preprocess_vars_safe
    - Else: pad/truncate and (optionally) running-mean-std normalize.
    """
    if expected <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Keep old behavior for defend_the_center-like 2-var scenarios
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

    # Stateless fallback: signed log scaling
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
    action_spec: dict,
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
            "USE_AMP": bool(agent.use_amp),
            "action_spec": action_spec,
        },
        "agent": agent.state_dict(),
        "rng": _get_rng_state(),
    }
    torch.save(ckpt, path)


def load_full_checkpoint(path: str, agent: "PPOAgent", *, expected_action_spec: dict):
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

    # Action spec mismatch guard (super important when tuning bins/head sizes)
    if "action_spec" in cfg and cfg["action_spec"] != expected_action_spec:
        raise ValueError(f"Checkpoint mismatch for action_spec:\nckpt={cfg['action_spec']}\ncur={expected_action_spec}")

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
def create_simple_game(visible: bool = False, bots: int = BOTS, map_name: str = "map01", async_player: bool = False):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)

    # Pick the actual CIG map explicitly (matches examples)
    game.set_doom_map(map_name)

    # Start a local deathmatch “server” with only this AI + bots (like cig_multiplayer_bots.py)
    game.add_game_args(
        "-host 1 -deathmatch +timelimit 2.0 "
        "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 "
        "+sv_spawnfarthest 0 +sv_nocrouch 1 +viz_respawn_delay 1 +viz_nocheat 1"
    )
    game.add_game_args("+viz_bots_path ../../scenarios/perfect_bots.cfg")  # same as your example script

    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if async_player else vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)  # match cfg, faster than 640x480

    # (Optional but recommended)
    if hasattr(game, "set_render_hud"):
        game.set_render_hud(False)

    # Training, shorter episodes
    game.set_episode_timeout(4200)  # ~2 mins, instead of 12

    game.init()

    # Spawn bots now (and also at every new_episode)
    game.send_game_command("removebots")
    for _ in range(bots):
        game.send_game_command("addbot")

    print("Doom initialized. For player type:", vzd.Mode.ASYNC_PLAYER if async_player else vzd.Mode.PLAYER)
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
        z = np.zeros(self.frame_shape, dtype=np.float32)
        for _ in range(self.stack_size):
            self.frames.append(z.copy())

    def push(self, frame_chw: np.ndarray):
        # frame_chw from utils.preprocess is (1, H, W)
        self.frames.append(frame_chw[0])

    def get(self) -> np.ndarray:
        # (K, H, W)
        return np.asarray(self.frames, dtype=np.float32)


# -----------------------------------------------------------------------------
# Factorized Action Space (deathmatch-friendly)
# -----------------------------------------------------------------------------
# Heads enabled (weapon + speed can be toggled)
ENABLE_SPEED_HEAD = False
ENABLE_WEAPON_HEAD = False


@dataclass(frozen=True)
class ActionSpec:
    """Defines the factorized action heads and their discrete sizes."""
    head_names: Tuple[str, ...]
    head_sizes: Tuple[int, ...]

    def as_dict(self) -> dict:
        return {"head_names": list(self.head_names), "head_sizes": list(self.head_sizes)}


class FactorizedActionMapper:
    """
    Maps per-head discrete indices -> ViZDoom action vector (len = n_buttons).

    This mapper assumes the button set from cig_learning.cfg, but it is robust:
    - If a button isn't present, it will be ignored.
    """
    def __init__(self, game: vzd.DoomGame):
        self.buttons = list(game.get_available_buttons())
        self.n_buttons = len(self.buttons)
        self.btn_to_idx = {b: i for i, b in enumerate(self.buttons)}

        # Cache indices (or None if absent)
        def idx(btn):
            return self.btn_to_idx.get(btn, None)

        self.i_attack = idx(vzd.Button.ATTACK)
        self.i_speed = idx(vzd.Button.SPEED)
        self.i_strafe_mod = idx(vzd.Button.STRAFE)
        self.i_use = idx(vzd.Button.USE)

        self.i_move_f = idx(vzd.Button.MOVE_FORWARD)
        self.i_move_b = idx(vzd.Button.MOVE_BACKWARD)
        self.i_move_l = idx(vzd.Button.MOVE_LEFT)
        self.i_move_r = idx(vzd.Button.MOVE_RIGHT)
        self.i_turn_l = idx(vzd.Button.TURN_LEFT)
        self.i_turn_r = idx(vzd.Button.TURN_RIGHT)

        self.i_sel_w = {
            1: idx(vzd.Button.SELECT_WEAPON1),
            2: idx(vzd.Button.SELECT_WEAPON2),
            3: idx(vzd.Button.SELECT_WEAPON3),
            4: idx(vzd.Button.SELECT_WEAPON4),
            5: idx(vzd.Button.SELECT_WEAPON5),
            6: idx(vzd.Button.SELECT_WEAPON6),
        }
        self.i_next_w = idx(vzd.Button.SELECT_NEXT_WEAPON)
        self.i_prev_w = idx(vzd.Button.SELECT_PREV_WEAPON)

        self.spec = self.action_spec()

    def action_spec(self) -> ActionSpec:
        names = ["move_fb", "strafe_lr", "turn_lr", "attack", "use"]
        sizes = [3, 3, 3, 2, 2]
        if ENABLE_SPEED_HEAD:
            names.append("speed")
            sizes.append(2)
        if ENABLE_WEAPON_HEAD:
            names.append("weapon")
            sizes.append(1 + 6 + 2)  # none, w1..w6, next, prev  => 9
        return ActionSpec(tuple(names), tuple(sizes))

    def decode(self, a: np.ndarray) -> List[int]:
        """
        a: (H,) indices for each head in the order of action_spec().head_names
        returns: ViZDoom action list of ints length n_buttons
        """
        a = np.asarray(a, dtype=np.int64).reshape(-1)
        spec = self.spec
        if a.shape[0] != len(spec.head_names):
            raise ValueError(f"Expected {len(spec.head_names)} head actions, got {a.shape[0]}")

        out = [0] * self.n_buttons

        # Helper setters (ignore if missing)
        def set_bin(idx_, val: int):
            if idx_ is not None:
                out[idx_] = int(val)

        # Read heads
        h = dict(zip(spec.head_names, a.tolist()))

        # move_fb: 0=back, 1=none, 2=forward
        mf = int(h["move_fb"])
        set_bin(self.i_move_b, 1 if mf == 0 else 0)
        set_bin(self.i_move_f, 1 if mf == 2 else 0)

        # strafe_lr: 0=left, 1=none, 2=right
        slr = int(h["strafe_lr"])
        set_bin(self.i_move_l, 1 if slr == 0 else 0)
        set_bin(self.i_move_r, 1 if slr == 2 else 0)

        # turn_lr: 0=left, 1=none, 2=right
        tlr = int(h["turn_lr"])
        set_bin(self.i_turn_l, 1 if tlr == 0 else 0)
        set_bin(self.i_turn_r, 1 if tlr == 2 else 0)

        # attack/use
        set_bin(self.i_attack, int(h["attack"]))
        set_bin(self.i_use, int(h["use"]))

        # speed
        if ENABLE_SPEED_HEAD:
            set_bin(self.i_speed, int(h["speed"]))
        else:
            # Always run, if SPEED is present
            set_bin(self.i_speed, 1)

        # Optional weapon selection
        if ENABLE_WEAPON_HEAD:
            w = int(h["weapon"])  # 0 none, 1..6 weapon, 7 next, 8 prev
            # clear all weapon selects (already zero)
            if 1 <= w <= 6:
                set_bin(self.i_sel_w.get(w, None), 1)
            elif w == 7:
                set_bin(self.i_next_w, 1)
            elif w == 8:
                set_bin(self.i_prev_w, 1)

        # We also keep STRAFE modifier off (we use dedicated movement controls)
        set_bin(self.i_strafe_mod, 0)

        return out


# -----------------------------------------------------------------------------
# FiLM Actor-Critic Network (factorized heads)
# -----------------------------------------------------------------------------
def norm2d(c: int) -> nn.Module:
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
    """Produces (gamma, beta) for a channel count C from a vars embedding."""
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
    """CNN backbone with FiLM modulation at multiple feature-map stages."""
    def __init__(self, in_channels: int, vars_dim: int, embed_dim: int = 128):
        super().__init__()
        self.in_channels = int(in_channels)
        self.vars_dim = int(vars_dim)
        self.embed_dim = int(embed_dim)

        # Vars embed used for all FiLM blocks.
        # LayerNorm is OK for >=5 deathmatch vars. If you ever train with 2 vars, consider Identity.
        self.vars_embed = nn.Sequential(
            nn.LayerNorm(self.vars_dim),
            nn.Linear(self.vars_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.embed_dim),
            nn.ReLU(inplace=True),
        )

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


class ActorCriticFiLMFactorized(nn.Module):
    """
    Actor-Critic with factorized categorical heads.

    Returns:
      actions: (B, H) int64 indices
      logp:    (B,) summed log-prob
      entropy: (B,) summed entropy
      value:   (B,) critic value
    """
    def __init__(self, head_sizes: List[int], num_vars: int, img_hw: Tuple[int, int], in_channels: int):
        super().__init__()
        self.num_vars = int(num_vars)
        self.head_sizes = [int(x) for x in head_sizes]
        self.num_heads = len(self.head_sizes)

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

        # One linear layer per head
        self.actor_heads = nn.ModuleList([nn.Linear(256, hs) for hs in self.head_sizes])
        self.critic = nn.Linear(256, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        # Orthogonal init for PPO
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            elif isinstance(m, nn.Linear):
                # Keep FiLM fc weights at zero (identity modulation)
                # (they're inside FiLM modules; skip them)
                if hasattr(self, "cnn") and m in [self.cnn.film1.fc, self.cnn.film2.fc, self.cnn.film3.fc, self.cnn.film4.fc]:
                    continue
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Smaller init on actor heads, typical PPO
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward_features(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        return self.fc(self.cnn(img, vars_))

    def _dists(self, features: torch.Tensor) -> List[Categorical]:
        return [Categorical(logits=head(features)) for head in self.actor_heads]

    def get_action_and_value(
        self,
        img: torch.Tensor,
        vars_: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ):
        features = self.forward_features(img, vars_)
        dists = self._dists(features)

        if action is None:
            acts = [d.sample() for d in dists]
            action = torch.stack(acts, dim=1)  # (B,H)
        else:
            # Expect (B,H)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            if action.shape[1] != len(dists):
                raise ValueError(f"Expected action shape (B,{len(dists)}) got {tuple(action.shape)}")

        logps = []
        ents = []
        for i, d in enumerate(dists):
            ai = action[:, i]
            logps.append(d.log_prob(ai))
            ents.append(d.entropy())

        logp = torch.stack(logps, dim=1).sum(dim=1)
        entropy = torch.stack(ents, dim=1).sum(dim=1)

        value = self.critic(features).squeeze(-1)
        return action, logp, entropy, value

    def get_value(self, img: torch.Tensor, vars_: torch.Tensor):
        features = self.forward_features(img, vars_)
        return self.critic(features).squeeze(-1)

    @torch.no_grad()
    def get_deterministic_action(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(img, vars_)
        actions = []
        for head in self.actor_heads:
            logits = head(features)
            actions.append(torch.argmax(logits, dim=1))
        return torch.stack(actions, dim=1)  # (B,H)


# -----------------------------------------------------------------------------
# PPO Rollout Buffer
# -----------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.img_states = []
        self.var_states = []
        self.actions = []     # (H,)
        self.log_probs = []   # float
        self.rewards = []     # float
        self.dones = []       # float
        self.values = []      # float

    def clear(self):
        self.img_states.clear()
        self.var_states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, img_state, var_state, action_heads, log_prob, reward, done, value):
        self.img_states.append(img_state)
        self.var_states.append(var_state)
        self.actions.append(np.asarray(action_heads, dtype=np.int64))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

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
                np.asarray([self.actions[i] for i in bi], dtype=np.int64),     # (B,H)
                np.asarray([self.log_probs[i] for i in bi], dtype=np.float32), # (B,)
                returns[bi].astype(np.float32),
                advantages[bi].astype(np.float32),
            )


# -----------------------------------------------------------------------------
# PPO Agent
# -----------------------------------------------------------------------------
class PPOAgent:
    def __init__(
        self,
        action_mapper: FactorizedActionMapper,
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
        self.action_mapper = action_mapper
        self.spec = action_mapper.action_spec()
        self.action_spec_dict = self.spec.as_dict()

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)

        self.use_amp = USE_AMP

        self.network = ActorCriticFiLMFactorized(
            head_sizes=list(self.spec.head_sizes),
            num_vars=NUM_VARS,
            img_hw=RESOLUTION_EFFECTIVE,
            in_channels=FRAME_STACK_SIZE,
        ).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scaler = SCALER
        self.buffer = RolloutBuffer()

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
        """
        Returns:
          action_heads: np.ndarray (H,)
          logp: float
        """
        img_t = torch.from_numpy(img_state).unsqueeze(0).to(DEVICE)   # (1,K,H,W)
        vars_t = torch.from_numpy(var_state).unsqueeze(0).to(DEVICE)  # (1,V)

        if deterministic:
            action = self.network.get_deterministic_action(img_t, vars_t)
            # Compute logp for bookkeeping (optional)
            _, logp, _, _ = self.network.get_action_and_value(img_t, vars_t, action)
            return action.squeeze(0).detach().cpu().numpy().astype(np.int64)
        else:
            action, logp, _, _ = self.network.get_action_and_value(img_t, vars_t)

        return action.squeeze(0).detach().cpu().numpy().astype(np.int64), float(logp.item())

    def compute_gae(self, next_value: float):
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        n = len(rewards)

        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_nonterminal = 1.0 - dones[t]
            next_values = next_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)

    def update(self, next_value: float) -> Dict[str, float]:
        returns, advantages = self.compute_gae(next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for img_b, var_b, act_b, oldlog_b, ret_b, adv_b in self.buffer.get_batches(
                self.mini_batch_size, returns, advantages
            ):
                img_t = torch.from_numpy(img_b).to(DEVICE)
                vars_t = torch.from_numpy(var_b).to(DEVICE)
                act_t = torch.from_numpy(act_b).to(DEVICE)         # (B,H)
                oldlog_t = torch.from_numpy(oldlog_b).to(DEVICE)   # (B,)
                ret_t = torch.from_numpy(ret_b).to(DEVICE)
                adv_t = torch.from_numpy(adv_b).to(DEVICE)

                with torch.amp.autocast(str(DEVICE), enabled=self.use_amp):
                    _, logp, entropy, value = self.network.get_action_and_value(img_t, vars_t, act_t)

                    ratio = torch.exp(logp - oldlog_t)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(value, ret_t)
                    entropy_bonus = entropy.mean()

                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

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

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy_bonus.item())
                n_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }

    def state_dict(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": (self.scaler.state_dict() if self.use_amp else None),
            "vars_rms": (self.vars_rms.state_dict() if self.vars_rms is not None else None),
        }

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
# Evaluation + training loops (same CLI style as ppo_late_fusion_gray.py)
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(game: vzd.DoomGame, agent: PPOAgent, mapper: FactorizedActionMapper, num_episodes: int, death_cap: int = MAX_DEATHS) -> float:
    agent.set_eval_mode()
    scores = []
    frame_stack = FrameStack(FRAME_STACK_SIZE, RESOLUTION_EFFECTIVE)

    for _ in trange(num_episodes, leave=False):
        game.new_episode()
        frame_stack.reset()
        game.send_game_command("removebots")
        for _ in range(BOTS):
            game.send_game_command("addbot")
        ep_reward = 0.0
        respawns_this_ep = 0

        while not game.is_episode_finished():
            if game.is_player_dead():
                respawns_this_ep += 1
                if respawns_this_ep >= death_cap:
                    break
                game.respawn_player()
                frame_stack.reset()
                continue

            gs = game.get_state()
            if gs is None:
                ep_reward = 0.0
                continue

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

            a_heads = agent.get_action(state_img, state_vars, deterministic=True)
            viz_action = mapper.decode(a_heads)

            prev_frags  = int(game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            prev_deaths = int(game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

            game.make_action(viz_action, FRAME_REPEAT_EFFECTIVE)

            frags  = int(game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            deaths = int(game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

            # Simple, stable shaping
            r = FRAG_REWARD * (frags - prev_frags) - DEATH_PENALTY * (deaths - prev_deaths) - TIME_PENALTY
            ep_reward += r

        scores.append(float(ep_reward))

    scores = np.asarray(scores, dtype=np.float32)
    print(
        "Results: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
            float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
        )
    )
    agent.set_train_mode()
    return float(scores.mean())


def train(
    game: vzd.DoomGame,
    agent: PPOAgent,
    mapper: FactorizedActionMapper,
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

        game.new_episode()
        frame_stack.reset()
        game.send_game_command("removebots")
        for _ in range(BOTS):
            game.send_game_command("addbot")

        train_episode_rewards = []
        ep_reward = 0.0

        # Collect rollout
        for _ in trange(STEPS_PER_EPOCH, desc="Collecting rollout", leave=False):
            gs = game.get_state()
            if gs is None:
                 # If we're dead or in a respawn transition, don't start a brand new episode.
                if game.is_player_dead():
                    game.respawn_player()
                # Otherwise, treat it as a rare glitch / terminal and restart.
                else:
                    train_episode_rewards.append(ep_reward)
                    ep_reward = 0.0
                    game.new_episode()
                    game.send_game_command("removebots")
                    for _ in range(BOTS):
                        game.send_game_command("addbot")
                frame_stack.reset()
                continue

            frame = preprocess(gs.screen_buffer, RESOLUTION_EFFECTIVE)
            frame_stack.push(frame)
            state_img = frame_stack.get()

            state_vars = preprocess_vars_safe_general(
                gs.game_variables,
                NUM_VARS,
                normalizer=agent.vars_rms,
                update=True,
                clip=5.0,
            )

            img_t = torch.from_numpy(state_img).unsqueeze(0).to(DEVICE)
            vars_t = torch.from_numpy(state_vars).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                actions_t, logp_t, _, value_t = agent.network.get_action_and_value(img_t, vars_t)

            a_heads = actions_t.squeeze(0).detach().cpu().numpy().astype(np.int64)
            viz_action = mapper.decode(a_heads)

            prev_frags  = int(game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            prev_deaths = int(game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

            game.make_action(viz_action, FRAME_REPEAT_EFFECTIVE)

            frags  = int(game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            deaths = int(game.get_game_variable(vzd.GameVariable.DEATHCOUNT))

            # Simple, stable shaping
            r = FRAG_REWARD * (frags - prev_frags) - DEATH_PENALTY * (deaths - prev_deaths) - TIME_PENALTY
            episode_finished = bool(game.is_episode_finished())
            player_dead = bool(game.is_player_dead())
            done = episode_finished or player_dead
            ep_reward += r

            agent.buffer.add(
                img_state=state_img,
                var_state=state_vars,
                action_heads=a_heads,
                log_prob=float(logp_t.item()),
                reward=r,
                done=float(done),
                value=float(value_t.item()),
            )
            global_step += 1

            if done:
                train_episode_rewards.append(ep_reward)
                ep_reward = 0.0

                if episode_finished:
                    game.new_episode()
                    game.send_game_command("removebots")
                    for _ in range(BOTS):
                        game.send_game_command("addbot")
                else:
                    # died, but match deathmatch semantics: continue the same episode
                    game.respawn_player()

                frame_stack.reset()
                continue

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

        stats = agent.update(next_value)
        print("\nPPO update stats:")
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
        if (epoch + 1) % FULL_EVAL_EVERY == 0:
            mean_test_reward = evaluate(game, agent, mapper, num_episodes=FULL_TEST_EPISODES, death_cap=MAX_DEATHS)
        elif (epoch + 1) % EVAL_EVERY_EPOCHS == 0:
            mean_test_reward = evaluate(game, agent, mapper, num_episodes=TEST_EPISODES, death_cap=TEST_DEATH_CAP)
        else:  # Skip eval this time
            mean_test_reward = 0
            print("Skipping evaluate...")

        if save_model and mean_test_reward > best_mean_reward:
            best_mean_reward = mean_test_reward
            print(f"New best model! Saving weights to: {model_savefile}")
            torch.save(agent.network.state_dict(), model_savefile)

        if save_checkpoint and ((epoch + 1) % checkpoint_interval_epochs == 0):
            print(f"Saving checkpoint to: {checkpoint_savefile}")
            save_full_checkpoint(
                checkpoint_savefile,
                agent,
                epoch=epoch + 1,
                global_step=global_step,
                best_mean_reward=best_mean_reward,
                action_spec=agent.action_spec_dict,
            )

        elapsed_min = (time() - start_time) / 60.0
        print(f"Total elapsed time: {elapsed_min:.2f} minutes")

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

    game = create_simple_game(visible=False, bots=BOTS, map_name="map01", async_player=False)

    mapper = FactorizedActionMapper(game)
    spec = mapper.action_spec()
    print("FACTOR_HEADS:", spec.head_names)
    print("HEAD_SIZES   :", spec.head_sizes)

    agent = PPOAgent(action_mapper=mapper)

    # Resume training (full) or load weights (inference only)
    start_epoch = 0
    start_global_step = 0
    best_mean_reward = float("-inf")

    if load_checkpoint and os.path.exists(checkpoint_savefile):
        print("Loading checkpoint from:", checkpoint_savefile)
        start_epoch, start_global_step, best_mean_reward = load_full_checkpoint(
            checkpoint_savefile,
            agent,
            expected_action_spec=agent.action_spec_dict,
        )
        print(f"Resuming from epoch={start_epoch}, global_step={start_global_step}, best_mean_reward={best_mean_reward:.2f}")
    elif load_model_weights and os.path.exists(model_savefile):
        print("Loading model weights from:", model_savefile)
        agent.network.load_state_dict(torch.load(model_savefile, map_location=DEVICE))
        agent.set_eval_mode()

    if not skip_learning:
        best_mean_reward = train(
            game,
            agent,
            mapper,
            start_epoch=start_epoch,
            start_global_step=start_global_step,
            best_mean_reward=best_mean_reward,
        )
        print("\n" + "=" * 60)
        print("Training finished.")
        print("=" * 60)

    game.close()
