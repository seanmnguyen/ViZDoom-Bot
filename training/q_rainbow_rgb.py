#!/usr/bin/env python3
"""
Rainbow (no recurrence) + Late-Fusion (RGB) for ViZDoom.

Implements (Rainbow core, minus recurrence):
- Double DQN
- Dueling network
- Distributional RL (C51)
- Noisy Nets exploration (no epsilon-greedy)
- Prioritized Experience Replay (PER) with SumTree
- N-step returns
"""

import itertools as it
import os
import random
from dataclasses import dataclass
from time import sleep, time
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from utils import *  # use shared constants + preprocess functions

# -----------------------------------------------------------------------------
# Scenario / save naming (README convention)
# -----------------------------------------------------------------------------
SCENARIO_NAME = "defend_the_center"
config_file_path = os.path.join(SCENARIO_PATH, f"{SCENARIO_NAME}.cfg")

MODEL_TYPE = os.path.splitext(os.path.basename(__file__))[0]
model_savefile = f"../models/{SCENARIO_NAME}/{MODEL_TYPE}.pth"
os.makedirs(os.path.dirname(model_savefile), exist_ok=True)

save_model = True
load_model = False
skip_learning = False

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

N_STEP = 3

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_EPS = 1e-6

TARGET_UPDATE_EVERY = 1000  # optimizer steps (not env steps)
GRAD_CLIP_NORM = 10.0

# start learning after some experience
LEARNING_STARTS = 2 * BATCH_SIZE
TRAIN_EVERY = 1
UPDATES_PER_TRAIN = 1

NUM_VARS = NUM_VARS = get_num_game_variables(config_file_path)

# TODO: ADJUST SCALING
BATCH_SIZE = 128
RESOLUTION = (96, 128)
TRAIN_EVERY = 1
UPDATES_PER_TRAIN = 1
TRAIN_EPOCHS = 75
LEARNING_RATE = 0.00025

# -----------------------------------------------------------------------------
# Environment helpers
# -----------------------------------------------------------------------------
def create_simple_game(visible: bool = False, async_player: bool = False):
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if async_player else vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    for gv in game.get_available_game_variables():
        print("GAME VAR:", gv)
        game.add_available_game_variable(gv)

    game.init()
    print("Doom initialized.")
    return game


def zeros_img_rgb() -> np.ndarray:
    return np.zeros((3, RESOLUTION[0], RESOLUTION[1]), dtype=np.float32)


def zeros_vars() -> np.ndarray:
    return np.zeros((NUM_VARS,), dtype=np.float32)


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
# CNN backbone (same spirit as your StrongCNN, with GroupNorm default)
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
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B,C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
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
    def __init__(self, in_channels: int = 3):
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
        return torch.flatten(x, 1)  # (B, 128*4*4=2048)


# -----------------------------------------------------------------------------
# Rainbow late-fusion dueling C51 network (no recurrence)
# -----------------------------------------------------------------------------
class RainbowLateFusionC51(nn.Module):
    def __init__(
        self,
        action_size: int,
        num_vars: int,
        atoms: int,
        vmin: float,
        vmax: float,
        in_channels: int = 3,
        img_h: int = RESOLUTION[0],
        img_w: int = RESOLUTION[1],
    ):
        super().__init__()
        self.action_size = action_size
        self.num_vars = num_vars
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
            nn.Dropout(p=0.10),
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

        # Dueling distributional heads (Noisy)
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
        """
        Returns logits: (B, A, atoms)
        """
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = self.fuse(torch.cat([img_feat, vars_feat], dim=1))  # (B,256)

        v = self.val_fc2(F.relu(self.val_fc1(fused), inplace=True)).view(-1, 1, self.atoms)          # (B,1,atoms)
        a = self.adv_fc2(F.relu(self.adv_fc1(fused), inplace=True)).view(-1, self.action_size, self.atoms)  # (B,A,atoms)
        logits = v + (a - a.mean(dim=1, keepdim=True))
        return logits

    def probs(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(img, vars_), dim=-1)  # (B,A,atoms)

    def q_values(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        p = self.probs(img, vars_)
        return (p * self.support.view(1, 1, -1)).sum(dim=-1)  # (B,A)


# -----------------------------------------------------------------------------
# Replay: PER SumTree + N-step accumulator
# -----------------------------------------------------------------------------
@dataclass
class Transition:
    img: np.ndarray
    vars_: np.ndarray
    action: int
    reward: float
    next_img: np.ndarray
    next_vars: np.ndarray
    done: bool


class NStepAccumulator:
    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buf: List[Transition] = []

    def reset(self):
        self.buf.clear()

    def _pop_one(self) -> Transition:
        first = self.buf[0]

        R = 0.0
        next_img = first.next_img
        next_vars = first.next_vars
        done_n = first.done

        last = min(self.n_step, len(self.buf))
        for i in range(last):
            tr = self.buf[i]
            R += (self.gamma ** i) * tr.reward
            next_img = tr.next_img
            next_vars = tr.next_vars
            done_n = tr.done
            if tr.done:
                break

        out = Transition(
            img=first.img,
            vars_=first.vars_,
            action=first.action,
            reward=float(R),
            next_img=next_img,
            next_vars=next_vars,
            done=bool(done_n),
        )
        self.buf.pop(0)
        return out

    def add(self, tr: Transition) -> List[Transition]:
        emitted = []
        self.buf.append(tr)

        if len(self.buf) >= self.n_step:
            emitted.append(self._pop_one())

        if tr.done:
            while len(self.buf) > 0:
                emitted.append(self._pop_one())

        return emitted


class SumTree:
    def __init__(self, capacity: int):
        assert capacity > 0 and (capacity & (capacity - 1) == 0) or True  # power-of-2 not required
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
    
    def get(self, s: float) -> Tuple[int, float, object]:
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break

            left_sum = float(self.tree[left])

            # IMPORTANT:
            # - if left_sum == 0, we must go right (otherwise s==0 walks into all-zero left subtree)
            # - use strict < (not <=) to avoid boundary selecting zero-mass left branch
            if left_sum > 0.0 and s < left_sum:
                idx = left
            else:
                s -= left_sum
                idx = right

        data_idx = idx - (self.capacity - 1)
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.alpha = alpha
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition: Transition, priority: Optional[float] = None):
        p = self.max_priority if priority is None else float(priority)
        p = max(p, PER_EPS)
        self.max_priority = max(self.max_priority, p)
        self.tree.add(p ** self.alpha, transition)

    def sample(self, batch_size: int, beta: float):
        batch = []
        idxs = np.empty((batch_size,), dtype=np.int64)
        priorities = np.empty((batch_size,), dtype=np.float32)

        total = self.tree.total()
        if not np.isfinite(total) or total <= 0.0:
            raise RuntimeError(f"PER: invalid total priority sum: {total}")

        segment = total / batch_size
        eps = np.nextafter(0.0, 1.0)  # smallest positive float > 0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            data = None
            p = 0.0
            idx = 0

            # Try a few times to avoid empty leaves / zero-priority leaves
            for _ in range(8):
                s = a + (b - a) * random.random()
                s = max(s, eps)
                s = min(s, np.nextafter(total, 0.0))  # keep strictly < total

                idx, p, data = self.tree.get(s)
                if data is not None and np.isfinite(p) and p > 0.0:
                    break

            # Hard fallback: sample anywhere in the tree range
            if data is None or (not np.isfinite(p)) or p <= 0.0:
                for _ in range(32):
                    s = max(random.random() * total, eps)
                    s = min(s, np.nextafter(total, 0.0))
                    idx, p, data = self.tree.get(s)
                    if data is not None and np.isfinite(p) and p > 0.0:
                        break

            if data is None:
                raise RuntimeError("PER: sampled empty transition (data=None). SumTree contains empty leaves in sampled mass.")

            idxs[i] = idx
            priorities[i] = p
            batch.append(data)

        # IS weights (numerically safe)
        probs = priorities / (total + 1e-12)
        probs = np.clip(probs, 1e-12, 1.0)

        weights = (self.tree.n_entries * probs) ** (-beta)
        weights /= (weights.max() + 1e-8)

        return batch, idxs, weights.astype(np.float32), probs.astype(np.float32)

    def update_priorities(self, idxs: np.ndarray, new_priorities: np.ndarray):
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        for idx, p in zip(idxs, new_priorities):
            p = float(max(float(p), PER_EPS))
            self.max_priority = max(self.max_priority, p)
            self.tree.update(int(idx), (p ** self.alpha))


# -----------------------------------------------------------------------------
# Agent (Rainbow without recurrence)
# -----------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, action_size: int, lr: float, discount_factor: float, memory_size: int, load_model: bool=False, model_weights: str="", batch_size: int=128):
        self.action_size = action_size
        self.gamma = discount_factor
        self.n_step = N_STEP
        self.gamma_n = self.gamma ** self.n_step

        self.q_net = RainbowLateFusionC51(
            action_size=action_size,
            num_vars=NUM_VARS,
            atoms=ATOMS,
            vmin=V_MIN,
            vmax=V_MAX,
            in_channels=3,
            img_h=RESOLUTION[0],
            img_w=RESOLUTION[1],
        ).to(DEVICE)

        self.target_net = RainbowLateFusionC51(
            action_size=action_size,
            num_vars=NUM_VARS,
            atoms=ATOMS,
            vmin=V_MIN,
            vmax=V_MAX,
            in_channels=3,
            img_h=RESOLUTION[0],
            img_w=RESOLUTION[1],
        ).to(DEVICE)

        if load_model:
            if model_weights is not None:  # weights inputted
                global model_savefile
                model_savefile = model_weights
            print("Loading Q-Rainbow model from:", model_savefile)
            sd = torch.load(model_savefile, map_location=DEVICE)  
            self.q_net.load_state_dict(sd)
            self.target_net.load_state_dict(sd)
            self.set_eval_mode()
        else:
            self.set_train_mode()

        self.opt = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay = PrioritizedReplayBuffer(capacity=memory_size, alpha=PER_ALPHA)
        self.nstep_acc = NStepAccumulator(n_step=self.n_step, gamma=self.gamma)

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

    def begin_episode(self):
        self.nstep_acc.reset()

    def end_episode(self):
        self.nstep_acc.reset()

    def update_target(self, hard: bool = True, tau: float = 1.0):
        if hard:
            self.target_net.load_state_dict(self.q_net.state_dict())
        else:
            with torch.no_grad():
                for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                    tp.data.mul_(1.0 - tau).add_(tau * p.data)

    @staticmethod
    def _project_distribution(next_prob, rewards, dones, gamma_n, support, vmin, vmax):
        """
        next_prob: (B, atoms)
        rewards:   (B,)
        dones:     (B,)
        returns projected distribution m: (B, atoms)
        """
        B, atoms = next_prob.shape
        delta_z = (vmax - vmin) / (atoms - 1)

        support = support.view(1, atoms)  # (1,atoms)
        tz = rewards.unsqueeze(-1) + (1.0 - dones.unsqueeze(-1)) * gamma_n * support
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

    def get_action(self, state_img: np.ndarray, state_vars: np.ndarray, eval_mode: bool = False) -> int:
        img_t = torch.from_numpy(np.expand_dims(state_img, axis=0)).float().to(DEVICE)   # (1,C,H,W)
        vars_t = torch.from_numpy(np.expand_dims(state_vars, axis=0)).float().to(DEVICE) # (1,V)

        if eval_mode:
            self.q_net.eval()
        else:
            self.q_net.train()
            self.q_net.reset_noise()

        with torch.no_grad():
            q = self.q_net.q_values(img_t, vars_t)  # (1,A)
            return int(torch.argmax(q, dim=1).item())

    def store_step(self, img, vars_, action, reward, next_img, next_vars, done):
        one = Transition(
            img=img, vars_=vars_, action=int(action), reward=float(reward),
            next_img=next_img, next_vars=next_vars, done=bool(done)
        )
        emitted = self.nstep_acc.add(one)
        for tr in emitted:
            self.replay.add(tr)

    def train_step(self) -> Optional[float]:
        if len(self.replay) < max(BATCH_SIZE, LEARNING_STARTS):
            return None

        self.set_train_mode()

        # anneal beta over total training horizon (roughly)
        # using optimizer steps as the schedule variable
        total_steps = max(1, TRAIN_EPOCHS * LEARNING_STEPS_PER_EPOCH // max(1, TRAIN_EVERY) * max(1, UPDATES_PER_TRAIN))
        frac = min(1.0, self.learn_step / total_steps)
        self.beta = PER_BETA_START + frac * (PER_BETA_END - PER_BETA_START)

        batch, idxs, is_w, _ = self.replay.sample(BATCH_SIZE, beta=self.beta)

        imgs      = np.stack([b.img for b in batch]).astype(np.float32)        # (B,C,H,W)
        vars_     = np.stack([b.vars_ for b in batch]).astype(np.float32)      # (B,V)
        actions   = np.array([b.action for b in batch], dtype=np.int64)        # (B,)
        rewards   = np.array([b.reward for b in batch], dtype=np.float32)      # (B,)
        next_imgs = np.stack([b.next_img for b in batch]).astype(np.float32)   # (B,C,H,W)
        next_vars = np.stack([b.next_vars for b in batch]).astype(np.float32)  # (B,V)
        dones     = np.array([b.done for b in batch], dtype=np.float32)        # (B,)

        imgs_t      = torch.from_numpy(imgs).to(DEVICE)
        vars_t      = torch.from_numpy(vars_).to(DEVICE)
        actions_t   = torch.from_numpy(actions).to(DEVICE)
        rewards_t   = torch.from_numpy(rewards).to(DEVICE)
        next_imgs_t = torch.from_numpy(next_imgs).to(DEVICE)
        next_vars_t = torch.from_numpy(next_vars).to(DEVICE)
        dones_t     = torch.from_numpy(dones).to(DEVICE)
        is_w_t      = torch.from_numpy(is_w).to(DEVICE)

        self.opt.zero_grad(set_to_none=True)

        # reset noisy params for this update
        self.q_net.reset_noise()
        self.target_net.reset_noise()

        use_amp = self.use_amp
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

        with autocast_ctx:
            # Current logits for chosen actions
            logits = self.q_net(imgs_t, vars_t)  # (B,A,atoms)
            atoms = logits.size(-1)
            act_idx = actions_t.view(-1, 1, 1).expand(-1, 1, atoms)
            logits_a = logits.gather(1, act_idx).squeeze(1)  # (B,atoms)
            log_prob_a = F.log_softmax(logits_a, dim=-1)

            with torch.no_grad():
                # Double DQN: select a* using ONLINE expected Q at next state
                next_logits_online = self.q_net(next_imgs_t, next_vars_t)  # (B,A,atoms)
                next_prob_online = torch.softmax(next_logits_online, dim=-1)
                support = self.q_net.support.view(1, 1, -1)
                next_q_online = (next_prob_online * support).sum(dim=-1)  # (B,A)
                next_actions = next_q_online.argmax(dim=1)  # (B,)

                # Evaluate distribution with TARGET net at a*
                next_logits_target = self.target_net(next_imgs_t, next_vars_t)  # (B,A,atoms)
                next_prob_target = torch.softmax(next_logits_target, dim=-1)
                next_act_idx = next_actions.view(-1, 1, 1).expand(-1, 1, atoms)
                next_prob_a = next_prob_target.gather(1, next_act_idx).squeeze(1)  # (B,atoms)

                # Project C51 target distribution
                target_dist = self._project_distribution(
                    next_prob=next_prob_a,
                    rewards=rewards_t,
                    dones=dones_t,
                    gamma_n=self.gamma_n,
                    support=self.q_net.support,
                    vmin=V_MIN,
                    vmax=V_MAX,
                )  # (B,atoms)

            per_sample_loss = -(target_dist * log_prob_a).sum(dim=-1)  # (B,)
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

        # Update PER priorities from per-sample loss
        new_prios = (per_sample_loss.detach().abs().clamp_min(PER_EPS) + PER_EPS).cpu().numpy()
        self.replay.update_priorities(idxs, new_prios)

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

    test_scores = []
    for _ in trange(TEST_EPISODES_PER_EPOCH, leave=False):
        game.new_episode()
        agent.begin_episode()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break
            img = preprocess_rgb(gs.screen_buffer, RESOLUTION)
            vars_ = preprocess_vars(gs.game_variables, NUM_VARS)
            a = agent.get_action(img, vars_, eval_mode=True)
            game.make_action(actions[a], FRAME_REPEAT)

        test_scores.append(game.get_total_reward())

    test_scores = np.array(test_scores, dtype=np.float32)
    print(
        "Results: mean: {:.1f} +/- {:.1f}, min: {:.1f}, max: {:.1f}".format(
            test_scores.mean(), test_scores.std(), test_scores.min(), test_scores.max()
        )
    )


def run(game, agent: DQNAgent, actions):
    start_time = time()
    global_step = 0

    for epoch in range(TRAIN_EPOCHS):
        print(f"\nEpoch #{epoch + 1}")
        game.new_episode()
        agent.begin_episode()
        train_scores = []
        losses = []

        for _ in trange(LEARNING_STEPS_PER_EPOCH, leave=False):
            gs = game.get_state()
            if gs is None:
                game.new_episode()
                agent.begin_episode()
                continue

            img = preprocess_rgb(gs.screen_buffer, RESOLUTION)
            vars_ = preprocess_vars(gs.game_variables, NUM_VARS)

            a = agent.get_action(img, vars_, eval_mode=False)
            r = game.make_action(actions[a], FRAME_REPEAT)
            done = game.is_episode_finished()

            if not done:
                ngs = game.get_state()
                if ngs is None:
                    next_img, next_vars, done = zeros_img_rgb(), zeros_vars(), True
                else:
                    next_img = preprocess_rgb(ngs.screen_buffer, RESOLUTION)
                    next_vars = preprocess_vars(ngs.game_variables, NUM_VARS)
            else:
                next_img, next_vars = zeros_img_rgb(), zeros_vars()

            agent.store_step(img, vars_, a, r, next_img, next_vars, done)

            if (global_step % TRAIN_EVERY) == 0:
                for _ in range(UPDATES_PER_TRAIN):
                    l = agent.train_step()
                    if l is not None:
                        losses.append(l)

            if done:
                train_scores.append(game.get_total_reward())
                agent.end_episode()
                game.new_episode()
                agent.begin_episode()

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

        elapsed = (time() - start_time) / 60.0
        print(f"Total elapsed time: {elapsed:.2f} minutes")

    game.close()


def watch_trained(agent: DQNAgent, actions):
    game = create_simple_game(visible=True, async_player=True)
    agent.set_eval_mode()

    total = 0.0
    for ep in range(EPISODES_TO_WATCH):
        game.new_episode()
        agent.begin_episode()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break
            img = preprocess_rgb(gs.screen_buffer, RESOLUTION)
            vars_ = preprocess_vars(gs.game_variables, NUM_VARS)
            a = agent.get_action(img, vars_, eval_mode=True)

            game.set_action(actions[a])
            for _ in range(FRAME_REPEAT):
                game.advance_action()

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
    print("LEARNING_RATE:", LEARNING_RATE)
    print("DISCOUNT_FACTOR:", DISCOUNT_FACTOR)
    print("TRAIN_EPOCHS:", TRAIN_EPOCHS)
    print("LEARNING_STEPS_PER_EPOCH:", LEARNING_STEPS_PER_EPOCH)
    print("TEST_EPISODES_PER_EPOCH:", TEST_EPISODES_PER_EPOCH)
    print("REPLAY_MEMORY_SIZE:", REPLAY_MEMORY_SIZE)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("FRAME_REPEAT:", FRAME_REPEAT)
    print("RESOLUTION:", RESOLUTION)
    print("EPISODES_TO_WATCH:", EPISODES_TO_WATCH)
    print("MODEL_SAVEFILE:", model_savefile)
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

    if load_model and os.path.exists(model_savefile):
        print("Loading weights from:", model_savefile)
        sd = torch.load(model_savefile, map_location=DEVICE)
        agent.q_net.load_state_dict(sd)
        agent.update_target(hard=True)

    if not skip_learning:
        run(game, agent, actions)
        print("======================================")
        print("Training finished. Time to watch!")
    else:
        game.close()

    watch_trained(agent, actions)
