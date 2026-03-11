#!/usr/bin/env python3

import numpy as np
import skimage.transform
import vizdoom as vzd

# ---------- CONSTANTS ----------
SCENARIO_PATH = "../scenarios/"

# Q-learning settings
LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
TRAIN_EPOCHS = 10
LEARNING_STEPS_PER_EPOCH = 2000
REPLAY_MEMORY_SIZE = 10000

# NN learning settings
BATCH_SIZE = 128

# Training regime
TEST_EPISODES_PER_EPOCH = 100

# Other parameters
FRAME_REPEAT = 12
RESOLUTION = (96, 128)
EPISODES_TO_WATCH = 10

# ---------- HELPER FUNCTIONS ----------
def preprocess_vars(v: np.ndarray, num_vars: int) -> np.ndarray:
    """
    v: game_state.game_variables (shape: [num_vars])
    returns float32 vector shape (num_vars,)
    """
    v = np.asarray(v, dtype=np.float32)
    # safety sizing
    if v.shape[0] != num_vars:
        out = np.zeros((num_vars,), dtype=np.float32)
        out[: min(num_vars, v.shape[0])] = v[: min(num_vars, v.shape[0])]
        v = out

    ammo, health = v[0], v[1]

    health = np.clip(health, 0.0, 100.0) / 100.0
    ammo   = np.clip(ammo,   0.0, 50.0)  / 50.0

    return np.array([health, ammo], dtype=np.float32)

def preprocess_vars_health(v: np.ndarray, num_vars: int) -> np.ndarray:
    """
    v: game_state.game_variables (shape: [num_vars])
    returns float32 vector shape (num_vars,)
    """
    v = np.asarray(v, dtype=np.float32)
    # safety sizing
    if v.shape[0] != num_vars:
        out = np.zeros((num_vars,), dtype=np.float32)
        out[: min(num_vars, v.shape[0])] = v[: min(num_vars, v.shape[0])]
        v = out

    health = v[0] 

    health = np.clip(health, 0.0, 100.0) / 100.0

    return np.array([health], dtype=np.float32)

def preprocess(img, resolution):
    """Down samples image to resolution"""
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    return img

def preprocess_rgb(img, resolution=(96, 128)):
    # img expected HxWx3 (RGB24)
    x = skimage.transform.resize(
        img, resolution, anti_aliasing=True, preserve_range=True
    ).astype(np.float32)
    x = np.transpose(x, (2, 0, 1))  # -> C,H,W
    return x

def preprocess_rgb_normalized(img, res=RESOLUTION):
    x = preprocess_rgb(img, res)
    x /= 255.0
    return x

def get_num_game_variables(scenario_path: str):
    game = vzd.DoomGame()
    game.load_config(scenario_path)
    num_game_variables = game.get_available_game_variables_size()
    game.close()
    return num_game_variables

def infer_expected_num_vars(agent, game: vzd.DoomGame) -> int:
    """
    Prefer the model's declared num_vars if present (avoids LayerNorm shape mismatches),
    otherwise fall back to demo.py behavior.
    """
    qn = getattr(agent, "q_net", None)
    if qn is not None and hasattr(qn, "num_vars"):
        try:
            return int(qn.num_vars)
        except Exception:
            pass
    return len(game.get_available_game_variables())

class RunningMeanStd:
    """
    Lightweight running mean / variance tracker (Welford-style updates).

    Useful for normalizing large / heterogeneous game-variable vectors (e.g., deathmatch).
    Store / load via state_dict() to make evaluation match training.
    """

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
        return {
            "mean": self.mean,
            "var": self.var,
            "count": float(self.count),
        }

    def load_state_dict(self, sd: dict):
        self.mean = np.asarray(sd.get("mean", self.mean), dtype=np.float32)
        self.var = np.asarray(sd.get("var", self.var), dtype=np.float32)
        self.count = float(sd.get("count", self.count))


def preprocess_vars_general(
    raw_vars: np.ndarray,
    expected: int,
    *,
    normalizer: RunningMeanStd | None = None,
    update: bool = False,
    clip: float = 5.0,
) -> np.ndarray:
    """
    Generic, scenario-agnostic variable preprocessing.

    - Pads / truncates to `expected`.
    - If `normalizer` is provided, uses running-mean-std normalization (recommended).
    - Otherwise uses a bounded log-tanh style transform to keep magnitudes sane.

    This is meant for scenarios like deathmatch where the variable vector is longer and
    different variables have very different scales.
    """
    v = np.asarray(raw_vars, dtype=np.float32).reshape(-1)
    out = np.zeros((expected,), dtype=np.float32)
    if expected <= 0:
        return out

    n = min(expected, v.size)
    out[:n] = v[:n]

    if normalizer is not None:
        if update:
            normalizer.update(out)
        return normalizer.normalize(out, clip=clip)

    # Stateless fallback: signed log1p scaling -> roughly [-1, 1] for |x| up to ~1000
    denom = np.log1p(1000.0).astype(np.float32)
    y = np.sign(out) * (np.log1p(np.abs(out)) / denom)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def preprocess_vars_safe(
    raw_vars: np.ndarray,
    expected: int,
    *,
    normalizer: RunningMeanStd | None = None,
    update: bool = False,
    clip: float = 5.0,
) -> np.ndarray:
    """
    Backwards-compatible vars preprocessor.

    - For defend_the_center.cfg (expected <= 2): preserves legacy behavior:
        output = [health/100, ammo/50, 0, 0, ...]
      where the raw vars are {AMMO2, HEALTH}.

    - For larger expected vectors (e.g., deathmatch): uses preprocess_vars_general().
    """
    if expected <= 0:
        return np.zeros((0,), dtype=np.float32)

    # Legacy mapping used across existing codepaths for defend_the_center
    if expected <= 2:
        v = np.asarray(raw_vars, dtype=np.float32).reshape(-1)
        out = np.zeros((expected,), dtype=np.float32)

        ammo = float(v[0]) if v.size >= 1 else 0.0
        health = float(v[1]) if v.size >= 2 else 0.0

        out[0] = np.clip(health, 0.0, 100.0) / 100.0
        if expected >= 2:
            out[1] = np.clip(ammo, 0.0, 50.0) / 50.0
        return out

    # Generic path for scenarios with many vars
    return preprocess_vars_general(
        raw_vars,
        expected,
        normalizer=normalizer,
        update=update,
        clip=clip,
    )

def print_config(
    device,
    learning_rate, 
    discount_factor,
    train_epochs,
    learning_steps_per_epoch,
    test_episodes_per_epoch,
    replay_memory_size,
    batch_size,
    frame_repeat,
    resolution,
    episodes_to_watch):
    
    print("----------MODEL CONFIGURATION----------")
    print("DEVICE:", device)
    print("Learning Rate:", learning_rate)
    print("Discount Factor:", discount_factor)
    print("Train Epochs:", train_epochs)
    print("Learning Steps per Epoch:", learning_steps_per_epoch)
    print("Test Episodes per Epoch:", test_episodes_per_epoch)
    print("Replay Memory Size:", replay_memory_size)
    print("Batch Size:", batch_size)
    print("Frame Repeat:", frame_repeat)
    print("Resolution:", resolution)
    print("Episodes to Watch:", episodes_to_watch)