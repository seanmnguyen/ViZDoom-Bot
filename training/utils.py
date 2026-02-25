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

def get_num_game_variables(scenario_path: str):
    game = vzd.DoomGame()
    game.load_config(scenario_path)
    num_game_variables = game.get_available_game_variables_size()
    game.close()
    return num_game_variables

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