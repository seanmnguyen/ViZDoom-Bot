#!/usr/bin/env python3

import numpy as np
import skimage.transform

SCENARIO_PATH = "../scenarios/"

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

    health, ammo = v[0], v[1]

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