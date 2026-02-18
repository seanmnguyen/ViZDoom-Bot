#!/usr/bin/env python3
"""
Fast evaluator for ViZDoom agents (headless by default).

Keeps the CLI args + model mappings from demo.py, but runs much faster when
--show False by:
  - using PLAYER mode (no rendering)
  - stepping with game.make_action(action, frame_repeat) (no advance_action loop, no sleeps)

Example:
  python eval_models.py -mt q_rainbow_rgb -mp ../models/q_rainbow_rgb.pth -s False -sc defend_the_center.cfg -n 30
"""

import argparse
import itertools as it
import os
from pathlib import Path
from time import sleep

import numpy as np
import torch
import vizdoom as vzd

from utils import *

# --- AGENT IMPORTS (same as demo.py) ---
from q_late_fusion import DQNAgent as DQNAgent_LateFusion
from q_late_fusion_rgb import DQNAgent as DQNAgent_LateFusionRGB
from q_cnn import DQNAgent as DQNAgent_CNN
from q_cnn_rgb import DQNAgent as DQNAgent_CNNRGB
from q_rainbow_rgb import DQNAgent as DQNAgent_RainbowRGB
from ppo_cnn import PPOAgent
from ppo_cnn_gray import PPOAgent as PPOAgent_Gray
from ppo_cnn_gray import FrameStack, FRAME_STACK_SIZE

# ---------- GLOBALS (same as demo.py; only used to construct agents) ----------
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
batch_size = 64
frame_repeat = 12

# ---------- MODEL MAPPINGS (copied from demo.py) ----------
MODEL_DEFAULT_SCENARIO = {
    "q_cnn": "defend_the_line.cfg",
    "q_cnn_rgb": "defend_the_line.cfg",
    "q_late_fusion": "defend_the_center.cfg",
    "q_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_cnn": "defend_the_line.cfg",
    "ppo_cnn_gray": "defend_the_center.cfg",
    "q_late_fusion_rgb_DC": "deadly_corridor.cfg",
    "q_rainbow_rgb": "defend_the_center.cfg",
}

AGENT_BY_MODEL = {
    "q_cnn": DQNAgent_CNN,
    "q_cnn_rgb": DQNAgent_CNNRGB,
    "q_late_fusion": DQNAgent_LateFusion,
    "q_late_fusion_rgb": DQNAgent_LateFusionRGB,
    "ppo_cnn": PPOAgent,
    "ppo_cnn_gray": PPOAgent_Gray,
    "q_late_fusion_rgb_DC": DQNAgent_LateFusionRGB,
    "q_rainbow_rgb": DQNAgent_RainbowRGB,
}

RESOLUTION_BY_MODEL = {
    "q_cnn": (30, 45),
    "q_cnn_rgb": (96, 128),
    "q_late_fusion": (96, 128),
    "q_late_fusion_rgb": (96, 128),
    "ppo_cnn": (30, 45),
    "ppo_cnn_gray": "defend_the_center.cfg",
    "q_late_fusion_rgb_DC": (96, 128),
    "q_rainbow_rgb": (96, 128),
}

GRAYSCALE = "GRAY8"
RGB = "RGB24"
COLOR_BY_MODEL = {
    "q_cnn": GRAYSCALE,
    "q_cnn_rgb": RGB,
    "q_late_fusion": GRAYSCALE,
    "q_late_fusion_rgb": RGB,
    "ppo_cnn": GRAYSCALE,
    "q_late_fusion_rgb_DC": RGB,
    "q_rainbow_rgb": RGB,
}

PPO_MODELS = {"ppo_cnn"}


# ---------- CLI PARSER (demo.py-compatible) ----------
def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'. Use True/False.")


def parse_cli():
    parser = argparse.ArgumentParser(description="Fast evaluate ViZDoom agents (no rendering by default).")

    parser.add_argument(
        "-mt", "--model_type",
        choices=list(AGENT_BY_MODEL.keys()),
        default="q_cnn",
        help="Model type."
    )

    parser.add_argument(
        "-mp", "--model_path",
        type=str,
        default=None,
        help="Path to model weights for loading. Defaults to ../models/<model_type>.pth"
    )

    parser.add_argument(
        "-s", "--show",
        type=str2bool,
        default=False,   # changed vs demo.py for speed; still accepts True/False
        metavar="BOOL",
        help="Show game window (True/False)."
    )

    parser.add_argument(
        "-sc", "--scenario",
        type=str,
        default=None,
        help="Scenario config file (e.g., defend_the_line.cfg). Defaults to training scenario for model type."
    )

    parser.add_argument(
        "-n", "--episodes",
        type=int,
        default=30,
        help="Number of evaluation episodes (default: 30)."
    )

    args = parser.parse_args()

    agent_builder = AGENT_BY_MODEL[args.model_type]

    default_path = Path("../models") / f"{args.model_type}.pth"
    model_path = Path(args.model_path) if args.model_path else default_path

    return args, agent_builder, model_path


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


@torch.no_grad()
def evaluate(game: vzd.DoomGame, agent, actions, *, model_type: str, resolution, episodes: int, visible_window: bool):
    # Set eval mode if supported
    if hasattr(agent, "set_eval_mode"):
        agent.set_eval_mode()
    if hasattr(agent, "eval"):
        try:
            agent.eval()
        except Exception:
            pass

    scores = []
    expected_num_vars = infer_expected_num_vars(agent, game)

    for ep in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            state_img = preprocess_fn(gs.screen_buffer, resolution)

            if model_type in PPO_MODELS:
                a = agent.get_action(state_img, deterministic=True)
            else:
                state_vars = preprocess_vars(gs.game_variables, expected_num_vars)
                # Prefer eval_mode=True if the agent supports it
                try:
                    a = agent.get_action(state_img, state_vars, eval_mode=True)
                except TypeError:
                    a = agent.get_action(state_img, state_vars)

            if visible_window:
                game.set_action(actions[a])
                for _ in range(frame_repeat):
                    game.advance_action()
            else:
                game.make_action(actions[a], frame_repeat)

        score = game.get_total_reward()
        scores.append(score)

        if visible_window:
            print(f"Episode {ep + 1} Total Score: {score}")
            sleep(0.2)
        elif ep % 10 == 0:
            print(f"Episode {ep + 1} Total Score: {score}")

    return np.asarray(scores, dtype=np.float32)


if __name__ == "__main__":
    # Uses GPU if available (same as demo.py)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    args, AgentBuilder, model_path = parse_cli()

    visible_window = args.show
    scenario_file = args.scenario if args.scenario else MODEL_DEFAULT_SCENARIO.get(
        args.model_type, "defend_the_center.cfg"
    )
    config_file_path = os.path.join(SCENARIO_PATH, scenario_file)

    model_loadfile = str(model_path)

    print("model_type:", args.model_type)
    print("load path :", model_loadfile)
    print("scenario  :", scenario_file)
    print("show      :", visible_window)
    print("episodes  :", args.episodes)

    # Initialize game
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if visible_window else vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Match demo.py's screen format selection, but without forcing HUD rendering in headless
    if COLOR_BY_MODEL[args.model_type] == RGB:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess_fn = preprocess_rgb
    elif COLOR_BY_MODEL[args.model_type] == GRAYSCALE:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        preprocess_fn = preprocess
    else:
        raise ValueError(f"Invalid color format for model type {args.model_type}")

    if hasattr(game, "set_render_hud"):
        game.set_render_hud(True)

    resolution = RESOLUTION_BY_MODEL[args.model_type]
    game.init()

    # Build action space
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Build agent (same constructor conventions as demo.py)
    if args.model_type in PPO_MODELS:
        agent = AgentBuilder(action_size=len(actions), load_model_path=model_path)
    else:
        agent = AgentBuilder(
            len(actions),
            lr=learning_rate,
            batch_size=batch_size,
            memory_size=replay_memory_size,
            discount_factor=discount_factor,
            load_model=True,
            model_weights=model_path,
        )

    # Evaluate
    scores = evaluate(
        game,
        agent,
        actions,
        model_type=args.model_type,
        resolution=resolution,
        episodes=args.episodes,
        visible_window=visible_window,
    )

    print("======================================")
    print("Score: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
        float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
    ))

    game.close()
