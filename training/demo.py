"""demo.py

Run a trained model in ViZDoom with an optional visible window.
"""

from __future__ import annotations

import argparse
import itertools as it
import os
from pathlib import Path
from time import sleep
from typing import Any, Dict

import numpy as np
import torch
import vizdoom as vzd

from utils import *
import model_registry as MODELS

# -----------------------------------------------------------------------------
# Globals (only used to construct agents)
# -----------------------------------------------------------------------------
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
batch_size = 64
frame_repeat = 12
EPISODES_TO_WATCH = 5

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def str2bool(v):
    """Parse bools from CLI strings."""
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: '{v}'. Use True/False.")


def parse_cli():
    parser = argparse.ArgumentParser(description="Run a trained ViZDoom agent with rendering (optional).")

    available_models = [k for k, v in MODELS.AGENT_BY_MODEL.items() if v is not None]
    if not available_models:
        raise RuntimeError("No models available to run (all agent imports failed).")

    parser.add_argument(
        "-mt", "--model_type",
        choices=available_models,
        default=available_models[0],
        help="Model type."
    )

    parser.add_argument(
        "-mp", "--model_path",
        type=str,
        default=None,
        help="Path to model weights for loading. Defaults to ../models/<scenario>/<model_type>.pth (if present)."
    )

    parser.add_argument(
        "-s", "--show",
        type=str2bool,
        default=True,
        metavar="BOOL",
        help="Show game window (True/False)."
    )

    parser.add_argument(
        "-sc", "--scenario",
        type=str,
        default=None,
        help="Scenario config file (e.g., defend_the_center.cfg). Defaults to training scenario for model type."
    )

    args = parser.parse_args()

    agent_builder = MODELS.AGENT_BY_MODEL[args.model_type]

    # Default path resolution:
    # 1) prefer ../models/<scenario_stem>/<model_type>.pth (newer layout)
    # 2) fallback to ../models/<model_type>.pth (older layout)
    default_path = Path("../models") / f"{args.model_type}.pth"
    scenario_default = MODELS.MODEL_DEFAULT_SCENARIO.get(args.model_type)
    if scenario_default:
        scen_stem = Path(scenario_default).stem
        alt = Path("../models") / scen_stem / f"{args.model_type}.pth"
        if alt.exists():
            default_path = alt

    model_path = Path(args.model_path) if args.model_path else default_path
    return args, agent_builder, model_path


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _infer_expected_num_vars(agent, game: vzd.DoomGame) -> int:
    """Prefer the model's declared num_vars if present."""
    qn = getattr(agent, "q_net", None)
    if qn is not None and hasattr(qn, "num_vars"):
        try:
            return int(qn.num_vars)
        except Exception:
            pass
    return len(game.get_available_game_variables())


def _load_weights_state_dict(path: Path, device: torch.device) -> Dict[str, Any]:
    """Load a pure state_dict saved by torch.save(model.state_dict())."""
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location=device)

    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unexpected weights file format at {path}")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args, AgentBuilder, model_path = parse_cli()

    model_loadfile = str(model_path)
    visible_window = args.show

    scenario_file = args.scenario if args.scenario else MODELS.MODEL_DEFAULT_SCENARIO.get(
        args.model_type, "defend_the_center.cfg"
    )
    config_file_path = os.path.join(SCENARIO_PATH, scenario_file)

    print("model_type:", args.model_type)
    print("load path :", model_loadfile)
    print("scenario  :", scenario_file)
    print("show      :", visible_window)

    # Initialize game
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(True)

    # Screen format selection
    color_mode = MODELS.COLOR_BY_MODEL[args.model_type]
    lazy_mod = MODELS.LAZY_STACK_MODULE_BY_MODEL.get(args.model_type)
    preprocess_fn = None

    # Keep frame_repeat consistent with the model file when available.
    frame_repeat_effective = int(getattr(lazy_mod, "FRAME_REPEAT", frame_repeat)) if lazy_mod is not None else frame_repeat

    if color_mode == MODELS.RGB:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess_fn = preprocess_rgb
    elif color_mode == MODELS.GRAYSCALE:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        preprocess_fn = preprocess
    elif color_mode == MODELS.AUTO and lazy_mod is not None:
        use_gray = bool(getattr(lazy_mod, "USE_GRAYSCALE", False))
        game.set_screen_format(vzd.ScreenFormat.GRAY8 if use_gray else vzd.ScreenFormat.RGB24)
        preprocess_fn = None  # lazy models use lazy_mod.preprocess_frame_u8
    else:
        raise ValueError(f"Invalid or unsupported color format for model type {args.model_type}")

    resolution = MODELS.RESOLUTION_BY_MODEL[args.model_type]
    game.init()

    # Build action space
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Build agent
    if args.model_type in MODELS.PPO_MODELS:
        agent = AgentBuilder(action_size=len(actions), load_model_path=model_path)
    elif args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL and lazy_mod is not None:
        agent = AgentBuilder(
            action_size=len(actions),
            lr=learning_rate,
            discount_factor=discount_factor,
            memory_size=replay_memory_size,
            batch_size=batch_size,
        )
        sd = _load_weights_state_dict(model_path, DEVICE)
        agent.q_net.load_state_dict(sd)
        if hasattr(agent, "target_net"):
            try:
                agent.target_net.load_state_dict(sd)
            except Exception:
                pass
        if hasattr(agent, "set_eval_mode"):
            agent.set_eval_mode()
        else:
            agent.q_net.eval()
    else:
        # Legacy DQN-based agents
        agent = AgentBuilder(
            len(actions),
            lr=learning_rate,
            batch_size=batch_size,
            memory_size=replay_memory_size,
            discount_factor=discount_factor,
            load_model=True,
            model_weights=model_path,
        )

    # Set up frame stacking if needed
    use_frame_stack = args.model_type in MODELS.FRAME_STACK_MODELS
    frame_stack = None
    if use_frame_stack:
        if args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL and lazy_mod is not None:
            frame_stack = lazy_mod.FrameStack(lazy_mod.FRAME_STACK_SIZE, lazy_mod.FRAME_C, resolution)
            frame_stack._inited = False
        elif color_mode == MODELS.GRAYSCALE:
            if MODELS.PPOFrameStackGray is None:
                raise RuntimeError("PPOFrameStackGray import failed, but frame stacking was requested.")
            frame_stack = MODELS.PPOFrameStackGray(MODELS.PPO_FRAME_STACK_SIZE_GRAY, resolution)
        elif color_mode == MODELS.RGB:
            if MODELS.PPOFrameStackRGB is None:
                raise RuntimeError("PPOFrameStackRGB import failed, but frame stacking was requested.")
            frame_stack = MODELS.PPOFrameStackRGB(MODELS.PPO_FRAME_STACK_SIZE_RGB, resolution)
        else:
            raise ValueError(f"Unsupported color mode {color_mode} for frame stacking.")

    # Play episodes
    total_score = 0.0
    for episode_num in range(EPISODES_TO_WATCH):
        game.new_episode()

        if use_frame_stack and frame_stack is not None:
            if args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL and lazy_mod is not None:
                frame_stack._inited = False
                frame_stack.frames.clear()
            else:
                frame_stack.reset()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            expected_num_vars = _infer_expected_num_vars(agent, game)
            state_vars = preprocess_vars_safe(gs.game_variables, expected_num_vars)

            if args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL and lazy_mod is not None:
                # Lazy-stack Rainbow expects uint8 stacked state images.
                frame_u8 = lazy_mod.preprocess_frame_u8(gs.screen_buffer)
                if use_frame_stack and frame_stack is not None:
                    if not getattr(frame_stack, "_inited", False):
                        frame_stack.reset(frame_u8)
                        frame_stack._inited = True
                    else:
                        frame_stack.append(frame_u8)
                    state_img = frame_stack.get()  # uint8 (C*K,H,W)
                else:
                    state_img = frame_u8
            else:
                assert preprocess_fn is not None
                state_img = preprocess_fn(gs.screen_buffer, resolution)
                if use_frame_stack and frame_stack is not None:
                    frame_stack.push(state_img)
                    state_img = frame_stack.get()

            # Action selection
            if args.model_type in MODELS.PPO_MODELS:
                if args.model_type in MODELS.LATE_FUSION_PPO_MODELS:
                    state_vars = preprocess_vars_safe(gs.game_variables, expected_num_vars)
                    a = agent.get_action(state_img, state_vars, deterministic=True)
                else:
                    a = agent.get_action(state_img, deterministic=True)
            elif args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL and lazy_mod is not None:
                a = agent.get_action(state_img, state_vars, eval_mode=True)
            else:
                a = agent.get_action(state_img, state_vars)

            game.set_action(actions[int(a)])
            for _ in range(frame_repeat_effective):
                game.advance_action()

        if visible_window:
            sleep(1.0)

        score = float(game.get_total_reward())
        total_score += score
        print(f"Episode {episode_num + 1} Total Score: {score}")

    print(f"-----Average Score: {total_score / EPISODES_TO_WATCH:.2f}-----")
    game.close()
