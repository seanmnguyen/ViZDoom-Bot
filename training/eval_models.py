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
import sys
from pathlib import Path
from time import sleep
from random import choice

import numpy as np
import torch
import vizdoom as vzd

from utils import *
import model_registry as MODELS

import q_rainbow_stacked as rainbow_lazy_mod
import ppo_late_fusion_rgb_corridor
import ppo_film_factorized_gray
from ppo_cnn_gray import FrameStack as PPOFrameStack

# ---------- GLOBALS (same as demo.py; only used to construct agents) ----------
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000
batch_size = 64
frame_repeat = 12


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
        choices=list(MODELS.AGENT_BY_MODEL.keys()) + ["random"],
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

    # Ignore remaining arguments for random model
    if args.model_type == "random":
        return args, None, None

    agent_builder = MODELS.AGENT_BY_MODEL[args.model_type]

    default_path = Path("../models") / f"{args.model_type}.pth"

    # Prefer ../models/<scenario_stem>/<model_type>.pth if it exists (matches newer training layout).
    scenario_default = MODELS.MODEL_DEFAULT_SCENARIO.get(args.model_type)
    if scenario_default:
        scen_stem = Path(scenario_default).stem
        alt = Path("../models") / scen_stem / f"{args.model_type}.pth"
        if alt.exists():
            default_path = alt
    model_path = Path(args.model_path) if args.model_path else default_path

    return args, agent_builder, model_path


@torch.no_grad()
def evaluate(game: vzd.DoomGame, agent, actions, *, model_type: str, resolution, episodes: int, visible_window: bool, use_frame_stack: bool, frame_stack):
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

        if use_frame_stack:
            if model_type == "q_rainbow_stacked":
                frame_stack.frames.clear()  # will be reset with first frame
                if hasattr(frame_stack, "_inited"):
                    frame_stack._inited = False
            else:
                frame_stack.reset()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break

            # Preprocess
            if model_type == "q_rainbow_stacked":
                # This model expects uint8 CHW frames and does its own lazy stacking convention.
                frame_u8 = rainbow_lazy_mod.preprocess_frame_u8(gs.screen_buffer)
                if use_frame_stack:
                    # Fill the stack with the first frame for a clean start (matches training).
                    if not getattr(frame_stack, "_inited", False):
                        frame_stack.reset(frame_u8)
                        frame_stack._inited = True
                    else:
                        frame_stack.append(frame_u8)
                    state_img = frame_stack.get()  # uint8 (C*K,H,W)
                else:
                    state_img = frame_u8
            else:
                state_img = preprocess_fn(gs.screen_buffer, resolution)

                # Apply frame stacking if needed
                if use_frame_stack:
                    frame_stack.push(state_img)
                    state_img = frame_stack.get()

            if model_type in MODELS.PPO_MODELS:
                if model_type in MODELS.PPO_STATE_VAR_MODELS:
                    # TODO: special exception for this model since it uses custom actions
                    if model_type == "ppo_late_fusion_rgb_corridor":
                        state_vars = ppo_late_fusion_rgb_corridor.preprocess_vars_corridor(gs.game_variables)
                    elif model_type == "ppo_film_factorized_gray":
                        state_vars = ppo_film_factorized_gray.preprocess_vars_safe_general(gs.game_variables, ppo_film_factorized_gray.NUM_VARS, normalizer=agent.vars_rms, update=False, clip=5.0)
                    else:
                        state_vars = preprocess_vars_safe(gs.game_variables, expected_num_vars)
                    a = agent.get_action(state_img, state_vars, deterministic=True)
                else:
                    a = agent.get_action(state_img, deterministic=True)
            else:
                state_vars = preprocess_vars_safe(gs.game_variables, expected_num_vars)
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

def run_random_agent(game, episodes, scenario):
    # Special button handling
    if scenario == "defend_the_line":
        game.add_available_button(vzd.Button.MOVE_LEFT)
        game.add_available_button(vzd.Button.MOVE_RIGHT)

    # Build action space
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    print("Number of Buttons available:", game.get_available_buttons_size())
    print("Number of Actions available:", len(actions))

    # Initialize game
    game.init()

    # Evaluate
    scores = []
    for ep in range(episodes):
        game.new_episode()

        while not game.is_episode_finished():
            gs = game.get_state()
            if gs is None:
                break
            
            random_action = choice(actions)
            if visible_window:
                game.set_action(random_action)
                for _ in range(frame_repeat):
                    game.advance_action()
            else:
                game.make_action(random_action, frame_repeat)

        score = game.get_total_reward()
        scores.append(score)

        if visible_window:
            print(f"Episode {ep + 1} Total Score: {score}")
            sleep(0.2)
        elif ep % 10 == 0:
            print(f"Episode {ep + 1} Total Score: {score}")

    scores = np.asarray(scores, dtype=np.float32)

    print("======================================")
    print("Score: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
        float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
    ))

if __name__ == "__main__":
    # Uses GPU if available (same as demo.py)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    args, AgentBuilder, model_path = parse_cli()

    visible_window = args.show
    scenario_file = args.scenario if args.scenario else MODELS.MODEL_DEFAULT_SCENARIO.get(
        args.model_type, "defend_the_center.cfg"
    )
    config_file_path = os.path.join(SCENARIO_PATH, scenario_file)

    model_loadfile = str(model_path)

    print("model_type:", args.model_type)
    print("load path :", model_loadfile)
    print("scenario  :", scenario_file)
    print("config    :", config_file_path)
    print("show      :", visible_window)
    print("episodes  :", args.episodes)

    # Initialize game
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER if visible_window else vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # If running dummy (just random actions), skip everything
    if args.model_type == "random":
        run_random_agent(game, args.episodes, args.scenario)
        game.close()
        sys.exit(0)

    # Match demo.py's screen format selection, but allow AUTO for modules that self-toggle RGB/Gray
    color_mode = MODELS.COLOR_BY_MODEL[args.model_type]
    if color_mode == MODELS.RGB:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess_fn = preprocess_rgb
    elif color_mode == MODELS.GRAYSCALE:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        preprocess_fn = preprocess
    elif color_mode == MODELS.AUTO and args.model_type == "q_rainbow_stacked":
        # Use the module's single switch (USE_GRAYSCALE) to decide.
        if rainbow_lazy_mod.USE_GRAYSCALE:
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
        else:
            game.set_screen_format(vzd.ScreenFormat.RGB24)
        # We will use rainbow_lazy_mod.preprocess_frame_u8 inside evaluate() for this model.
        preprocess_fn = None
    else:
        raise ValueError(f"Invalid color format for model type {args.model_type}")
    
    if hasattr(game, "set_render_hud"):
        game.set_render_hud(True)

    resolution = MODELS.RESOLUTION_BY_MODEL[args.model_type]
    game.init()

    # Build action space
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    # TODO: special exception for this model since it uses custom actions
    if args.model_type == "ppo_late_fusion_rgb_corridor":
        actions = ppo_late_fusion_rgb_corridor.get_deadly_corridor_actions()

    # Build agent
    if args.model_type == "ppo_film_factorized_gray":
        mapper = MODELS.FactorizedActionMapper(game)
        agent = AgentBuilder(action_mapper=mapper)
    elif args.model_type in MODELS.PPO_MODELS:
        agent = AgentBuilder(action_size=len(actions), load_model_path=model_path)
    elif args.model_type == "q_rainbow_stacked":
        # This agent's constructor does not take load_model/model_weights; load weights manually.
        agent = AgentBuilder(
            action_size=len(actions),
            lr=learning_rate,
            discount_factor=discount_factor,
            memory_size=replay_memory_size,
            batch_size=batch_size,
        )
        try:
            sd = torch.load(model_path, map_location=DEVICE, weights_only=True)
        except TypeError:
            sd = torch.load(model_path, map_location=DEVICE)
        agent.q_net.load_state_dict(sd)
        agent.set_eval_mode() if hasattr(agent, "set_eval_mode") else None
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

    # Set up frame stacking if needed
    use_frame_stack = args.model_type in MODELS.FRAME_STACK_MODELS
    if use_frame_stack:
        if args.model_type in MODELS.LAZY_STACK_MODULE_BY_MODEL:
            # Uses its own stacker: stores uint8 frames (C,H,W) and concatenates to (C*K,H,W).
            frame_stack = rainbow_lazy_mod.FrameStack(rainbow_lazy_mod.FRAME_STACK_SIZE, rainbow_lazy_mod.FRAME_C, resolution)
        else:
            FrameStack = MODELS.FRAME_STACK_MODELS[args.model_type]
            frame_stack = FrameStack(MODELS.FRAME_STACK_SIZE[args.model_type], resolution)
    else:
        frame_stack = None

    # Evaluate
    scores = evaluate(
        game,
        agent,
        actions,
        model_type=args.model_type,
        resolution=resolution,
        episodes=args.episodes,
        visible_window=visible_window,
        use_frame_stack=use_frame_stack,
        frame_stack=frame_stack,
    )

    print("======================================")
    print("Score: mean {:.2f} +/- {:.2f}, min {:.2f}, max {:.2f}".format(
        float(scores.mean()), float(scores.std()), float(scores.min()), float(scores.max())
    ))

    game.close()