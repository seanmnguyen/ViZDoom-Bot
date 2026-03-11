#!/usr/bin/env python3
import os
import argparse
import itertools as it
from pathlib import Path

import numpy as np
import torch
import vizdoom as vzd

from utils import *
import model_registry as MODELS

import q_rainbow_stacked as rainbow_lazy_mod
import ppo_late_fusion_rgb_corridor
import ppo_film_factorized_gray
from ppo_cnn_gray import FrameStack as PPOFrameStack

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--join", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--name", default="AI")
    ap.add_argument("--color", type=int, default=0)
    ap.add_argument("--show", action="store_true")

    ap.add_argument("--model_type", choices=list(MODELS.AGENT_BY_MODEL.keys()), required=True)
    ap.add_argument("--model_path", type=str, required=True)

    ap.add_argument("--frame_repeat", type=int, default=12)  # match your training
    return ap.parse_args()

def main():
    # Uses GPU if available (same as demo.py)
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")

    args = build_args()
    model_path = Path(args.model_path)
    model_type = args.model_type
    color_mode = MODELS.COLOR_BY_MODEL[args.model_type]

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
    # game.set_doom_map("map01")

    # Join host
    game.add_game_args(f"-join {args.join} -port {args.port}")
    game.add_game_args(f"+name {args.name} +colorset {args.color}")

    # Async is recommended for multiplayer
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_window_visible(args.show)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    for gv in game.get_available_game_variables():
        game.add_available_game_variable(gv)

    resolution = MODELS.RESOLUTION_BY_MODEL[model_type]

    game.init()

    # Build discrete action space 
    n_buttons = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]
    print("ACTIONS:", len(actions))

    # Build agent
    AgentBuilder = MODELS.AGENT_BY_MODEL[model_type]
    learning_rate = 0.00025
    discount_factor = 0.99
    replay_memory_size = 10000
    batch_size = 128

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

    expected_num_vars = infer_expected_num_vars(agent, game)
    last_frags = None

    while not game.is_episode_finished():
        if game.is_player_dead():
            game.respawn_player()
            if frame_stack is not None:
                frame_stack.reset()
            continue

        gs = game.get_state()
        if gs is None:
            continue

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

        # Build observation
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

        # Act
        game.make_action(actions[a], args.frame_repeat)

        # Optional: print frag updates
        frags = int(game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        if last_frags is None or frags != last_frags:
            last_frags = frags
            print(f"[{args.name}] frags = {frags}")

    # End-of-match scoreboard
    server_state = game.get_server_state()
    if server_state is not None:
        print("=== SCOREBOARD ===")
        for i in range(len(server_state.players_in_game)):
            if server_state.players_in_game[i]:
                print(f"{server_state.players_names[i]}: {server_state.players_frags[i]}")

    game.close()

if __name__ == "__main__":
    main()