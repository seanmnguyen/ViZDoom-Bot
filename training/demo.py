# Script to load a model and run it against a scenario

import itertools as it
import os
from time import sleep

import torch
import argparse
from pathlib import Path

import vizdoom as vzd
from utils import *

from q_late_fusion import DQNAgent as DQNAgent_LateFusion
from q_late_fusion_rgb import DQNAgent as DQNAgent_LateFusionRGB
from q_cnn import DQNAgent as DQNAgent_CNN
from q_cnn_rgb import DQNAgent as DQNAgent_CNNRGB
from ppo_cnn import PPOAgent

# ---------- GLOBALS ----------
# Default scenario for each model type (matches training configs)
MODEL_DEFAULT_SCENARIO = {
    "q_cnn": "defend_the_line.cfg",
    "q_cnn_rgb": "defend_the_line.cfg",
    "q_late_fusion": "defend_the_center.cfg",
    "q_late_fusion_rgb": "defend_the_center.cfg",
    "ppo_cnn": "defend_the_line.cfg",
}

# Just necessary for building the agent, can mostly ignore
# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Other parameters
frame_repeat = 12
EPISODES_TO_WATCH = 5

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


# ---------- CLI PARSER ----------
# Map model type -> agent class
AGENT_BY_MODEL = {
    "q_cnn": DQNAgent_CNN,
    "q_cnn_rgb": DQNAgent_CNNRGB,
    "q_late_fusion": DQNAgent_LateFusion,
    "q_late_fusion_rgb": DQNAgent_LateFusionRGB,
    "ppo_cnn": PPOAgent,
}

# Map model type -> resolution (for preprocessing)
RESOLUTION_BY_MODEL = {
    "q_cnn": (30, 45),
    "q_cnn_rgb": (96, 128),
    "q_late_fusion": (30, 45),
    "q_late_fusion_rgb": (96, 128),
    "ppo_cnn": (30, 45),
}

# Map model type -> RGB or grayscale
GRAYSCALE = "GRAY8"
RGB = "RGB24"
COLOR_BY_MODEL = {
    "q_cnn": GRAYSCALE,
    "q_cnn_rgb": RGB,
    "q_late_fusion": GRAYSCALE,
    "q_late_fusion_rgb": RGB,
    "ppo_cnn": GRAYSCALE,
}

# PPO model interface
PPO_MODELS = {"ppo_cnn"}


def str2bool(v):
    """Parse bools from CLI strings."""
    if isinstance(v, bool):
        return v
    v = v.strip().lower()
    if v in {"true", "t", "1", "yes", "y"}:
        return True
    if v in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: '{v}'. Use True/False."
    )


def parse_cli():
    '''
    Parse command line arguments.

    Returns: args, agent_builder, model_path
    '''
    parser = argparse.ArgumentParser(
        description="Train/evaluate ViZDoom agents."
    )

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
        default=True,
        metavar="BOOL",
        help="Show game window (True/False)."
    )

    parser.add_argument(
        "-sc", "--scenario",
        type=str,
        default=None,
        help="Scenario config file (e.g., defend_the_line.cfg). Defaults to training scenario for model type."
    )

    # -h / --help is automatically provided by argparse
    args = parser.parse_args()

    # Resolve model builder
    agent_builder = AGENT_BY_MODEL[args.model_type]

    # Resolve default paths if not provided
    default_path = Path("../models") / f"{args.model_type}.pth"
    model_path = Path(args.model_path) if args.model_path else default_path

    return args, agent_builder, model_path


# ---------- DRIVER ----------
if __name__ == "__main__":
    # Get Model Type, Agent, and set model path
    args, AgentBuilder, model_path = parse_cli()

    model_loadfile = str(model_path)     # for loading
    visible_window = args.show

    # Resolve scenario config - use CLI arg or default for model type
    scenario_file = args.scenario if args.scenario else MODEL_DEFAULT_SCENARIO.get(
        args.model_type, "defend_the_center.cfg")
    config_file_path = os.path.join(SCENARIO_PATH, scenario_file)

    print("model_type:", args.model_type)
    print("load path :", model_loadfile)
    print("scenario  :", scenario_file)
    print("show      :", visible_window)

    # Initialize game and actions with window visible
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(True)

    # Specific configs for QLateFusionRGB
    if COLOR_BY_MODEL[args.model_type] == RGB:
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        preprocess = preprocess_rgb
    elif COLOR_BY_MODEL[args.model_type] == GRAYSCALE:
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        preprocess = preprocess
    else:
        raise ValueError(f"Invalid color format for model type {args.model_type}")

    resolution = RESOLUTION_BY_MODEL[args.model_type]
    game.init()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    if args.model_type in PPO_MODELS:
        # PPO agents have a different constructor
        agent = AgentBuilder(
            action_size=len(actions),
            load_model_path=model_path,
        )
    else:
        # DQN-based agents
        agent = AgentBuilder(
            len(actions),
            lr=learning_rate,
            batch_size=batch_size,
            memory_size=replay_memory_size,
            discount_factor=discount_factor,
            load_model=True,
            model_weights=model_path,
        )

    # Play episode with model
    total_score = 0
    for episode_num in range(EPISODES_TO_WATCH):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            state_img = preprocess(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, len(
                game.get_available_game_variables()))

            # PPO agents use deterministic=True for evaluation
            if args.model_type in PPO_MODELS:
                best_action_index = agent.get_action(
                    state_img, deterministic=True)
            else:
                best_action_index = agent.get_action(state_img, state_vars)

            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        if visible_window:
            sleep(1.0)
        score = game.get_total_reward()
        total_score += score
        print(f"Episode {episode_num + 1} Total Score: {score}")
    print(f"-----Average Score: {total_score / EPISODES_TO_WATCH}-----")
