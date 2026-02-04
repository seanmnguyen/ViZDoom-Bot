# Script to load a model and run it against a scenario

import itertools as it
import os
import sys
from time import sleep

import torch
from tqdm import trange

import vizdoom as vzd
from utils import *

from late_fusion import DQNAgent as DQNAgent_LateFusion
from basic_dqn import DQNAgent as DQNAgent_Basic

# ---------- GLOBALS ----------
# The main configurations for this demo:
save_model = True
load_model = True
visible_window = True
# Configuration file path
config_file_path = os.path.join(vzd.scenarios_path, "defend_the_line.cfg")

# Just necessary for building the agent, can mostly ignore
# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 5
learning_steps_per_epoch = 2000
replay_memory_size = 10000

# NN learning settings
batch_size = 64

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (30, 45)
episodes_to_watch = 5

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")


# ---------- DRIVER ----------
if __name__ == "__main__":
    # Get Model Type, set model_savefile, and Agent
    if len(sys.argv) < 2:
        print("Command line argument expected: <model_type>")
        print("Options: 'q_cnn', 'q_late_fusion")
        exit()
    model_type = sys.argv[1]
    if model_type == "q_cnn":
        AgentBuilder = DQNAgent_Basic
        model_savefile = f"../models/{model_type}.pth"
    elif model_type == "q_late_fusion":
        AgentBuilder = DQNAgent_LateFusion
        model_savefile = f"../models/{model_type}.pth"
    elif model_type == "late_fusion_long":
        AgentBuilder = DQNAgent_LateFusion
        model_savefile = f"../models/{model_type}.pth"
    else:
        print(f"Unexpected model type: {model_type}")
        exit()
    # model_savefile = f"../models/{model_type}.pth"

    if len(sys.argv) == 3:
        if sys.argv[2] == "True":
            visible_window = True
        elif sys.argv[2] == "False":
            visible_window = False
        else:
            print("Invalid visible window argument. Expected 'True' or 'False'")

    # Initialize game and actions with window visible
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(visible_window)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    # TODO: For late_fusion: modifying .cfg file not working;
    # Remove manual button additions when .cfg changes recognized
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.init()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = AgentBuilder(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
        model_weights=model_savefile,
    )

    # Play episode with model
    total_score = 0
    for episode_num in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            state_img = preprocess(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, len(game.get_available_game_variables()))
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
    print(f"-----Average Score: {total_score / episodes_to_watch}-----")
