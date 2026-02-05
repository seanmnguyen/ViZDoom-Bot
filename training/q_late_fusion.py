#!/usr/bin/env python3

import itertools as it
import os
import random
from collections import deque
from time import sleep, time

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

import vizdoom as vzd
from utils import *

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
episodes_to_watch = 10

model_savefile = "../models/q_late_fusion.pth"
save_model = True
load_model = False
skip_learning = False

# Configuration file path
config_file_path = os.path.join(SCENARIO_PATH, "defend_the_center.cfg")

# Uses GPU if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device("cpu")

GAME_VARS = [
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.AMMO2,
]
NUM_VARS = len(GAME_VARS)
ARCH = "late_fusion"  # or "film"


def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    for gv in GAME_VARS:
        game.add_available_game_variable(gv)
    game.init()
    print("Doom initialized.")

    return game


def test(game, agent):
    """Runs a test_episodes_per_epoch episodes and prints the result"""
    print("\nTesting...")
    test_scores = []
    for test_episode in trange(test_episodes_per_epoch, leave=False):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            state_img = preprocess(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)
            best_action_index = agent.get_action(state_img, state_vars)

            game.make_action(actions[best_action_index], frame_repeat)
        r = game.get_total_reward()
        test_scores.append(r)

    test_scores = np.array(test_scores)
    print(
        "Results: mean: {:.1f} +/- {:.1f},".format(
            test_scores.mean(), test_scores.std()
        ),
        "min: %.1f" % test_scores.min(),
        "max: %.1f" % test_scores.max(),
    )


def run(game, agent, actions, num_epochs, frame_repeat, steps_per_epoch=2000):
    """
    Run num epochs of training episodes.
    Skip frame_repeat number of frames after each action.
    """

    start_time = time()

    for epoch in range(num_epochs):
        game.new_episode()
        train_scores = []
        global_step = 0
        print(f"\nEpoch #{epoch + 1}")

        for _ in trange(steps_per_epoch, leave=False):
            game_state = game.get_state()
            state_img = preprocess(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)

            action = agent.get_action(state_img, state_vars)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_game_state = game.get_state()
                next_img = preprocess(next_game_state.screen_buffer, resolution)
                next_vars = preprocess_vars(next_game_state.game_variables, NUM_VARS)
            else:
                next_img = np.zeros((1, 30, 45), dtype=np.float32)
                next_vars = np.zeros((NUM_VARS,), dtype=np.float32)

            agent.append_memory(state_img, state_vars, action, reward, next_img, next_vars, done)

            if global_step > agent.batch_size:
                agent.train()

            if done:
                train_scores.append(game.get_total_reward())
                game.new_episode()

            global_step += 1

        agent.update_target_net()
        train_scores = np.array(train_scores)

        print(
            "Results: mean: {:.1f} +/- {:.1f},".format(
                train_scores.mean(), train_scores.std()
            ),
            "min: %.1f," % train_scores.min(),
            "max: %.1f," % train_scores.max(),
        )

        test(game, agent)
        if save_model:
            print("Saving the network weights to:", model_savefile)
            torch.save(agent.q_net.state_dict(), model_savefile)
        print("Total elapsed time: %.2f minutes" % ((time() - start_time) / 60.0))

    game.close()
    return agent, game

class LateFusionDuelQNet(nn.Module):
    """
    Late fusion: CNN(img) + MLP(vars) -> concat -> dueling head.
    Keeps your conv stack (same as current DuelQNet).
    """
    def __init__(self, available_actions_count: int, num_vars: int):
        super().__init__()

        # same conv backbone you already had
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # NOTE: flattens to 192 for resolution (30,45)
        self.img_fc = nn.Sequential(
            nn.Linear(192, 128),
            nn.ReLU(),
        )

        self.vars_mlp = nn.Sequential(
            nn.LayerNorm(num_vars),
            nn.Linear(num_vars, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        fused_dim = 128 + 64
        self.state_fc = nn.Sequential(nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage_fc = nn.Sequential(
            nn.Linear(fused_dim, 64), nn.ReLU(), nn.Linear(64, available_actions_count)
        )

    def forward(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # (B, 192)

        img_feat = self.img_fc(x)          # (B, 128)
        vars_feat = self.vars_mlp(vars_)   # (B, 64)

        fused = torch.cat([img_feat, vars_feat], dim=1)  # (B, 192)
        state_value = self.state_fc(fused)               # (B, 1)
        advantage = self.advantage_fc(fused)             # (B, A)

        q = state_value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q

def _apply_film(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    # h: (B,C,H,W), gamma/beta: (B,C)
    g = gamma.unsqueeze(-1).unsqueeze(-1)
    b = beta.unsqueeze(-1).unsqueeze(-1)
    return (1.0 + g) * h + b


class DQNAgent:
    def __init__(
        self,
        action_size,
        memory_size,
        batch_size,
        discount_factor,
        lr,
        load_model,
        epsilon=1,
        epsilon_decay=0.9996,
        epsilon_min=0.1,
        model_weights=None
    ):
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.discount = discount_factor
        self.lr = lr
        self.memory = deque(maxlen=memory_size)
        self.criterion = nn.MSELoss()

        if load_model:
            if model_weights is not None:  # weights inputted
                global model_savefile
                model_savefile = model_weights
            print("Loading Q-LateFusion model from:", model_savefile)
            sd = torch.load(model_savefile, map_location=DEVICE)  # now this is a dict

            # TODO: may need to refactor to be compatible with FiLMDuelQNet
            self.q_net = LateFusionDuelQNet(action_size, NUM_VARS).to(DEVICE)
            self.q_net.load_state_dict(sd)

            self.target_net = LateFusionDuelQNet(action_size, NUM_VARS).to(DEVICE)
            self.target_net.load_state_dict(sd)

            self.q_net.eval()
            self.target_net.eval()
            self.epsilon = 0.0

        else:
            print("Initializing new model")
            if ARCH == "late_fusion":
                self.q_net = LateFusionDuelQNet(action_size, NUM_VARS).to(DEVICE)
                self.target_net = LateFusionDuelQNet(action_size, NUM_VARS).to(DEVICE)
            elif ARCH == "film":
                self.q_net = FiLMDuelQNet(action_size, NUM_VARS).to(DEVICE)
                self.target_net = FiLMDuelQNet(action_size, NUM_VARS).to(DEVICE)
            else:
                raise ValueError(f"Unknown ARCH: {ARCH}")

        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)

    def get_action(self, state_img, state_vars):
        if np.random.uniform() < self.epsilon:
            return random.choice(range(self.action_size))
        else:
            # add batch dimension
            state_img = np.expand_dims(state_img, axis=0)      # (1,1,H,W)
            state_vars = np.expand_dims(state_vars, axis=0)    # (1,V)

            img_t = torch.from_numpy(state_img).float().to(DEVICE)
            vars_t = torch.from_numpy(state_vars).float().to(DEVICE)

            action = torch.argmax(self.q_net(img_t, vars_t), dim=1).item()
            return action

    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, img, vars_, action, reward, next_img, next_vars, done):
        self.memory.append((img, vars_, action, reward, next_img, next_vars, done))

    def train(self):
        batch = random.sample(self.memory, self.batch_size)

        imgs      = np.stack([b[0] for b in batch]).astype(np.float32)  # (B,1,H,W)
        vars_     = np.stack([b[1] for b in batch]).astype(np.float32)  # (B,V)
        actions   = np.array([b[2] for b in batch], dtype=np.int64)     # (B,)
        rewards   = np.array([b[3] for b in batch], dtype=np.float32)   # (B,)
        next_imgs = np.stack([b[4] for b in batch]).astype(np.float32)  # (B,1,H,W)
        next_vars = np.stack([b[5] for b in batch]).astype(np.float32)  # (B,V)
        dones     = np.array([b[6] for b in batch], dtype=np.float32)   # (B,) 1.0 if done else 0.0

        imgs_t      = torch.from_numpy(imgs).to(DEVICE)
        vars_t      = torch.from_numpy(vars_).to(DEVICE)
        actions_t   = torch.from_numpy(actions).to(DEVICE)
        rewards_t   = torch.from_numpy(rewards).to(DEVICE)
        next_imgs_t = torch.from_numpy(next_imgs).to(DEVICE)
        next_vars_t = torch.from_numpy(next_vars).to(DEVICE)
        dones_t     = torch.from_numpy(dones).to(DEVICE)

        # Q(s,a)
        q = self.q_net(imgs_t, vars_t)  # (B,A)
        q_sa = q.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # Double DQN: a* = argmax_a Q_online(s',a)
            next_q_online = self.q_net(next_imgs_t, next_vars_t)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)  # (B,1)

            # target uses target net
            next_q_target = self.target_net(next_imgs_t, next_vars_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)  # (B,)

            targets = rewards_t + self.discount * (1.0 - dones_t) * next_q

        self.opt.zero_grad()
        loss = self.criterion(targets, q_sa)
        loss.backward()
        self.opt.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


if __name__ == "__main__":
    # Initialize game and actions
    game = create_simple_game()
    n = game.get_available_buttons_size()
    actions = [list(a) for a in it.product([0, 1], repeat=n)]

    # Initialize our agent with the set parameters
    agent = DQNAgent(
        len(actions),
        lr=learning_rate,
        batch_size=batch_size,
        memory_size=replay_memory_size,
        discount_factor=discount_factor,
        load_model=load_model,
    )

    # Run the training for the set number of epochs
    if not skip_learning:
        agent, game = run(
            game,
            agent,
            actions,
            num_epochs=train_epochs,
            frame_repeat=frame_repeat,
            steps_per_epoch=learning_steps_per_epoch,
        )

        print("======================================")
        print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
    game.close()
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.init()

    total_score = 0
    for episode_num in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            state_img = preprocess(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)
            best_action_index = agent.get_action(state_img, state_vars)

            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
        sleep(1.0)
        score = game.get_total_reward()
        total_score += score
        print(f"Episode {episode_num + 1} Total score: {score}")
    print(f"-----Average Score: {total_score / episodes_to_watch}-----")
