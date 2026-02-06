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
import torch.nn.functional as F
from tqdm import trange

import vizdoom as vzd
from utils import *

# Q-learning settings
learning_rate = 0.00025
discount_factor = 0.99
train_epochs = 50
learning_steps_per_epoch = 5000
replay_memory_size = 10000

# NN learning settings
batch_size = 200  # 64 default; 256 too long; 128, 192 okay

# Training regime
test_episodes_per_epoch = 100

# Other parameters
frame_repeat = 12
resolution = (96, 128)
episodes_to_watch = 10

model_savefile = "../models/q_late_fusion_rgb.pth"
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

print("----------MODEL CONFIGURATION----------")
print("DEVICE:", DEVICE)
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

def create_simple_game():
    print("Initializing doom...")
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
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
            state_img = preprocess_rgb(game_state.screen_buffer, resolution)
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
            state_img = preprocess_rgb(game_state.screen_buffer, resolution)
            state_vars = preprocess_vars(game_state.game_variables, NUM_VARS)

            action = agent.get_action(state_img, state_vars)
            reward = game.make_action(actions[action], frame_repeat)
            done = game.is_episode_finished()

            if not done:
                next_game_state = game.get_state()
                next_img = preprocess_rgb(next_game_state.screen_buffer, resolution)
                next_vars = preprocess_vars(next_game_state.game_variables, NUM_VARS)
            else:
                next_img = np.zeros((3, resolution[0], resolution[1]), dtype=np.float32)
                next_vars = np.zeros((NUM_VARS,), dtype=np.float32)

            agent.append_memory(state_img, state_vars, action, reward, next_img, next_vars, done)

            if len(agent.memory) > agent.batch_size:
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

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        return x * s


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_se: bool = True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.se = SEBlock(out_ch) if use_se else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity, inplace=True)
        return out

class StrongCNN(nn.Module):
    """
    Stronger CNN for distant target detection:
    - less aggressive early downsampling
    - deeper residual stages
    - optional SE attention
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.stage1 = ResBlock(32, 32, stride=1, use_se=True)
        self.stage2 = nn.Sequential(
            ResBlock(32, 64, stride=2, use_se=True),   # downsample
            ResBlock(64, 64, stride=1, use_se=True),
        )
        self.stage3 = nn.Sequential(
            ResBlock(64, 96, stride=2, use_se=True),   # downsample
            ResBlock(96, 96, stride=1, use_se=True),
        )

        # slightly larger receptive field for distant objects
        self.context = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # keep some spatial layout (4x4) before flatten
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.context(x)
        x = self.pool(x)               # (B,128,4,4)
        x = torch.flatten(x, 1)        # (B,2048)
        return x

class LateFusionDuelQNet(nn.Module):
    """
    Late fusion: stronger CNN(img) + MLP(vars) -> concat -> dueling heads
    """
    def __init__(
        self,
        available_actions_count: int,
        num_vars: int,
        in_channels: int = 1,
        img_h: int = 72,
        img_w: int = 108
    ):
        super().__init__()

        self.cnn = StrongCNN(in_channels=in_channels)

        # infer CNN output dim automatically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_h, img_w)
            cnn_dim = self.cnn(dummy).shape[1]

        self.img_fc = nn.Sequential(
            nn.Linear(cnn_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

        self.vars_mlp = nn.Sequential(
            nn.LayerNorm(num_vars),
            nn.Linear(num_vars, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        fused_dim = 128 + 64
        self.state_fc = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.advantage_fc = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, available_actions_count),
        )

    def forward(self, img: torch.Tensor, vars_: torch.Tensor) -> torch.Tensor:
        img_feat = self.img_fc(self.cnn(img))
        vars_feat = self.vars_mlp(vars_)
        fused = torch.cat([img_feat, vars_feat], dim=1)

        v = self.state_fc(fused)               # (B,1)
        a = self.advantage_fc(fused)           # (B,A)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

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

            self.q_net = LateFusionDuelQNet(
                action_size, NUM_VARS, in_channels=3, img_h=resolution[0], img_w=resolution[1]).to(DEVICE)
            self.q_net.load_state_dict(sd)

            self.target_net = LateFusionDuelQNet(
                action_size, NUM_VARS, in_channels=3, img_h=resolution[0], img_w=resolution[1]).to(DEVICE)
            self.target_net.load_state_dict(sd)

            self.q_net.eval()
            self.target_net.eval()
            self.epsilon = 0.0

        else:
            print("Initializing new model")
            self.q_net = LateFusionDuelQNet(
                action_size, NUM_VARS, in_channels=3, img_h=resolution[0], img_w=resolution[1]).to(DEVICE)
            self.target_net = LateFusionDuelQNet(
                action_size, NUM_VARS, in_channels=3, img_h=resolution[0], img_w=resolution[1]).to(DEVICE)

        self.opt = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.update_target_net()

        if load_model:
            self.set_eval_mode()     # load-for-demo case
        else:
            self.set_train_mode()    # normal training case


    def get_action(self, state_img, state_vars):
        # epsilon-greedy
        if np.random.uniform() < self.epsilon:
            return random.randrange(self.action_size)

        # add batch dimension: (C,H,W)->(1,C,H,W), (V,)->(1,V)
        state_img = np.expand_dims(state_img, axis=0)
        state_vars = np.expand_dims(state_vars, axis=0)

        img_t = torch.from_numpy(state_img).float().to(DEVICE)
        vars_t = torch.from_numpy(state_vars).float().to(DEVICE)

        # inference only (no autograd graph)
        with torch.no_grad():
            q_values = self.q_net(img_t, vars_t)
            action = int(torch.argmax(q_values, dim=1).item())

        return action


    def update_target_net(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def append_memory(self, img, vars_, action, reward, next_img, next_vars, done):
        self.memory.append((img, vars_, action, reward, next_img, next_vars, done))

    def train(self):
        self.q_net.train()
        self.target_net.eval()

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

    def set_train_mode(self):
        self.q_net.train()
        self.target_net.eval()   # keep target frozen/stable

    def set_eval_mode(self):
        self.q_net.eval()
        self.target_net.eval()
        self.epsilon = 0.0



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
    agent.set_eval_mode()
    for episode_num in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            game_state = game.get_state()
            assert game_state is not None
            state_img = preprocess_rgb(game_state.screen_buffer, resolution)
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
