#!/usr/bin/env python3

from vizdoom import *
import random
import time

# Create DoomGame instance
game = DoomGame()

# Load basic scenario
game.load_config("scenarios/basic.cfg")

# Set screen resolution and format
game.set_screen_resolution(ScreenResolution.RES_640X480)
game.set_window_visible(True)

# Initialize the game
game.init()

print("Starting game...")

# Run a few episodes
episodes = 3
for i in range(episodes):
    print(f"\nEpisode #{i + 1}")
    game.new_episode()
    
    while not game.is_episode_finished():
        # Get the game state
        state = game.get_state()
        
        # Make a random action
        action = [random.choice([0, 1]) for _ in range(game.get_available_buttons_size())]
        
        # Perform action and get reward
        reward = game.make_action(action)
        
        time.sleep(0.02)  # Slow it down a bit to watch
    
    print(f"Episode finished. Total reward: {game.get_total_reward()}")

game.close()
print("\nTest complete!")
