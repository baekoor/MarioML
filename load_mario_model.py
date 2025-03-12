#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mario RL Model Fix - Comprehensive solution that fixes both path and import issues
"""

import gym
import torch
import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym import spaces

# Define SMBGrid class (This should match what's expected in your code)


class SMBGrid:
    def __init__(self, env):
        self.ram = env.unwrapped.ram
        self.screen_size_x = 16     # rendered screen size
        self.screen_size_y = 13

        self.mario_level_x = self.ram[0x6d]*256 + self.ram[0x86]
        # mario's position on the rendered screen
        self.mario_x = self.ram[0x3ad]
        self.mario_y = self.ram[0x3b8] + 16  # top edge of (big) mario

        # left edge pixel of the rendered screen in level
        self.x_start = self.mario_level_x - self.mario_x
        self.rendered_screen = self.get_rendered_screen()

    def tile_loc_to_ram_address(self, x, y):
        '''
        convert (x, y) in Current tile (32x13, stored as 16x26 in ram) to ram address
        x: 0 to 31
        y: 0 to 12
        '''
        page = x // 16
        x_loc = x % 16
        y_loc = page*13 + y

        address = 0x500 + x_loc + y_loc*16

        return address

    def get_rendered_screen(self):
        '''
        Get the rendered screen (16 x 13) from ram
        empty: 0
        tile: 1
        enemy: -1
        mario: 2
        '''

        # Get background tiles
        rendered_screen = np.zeros((self.screen_size_y, self.screen_size_x))
        screen_start = int(np.rint(self.x_start / 16))

        for i in range(self.screen_size_x):
            for j in range(self.screen_size_y):
                x_loc = (screen_start + i) % (self.screen_size_x * 2)
                y_loc = j
                address = self.tile_loc_to_ram_address(x_loc, y_loc)

                # Convert all types of tile to 1
                if self.ram[address] != 0:
                    rendered_screen[j, i] = 1

        # Add mario
        x_loc = (self.mario_x + 8) // 16
        # top 2 rows in the rendered screen aren't stored in ram
        y_loc = (self.mario_y - 32) // 16
        if x_loc < 16 and y_loc < 13:
            rendered_screen[y_loc, x_loc] = 2

        # Add enemies
        for i in range(5):
            # check if the enemy is drawn
            if self.ram[0xF + i] == 1:
                enemy_x = self.ram[0x6e + i]*256 + \
                    self.ram[0x87 + i] - self.x_start
                enemy_y = self.ram[0xcf + i]
                x_loc = (enemy_x + 8) // 16
                y_loc = (enemy_y + 8 - 32) // 16

                # check if the enemy is inside the rendered screen
                if 0 <= x_loc < 16 and 0 <= y_loc < 13:
                    rendered_screen[y_loc, x_loc] = -1

        return rendered_screen

# Define SMBRamWrapper class


class SMBRamWrapper(gym.ObservationWrapper):
    def __init__(self, env, crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=2):
        '''
        crop_dim: [x0, x1, y0, y1]
        obs shape = (height, width, n_stack), n_stack=0 is the most recent frame
        n_skip: e.g. n_stack=4, n_skip=2, use frames [0, 2, 4, 6]
        '''
        gym.Wrapper.__init__(self, env)
        self.crop_dim = crop_dim
        self.n_stack = n_stack
        self.n_skip = n_skip
        # Modified from stable_baselines3.common.atari_wrappers.WarpFrame()
        self.width = crop_dim[1] - crop_dim[0]
        self.height = crop_dim[3] - crop_dim[2]
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(self.height, self.width, self.n_stack), dtype=int
        )

        self.frame_stack = np.zeros(
            (self.height, self.width, (self.n_stack-1)*self.n_skip+1))

    def observation(self, obs):
        # Use SMBGrid directly since we've defined it in this file
        grid = SMBGrid(self.env)
        frame = grid.rendered_screen  # 2d array
        frame = self.crop_obs(frame)

        self.frame_stack[:, :, 1:] = self.frame_stack[:,
                                                      :, :-1]  # shift frame_stack by 1
        self.frame_stack[:, :, 0] = frame  # add current frame to stack
        obs = self.frame_stack[:, :, ::self.n_skip]
        return obs

    def reset(self):
        obs = self.env.reset()
        self.frame_stack = np.zeros(
            (self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        grid = SMBGrid(self.env)  # Use SMBGrid directly
        frame = grid.rendered_screen  # 2d array
        frame = self.crop_obs(frame)
        for i in range(self.frame_stack.shape[-1]):
            self.frame_stack[:, :, i] = frame
        obs = self.frame_stack[:, :, ::self.n_skip]
        return obs

    def crop_obs(self, im):
        '''
        Crop observed frame image to reduce input size
        Returns cropped_frame = original_frame[y0:y1, x0:x1]
        '''
        [x0, x1, y0, y1] = self.crop_dim
        im_crop = im[y0:y1, x0:x1]
        return im_crop


def load_smb_env(name='SuperMarioBros-1-1-v1', crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=4):
    '''
    Wrapper function for loading and processing smb env
    '''
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env_wrap = SMBRamWrapper(env, crop_dim, n_stack=n_stack, n_skip=n_skip)
    env_wrap = DummyVecEnv([lambda: env_wrap])

    return env_wrap


class SMB:
    '''
    Wrapper function containing the processed environment and the loaded model
    '''

    def __init__(self, env, model):
        self.env = env
        self.model = model

    def play(self, episodes=5, deterministic=False, render=True, return_eval=False):
        total_score = 0
        final_info = {}

        for episode in range(1, episodes+1):
            states = self.env.reset()
            done = False
            score = 0

            if render:
                while not done:
                    self.env.render()
                    action, _ = self.model.predict(
                        states, deterministic=deterministic)
                    states, reward, done, info = self.env.step(action)
                    score += reward
                    time.sleep(0.01)
                print(f'Episode:{episode} Score:{score}')
            else:
                while not done:
                    action, _ = self.model.predict(
                        states, deterministic=deterministic)
                    states, reward, done, info = self.env.step(action)
                    score += reward

            total_score += score
            final_info = info

        if return_eval:
            return total_score, final_info
        else:
            return None

# Create a custom PPO class that allows policy replacement


class CustomPPO(PPO):
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy="MlpPolicy", env=env, **kwargs)
        # Replace the policy with our loaded policy
        self.policy = policy


def get_model_info():
    """Get information about available models"""
    MODEL_DIR = './models'
    print(f"Looking for models in {os.path.abspath(MODEL_DIR)}")

    if not os.path.exists(MODEL_DIR):
        print(f"ERROR: {MODEL_DIR} directory not found!")
        return

    models = os.listdir(MODEL_DIR)
    print(f"Available models: {models}")

    # Check for pre-trained-1
    model_name = 'pre-trained-1'
    model_dir = os.path.join(MODEL_DIR, model_name)
    if os.path.isdir(model_dir):
        print(f"Model {model_name} directory exists: {model_dir}")
        files = os.listdir(model_dir)
        print(f"Files in {model_name}: {files}")

        # Check for policy file
        policy_path = os.path.join(model_dir, "policy.pth")
        if os.path.exists(policy_path):
            print(f"Policy file exists: {policy_path}")
        else:
            print(f"Policy file NOT found: {policy_path}")
    else:
        print(f"Model {model_name} directory NOT found: {model_dir}")

        # Check for zip file
        zip_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
        if os.path.exists(zip_path):
            print(f"Model zip file exists: {zip_path}")
        else:
            print(f"Model zip file NOT found: {zip_path}")


def load_and_play_mario(model_name='pre-trained-1', episodes=1):
    """Load and play a pre-trained Mario model"""
    # First, get info about available models
    get_model_info()

    MODEL_DIR = './models'
    crop_dim = [0, 16, 0, 13]
    n_stack = 4
    n_skip = 4

    if model_name == 'pre-trained-2':
        n_stack = 1
        n_skip = 1
    elif model_name == 'pre-trained-3':
        n_stack = 2
        n_skip = 4

    # Load the environment
    print("\nLoading environment...")
    env_wrap = load_smb_env('SuperMarioBros-1-1-v1', crop_dim, n_stack, n_skip)

    # Try loading the model
    print(f"\nLoading model {model_name}...")

    # Check if policy exists in the directory
    policy_path = os.path.join(MODEL_DIR, model_name, "policy.pth")
    zip_path = os.path.join(MODEL_DIR, f"{model_name}.zip")

    if os.path.exists(policy_path):
        try:
            # Direct policy loading approach
            policy = ActorCriticPolicy(
                observation_space=env_wrap.observation_space,
                action_space=env_wrap.action_space,
                lr_schedule=lambda _: 0.0003  # Default learning rate
            )

            # Load policy weights
            policy_weights = torch.load(policy_path)
            policy.load_state_dict(policy_weights)

            # Create PPO model
            model = CustomPPO(policy=policy, env=env_wrap)
            print("Successfully loaded model from policy file!")

        except Exception as e:
            print(f"Error loading policy: {e}")
            return None
    elif os.path.exists(zip_path):
        try:
            # Try the zip file
            model = PPO.load(zip_path, env=env_wrap)
            print("Successfully loaded model from zip file!")

        except Exception as e:
            print(f"Error loading from zip file: {e}")
            return None
    else:
        print(f"Neither policy file nor zip file found for model {model_name}")
        return None

    # Create SMB instance and play
    print("\nStarting gameplay...")
    smb = SMB(env_wrap, model)
    score, info = smb.play(
        episodes=episodes, deterministic=True, render=True, return_eval=True)

    print(f"Final score: {score}")
    print(f"Game info: {info}")

    return smb


if __name__ == "__main__":
    smb = load_and_play_mario(model_name='best_model_1600000', episodes=1)

    if smb is None:
        print("\nTrying alternative model...")
        smb = load_and_play_mario(model_name='pre-trained-2', episodes=1)

    if smb is None:
        print("\nTrying another alternative model...")
        smb = load_and_play_mario(model_name='pre-trained-3', episodes=1)
