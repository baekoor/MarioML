import gym
import torch
import numpy as np
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


class MarioGrid:
    def __init__(self, env):
        self.ram = env.unwrapped.ram
        self.screen_size_x = 16
        self.screen_size_y = 13

        # Mario X pos
        self.mario_x = self.ram[0x3ad]
        self.mario_y = self.ram[0x3b8] + 16
        self.mario_level_x = self.ram[0x6d]*256 + self.ram[0x86]

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

# Reimplemented wrapper without external dependencies


class RamWrapper(gym.ObservationWrapper):
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

        self.width = crop_dim[1] - crop_dim[0]
        self.height = crop_dim[3] - crop_dim[2]
        self.observation_space = gym.spaces.Box(
            low=-1, high=2, shape=(self.height, self.width, self.n_stack), dtype=int
        )

        self.frame_stack = np.zeros(
            (self.height, self.width, (self.n_stack-1)*self.n_skip+1))

    def observation(self, obs):
        grid = MarioGrid(self.env)
        frame = grid.rendered_screen
        frame = self.crop_obs(frame)

        self.frame_stack[:, :, 1:] = self.frame_stack[:,
                                                      :, :-1]
        self.frame_stack[:, :, 0] = frame
        obs = self.frame_stack[:, :, ::self.n_skip]
        return obs

    def reset(self):
        obs = self.env.reset()
        self.frame_stack = np.zeros(
            (self.height, self.width, (self.n_stack-1)*self.n_skip+1))
        grid = MarioGrid(self.env)
        frame = grid.rendered_screen
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

# Load environment function


def load_env(name='SuperMarioBros-1-1-v1', crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=4):
    '''
    Wrapper function for loading and processing smb env
    '''
    env = gym_super_mario_bros.make(name)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env_wrap = RamWrapper(env, crop_dim, n_stack=n_stack, n_skip=n_skip)
    env_wrap = DummyVecEnv([lambda: env_wrap])

    return env_wrap

# Custom Mario utils class for model interaction


class MarioUtils:
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

            if render == True:
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

        if return_eval == True:
            return total_score, final_info
        else:
            return


# === Main Program Function ===

def load_and_play_mario(model_name='pre-trained-1', episodes=1):
    """
    Load and play a pre-trained Mario model
    """
    MODEL_DIR = './models'
    crop_dim = [0, 16, 0, 13]
    n_stack = 4
    n_skip = 4

    # Load the environment
    print("Loading environment...")
    env_wrap = load_env('SuperMarioBros-1-1-v1', crop_dim, n_stack, n_skip)

    # Create a custom model using direct policy loading
    print(f"Loading model {model_name}...")

    # Method 1: Try direct policy loading
    try:
        policy_path = os.path.join(MODEL_DIR, model_name, "policy.pth")
        optimizer_path = os.path.join(
            MODEL_DIR, model_name, "policy.optimizer.pth")

        # Create policy and load state dict
        policy = ActorCriticPolicy(
            observation_space=env_wrap.observation_space,
            action_space=env_wrap.action_space,
            lr_schedule=lambda _: 0.0003  # Default learning rate
        )

        # Load policy weights
        policy_weights = torch.load(policy_path)
        policy.load_state_dict(policy_weights)

        # Create PPO model
        model = PPO(
            policy=policy,
            env=env_wrap,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )

        print("Successfully loaded model!")

    except Exception as e:
        print(f"Error loading model directly: {e}")
        print("Attempting to use zip file instead...")

        try:
            # Method 2: Try the zip file
            model = PPO.load(
                os.path.join(MODEL_DIR, f"{model_name}.zip"),
                env=env_wrap
            )
            print("Successfully loaded model from zip file!")

        except Exception as e2:
            print(f"Error loading from zip file: {e2}")
            print("Creating a new PPO model instead...")

            # Method 3: Create a fresh model if all else fails
            model = PPO(
                policy="MlpPolicy",
                env=env_wrap,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1
            )

    print("Starting gameplay...")
    marioUtils = MarioUtils(env_wrap, model)
    score, info = marioUtils.play(
        episodes=episodes, deterministic=True, render=True, return_eval=True)

    print(f"Final score: {score}")
    print(f"Game info: {info}")

    return marioUtils


if __name__ == "__main__":
    marioUtils = load_and_play_mario(model_name='pre-trained-1', episodes=1)
