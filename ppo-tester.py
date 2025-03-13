import os
import sys
import time
import signal
import random
import keyboard
import threading
import numpy as np
from stable_baselines3 import PPO
from gym_utils import load_smb_env, SMB


def signal_handler(sig, frame):
    """Handle interrupt signals and exit gracefully"""
    print("\n\nStopping Mario AI - Cleaning up resources...")
    if 'env_wrap' in globals():
        try:
            env_wrap.close()
            print("Environment closed successfully")
        except Exception as e:
            print(f"Error closing environment: {e}")
    print("Exiting...")
    sys.exit(0)


# Set up signal handlers for graceful exit
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def check_for_exit_key():
    """Monitor for escape key to exit"""
    try:
        keyboard.wait('esc')
        print("\nEscape key pressed - exiting...")
        os.kill(os.getpid(), signal.SIGINT)
    except:
        pass


class SMBWithRandomness:
    """
    Custom wrapper for SMB class to add controlled randomness
    """

    def __init__(self, env, model, random_chance=0.05):
        self.env = env
        self.model = model
        self.random_chance = random_chance
        self.action_space = env.action_space
        self.action_counts = {}  # Track actions for stats

    def play(self, episodes=1, deterministic=True, render=True, return_eval=False):
        total_score = 0
        total_steps = 0
        final_info = {}
        action_history = []

        for episode in range(1, episodes+1):
            states = self.env.reset()
            done = False
            score = 0
            steps = 0

            if render:
                while not done:
                    self.env.render()
                    action, _ = self.model.predict(
                        states, deterministic=deterministic)
                    action_source = "model"

                    # Track action for statistics
                    action_key = f"{action[0]}"
                    if action_key not in self.action_counts:
                        self.action_counts[action_key] = 0
                    self.action_counts[action_key] += 1
                    action_history.append((action[0], action_source))

                    states, reward, done, info = self.env.step(action)
                    score += reward
                    steps += 1
                    time.sleep(0.01)

                print(f'Episode:{episode} Score:{score} Steps:{steps}')
                # Print action distribution
                total_actions = sum(self.action_counts.values())
                if total_actions > 0:
                    print("Action distribution:")
                    for action, count in sorted(self.action_counts.items()):
                        print(
                            f"  Action {action}: {count} times ({count/total_actions*100:.1f}%)")
            else:
                while not done:
                    if random.random() < self.random_chance:
                        action = [self.action_space.sample()]
                    else:
                        action, _ = self.model.predict(
                            states, deterministic=deterministic)

                    states, reward, done, info = self.env.step(action)
                    score += reward
                    steps += 1

            total_score += score
            total_steps += steps
            final_info = info

        if return_eval:
            return total_score, total_steps
        else:
            return action_history


def run_mario_ai(model_name='best_model_6300000', crop_dim=[0, 16, 0, 13], n_stack=4, n_skip=4,
                 render=True, deterministic=True, random_chance=0.05, delay_between_episodes=1.0):
    """
    Run the Mario AI model in a continuous loop until manually stopped
    """
    MODEL_DIR = './models/v1'
    version = 'SuperMarioBros-1-1-v1'
    episode_count = 0
    total_reward = 0
    total_steps = 0

    # Print startup message
    print("\n" + "="*50)
    print(f"Starting Mario AI with model: {model_name}")
    print(f"Using {random_chance*100:.1f}% random actions")
    print("Press CTRL+C or ESC to stop")
    print("="*50 + "\n")

    # Load environment
    print("Loading environment...")
    global env_wrap
    env_wrap = load_smb_env(version, crop_dim, n_stack, n_skip)
    print("Environment loaded successfully")

    # Load model
    try:
        print(f"Loading model from: {os.path.join(MODEL_DIR, model_name)}")
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        model = PPO.load(os.path.join(MODEL_DIR, model_name),
                         env=env_wrap, custom_objects=custom_objects)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Creating a new model...")
        model = PPO('MlpPolicy', env_wrap)
        print("New model created (Note: This model has no training)")

    # Create custom SMB instance with randomness
    smb = SMBWithRandomness(env_wrap, model, random_chance=random_chance)

    # Main loop - run continuously until interrupted
    try:
        while True:
            episode_count += 1
            print(f"\nStarting episode #{episode_count}")
            start_time = time.time()

            # Play one episode
            try:
                rewards, steps = smb.play(
                    episodes=1, deterministic=deterministic, render=render, return_eval=True)
                episode_duration = time.time() - start_time
                total_reward += rewards
                total_steps += steps
                avg_reward = total_reward / episode_count
                avg_steps = total_steps / episode_count

                print(
                    f"Episode #{episode_count} completed in {episode_duration:.2f} seconds")
                print(f"Reward: {rewards}, Steps: {steps}")
                print(
                    f"Avg Reward: {avg_reward:.2f}, Avg Steps: {avg_steps:.2f}")
            except Exception as e:
                print(f"Error during episode: {e}")

            # Wait between episodes
            print(
                f"Waiting {delay_between_episodes} seconds before next episode...")
            time.sleep(delay_between_episodes)

    except KeyboardInterrupt:
        # This is handled by the signal handler
        pass
    finally:
        # Make sure to clean up resources
        try:
            env_wrap.close()
            print("Environment closed successfully")
        except Exception as e:
            print(f"Error closing environment: {e}")


if __name__ == "__main__":
    # Start keyboard monitoring thread
    keyboard_thread = threading.Thread(target=check_for_exit_key)
    keyboard_thread.daemon = True
    keyboard_thread.start()

    # Process command line arguments
    model_name = 'best_model_6300000'
    deterministic_mode = True  # Base model is deterministic
    random_chance = 0.05       # Default 5% randomness

    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    # Check if random chance is provided
    if len(sys.argv) > 2:
        try:
            random_chance = float(sys.argv[2])
            # Ensure it's a valid percentage
            random_chance = max(0.0, min(1.0, random_chance))
        except ValueError:
            print(
                f"Invalid random chance value: {sys.argv[2]}. Using default: 0.05 (5%)")

    # Different model configurations
    model_configs = {
        'pre-trained-1': {'crop_dim': [0, 16, 0, 13], 'n_stack': 4, 'n_skip': 4},
        'best_model_6300000': {'crop_dim': [0, 16, 0, 13], 'n_stack': 4, 'n_skip': 4},
        'pre-trained-3': {'crop_dim': [0, 16, 0, 13], 'n_stack': 2, 'n_skip': 4},
    }

    # Get model config or use default
    config = model_configs.get(
        model_name, model_configs['best_model_6300000'])

    # Run the AI
    run_mario_ai(
        model_name=model_name,
        crop_dim=config['crop_dim'],
        n_stack=config['n_stack'],
        n_skip=config['n_skip'],
        deterministic=deterministic_mode,
        random_chance=random_chance
    )
