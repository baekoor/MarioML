import os
import sys
import time
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_trainer import DuelingDQN, arrange
from wrappers import wrap_mario


def load_model(model_path, device):
    """Robust model loading function"""
    # Set up network
    env = gym_super_mario_bros.make('SuperMarioBros-v3')
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)

    n_frame = 4
    n_action = env.action_space.n
    print(f"Action space size: {n_action}")
    print(f"Available actions: {COMPLEX_MOVEMENT}")

    q_network = DuelingDQN(n_frame, n_action, device).to(device)

    try:
        # Try loading the checkpoint
        checkpoint = torch.load(model_path, map_location=device)

        # Try different loading strategies
        if 'q_network_state_dict' in checkpoint:
            state_dict = checkpoint['q_network_state_dict']
        elif isinstance(checkpoint, dict):
            # Try finding the state dict
            for key, value in checkpoint.items():
                if 'state_dict' in key.lower():
                    state_dict = value
                    break
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Remove any keys that might not match current model
        current_keys = set(q_network.state_dict().keys())
        filtered_state_dict = {k: v for k,
                               v in state_dict.items() if k in current_keys}

        # Load the filtered state dict
        q_network.load_state_dict(filtered_state_dict, strict=False)

        print("Model loaded successfully!")
        return q_network, env

    except Exception as e:
        print(f"Error loading model: {e}")
        env.close()
        return None, None


def action_selector(q_network, state, device, exploration_rate=0.2):
    """
    Intelligent action selection with exploration
    - Use network Q-values with probabilistic exploration
    """
    with torch.no_grad():
        # Get Q-values from the network
        q_values = q_network(state)

        # Explore vs exploit decision
        if random.random() < exploration_rate:
            # Random action selection
            action = random.randint(0, q_values.shape[1] - 1)
        else:
            # Network's top actions
            top_actions = q_values.topk(3)

            # Weighted random selection from top actions
            probs = F.softmax(top_actions.values[0], dim=0)
            action = top_actions.indices[0][torch.multinomial(probs, 1)].item()

        return action


def continuous_evaluation(model_path, render=True, max_episodes=None):
    """Continuously evaluate the model with intelligent action selection"""
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load model
    q_network, env = load_model(model_path, device)

    if q_network is None:
        print("Failed to load model. Exiting.")
        return

    episode_count = 0
    try:
        while max_episodes is None or episode_count < max_episodes:
            # Reset environment
            state = env.reset()
            state = arrange(state)

            done = False
            total_reward = 0
            step_count = 0
            max_distance = 0
            action_counts = {}

            # Single episode evaluation
            while not done:
                if render:
                    env.render()
                    time.sleep(0.02)  # Add a small delay for rendering

                # Intelligent action selection
                action = action_selector(q_network, state, device)

                # Track action distribution
                action_counts[action] = action_counts.get(action, 0) + 1

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Track max distance
                max_distance = max(max_distance, info.get('x_pos', 0))

                # Update state
                state = arrange(next_state)

                total_reward += reward
                step_count += 1

                # Optional: break if too many steps
                if step_count > 5000:
                    print("Max steps reached. Breaking episode.")
                    break

            # Print episode summary
            episode_count += 1
            print(f"Episode {episode_count}: "
                  f"Reward: {total_reward:.2f}, "
                  f"Steps: {step_count}, "
                  f"Max Distance: {max_distance}")

            # Print action distribution
            print("Action Distribution:")
            for action, count in sorted(action_counts.items()):
                print(f"Action {action}: {count} times")

            # Optional: ask to continue
            if max_episodes is None:
                try:
                    cont = input(
                        "Press Enter to continue, 'q' to quit: ").lower()
                    if cont == 'q':
                        break
                except KeyboardInterrupt:
                    break

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Mario Model Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable environment rendering")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of episodes to run")

    args = parser.parse_args()

    # Render unless explicitly disabled
    render = not args.no_render

    continuous_evaluation(
        args.model_path,
        render=render,
        max_episodes=args.max_episodes
    )


if __name__ == "__main__":
    main()
