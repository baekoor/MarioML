import os
import sys
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from network import DuelingDQN, arrange
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


def select_action(q_network, state, device, exploration_rate=0.0):
    """
    Intelligent action selection with optional exploration
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
            # Get top actions
            top_actions = q_values.topk(3)

            # Either pick best action or do weighted random selection
            if random.random() < 0.8:  # 80% of the time pick the best action
                action = top_actions.indices[0][0].item()
            else:  # 20% of the time choose from top 3 with probability weights
                probs = F.softmax(top_actions.values[0], dim=0)
                action = top_actions.indices[0][torch.multinomial(
                    probs, 1)].item()

        return action, q_values.cpu().numpy()[0]


def evaluate_model(model_path, render=True, max_episodes=None, exploration_rate=0.05, delay=0.02,
                   save_stats=False, stats_dir="evaluation_stats"):
    """Evaluate the model with detailed statistics"""
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create stats directory if saving stats
    if save_stats:
        os.makedirs(stats_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_filename = os.path.join(stats_dir, f"eval_stats_{timestamp}.txt")
        stats_file = open(stats_filename, "w")
        stats_file.write(f"Evaluation of model: {model_path}\n")
        stats_file.write(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Load model
    q_network, env = load_model(model_path, device)

    if q_network is None:
        print("Failed to load model. Exiting.")
        return

    # Stats collection across all episodes
    all_rewards = []
    all_distances = []
    all_steps = []
    all_action_counts = defaultdict(int)

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
            action_counts = defaultdict(int)

            # Track Q-values for visualization
            action_q_values = defaultdict(list)
            positions = []
            rewards = []

            # Single episode evaluation
            while not done:
                if render:
                    env.render()
                    time.sleep(delay)  # Delay for rendering visibility

                # Select action with minimal exploration
                action, q_vals = select_action(
                    q_network, state, device, exploration_rate)

                # Track action distribution and Q-values
                action_counts[action] += 1
                all_action_counts[action] += 1
                for a, q in enumerate(q_vals):
                    action_q_values[a].append(q)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Track metrics
                x_pos = info.get('x_pos', 0)
                positions.append(x_pos)
                rewards.append(reward)
                max_distance = max(max_distance, x_pos)

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
            print(f"\nEpisode {episode_count}:")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Max Distance: {max_distance}")

            # Print action distribution
            print("\nAction Distribution:")
            for action, count in sorted(action_counts.items()):
                action_name = COMPLEX_MOVEMENT[action]
                percentage = (count / step_count) * 100
                print(
                    f"  Action {action} ({action_name}): {count} times ({percentage:.1f}%)")

            # Save stats if requested
            if save_stats:
                stats_file.write(f"\n--- Episode {episode_count} ---\n")
                stats_file.write(f"Reward: {total_reward:.2f}\n")
                stats_file.write(f"Steps: {step_count}\n")
                stats_file.write(f"Max Distance: {max_distance}\n")
                stats_file.write("Action Distribution:\n")
                for action, count in sorted(action_counts.items()):
                    action_name = COMPLEX_MOVEMENT[action]
                    percentage = (count / step_count) * 100
                    stats_file.write(
                        f"  Action {action} ({action_name}): {count} times ({percentage:.1f}%)\n")

            # Collect stats for all episodes
            all_rewards.append(total_reward)
            all_distances.append(max_distance)
            all_steps.append(step_count)

            # Visualize episode data
            if save_stats:
                # Create plots directory
                plots_dir = os.path.join(stats_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)

                # Create plots
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

                # Plot positions over time
                ax1.plot(positions)
                ax1.set_title(f'Mario Position - Episode {episode_count}')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('X Position')
                ax1.grid(True, alpha=0.3)

                # Plot rewards over time
                ax2.plot(rewards)
                ax2.set_title('Rewards')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('Reward')
                ax2.grid(True, alpha=0.3)

                # Plot Q-values for each action
                for action, values in action_q_values.items():
                    if len(values) > 0:  # Only plot if we have values
                        ax3.plot(
                            values, label=f'Action {action}: {COMPLEX_MOVEMENT[action]}')
                ax3.set_title('Q-Values by Action')
                ax3.set_xlabel('Step')
                ax3.set_ylabel('Q-Value')
                ax3.legend(loc='best')
                ax3.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(os.path.join(
                    plots_dir, f"episode_{episode_count}.png"))
                plt.close()

            # Optional: ask to continue if max_episodes is not set
            if max_episodes is None:
                try:
                    cont = input(
                        "\nPress Enter to continue, 'q' to quit: ").lower()
                    if cont == 'q':
                        break
                except KeyboardInterrupt:
                    break

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()

        # Print overall statistics
        if episode_count > 0:
            print("\n===== Overall Statistics =====")
            print(f"Episodes: {episode_count}")
            print(f"Avg Reward: {np.mean(all_rewards):.2f}")
            print(f"Avg Distance: {np.mean(all_distances):.2f}")
            print(f"Max Distance: {np.max(all_distances):.2f}")
            print(f"Avg Steps: {np.mean(all_steps):.2f}")

            print("\nOverall Action Distribution:")
            total_steps = sum(all_steps)
            for action, count in sorted(all_action_counts.items()):
                action_name = COMPLEX_MOVEMENT[action]
                percentage = (count / total_steps) * 100
                print(
                    f"  Action {action} ({action_name}): {count} times ({percentage:.1f}%)")

            if save_stats:
                stats_file.write("\n===== Overall Statistics =====\n")
                stats_file.write(f"Episodes: {episode_count}\n")
                stats_file.write(f"Avg Reward: {np.mean(all_rewards):.2f}\n")
                stats_file.write(
                    f"Avg Distance: {np.mean(all_distances):.2f}\n")
                stats_file.write(
                    f"Max Distance: {np.max(all_distances):.2f}\n")
                stats_file.write(f"Avg Steps: {np.mean(all_steps):.2f}\n")

                stats_file.write("\nOverall Action Distribution:\n")
                for action, count in sorted(all_action_counts.items()):
                    action_name = COMPLEX_MOVEMENT[action]
                    percentage = (count / total_steps) * 100
                    stats_file.write(
                        f"  Action {action} ({action_name}): {count} times ({percentage:.1f}%)\n")

                stats_file.close()
                print(f"\nDetailed statistics saved to {stats_filename}")


def main():
    parser = argparse.ArgumentParser(description="Mario Model Evaluator")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model file")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable environment rendering")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of episodes to run")
    parser.add_argument("--exploration-rate", type=float, default=0.05,
                        help="Exploration rate (0.0 for purely greedy, higher for more exploration)")
    parser.add_argument("--delay", type=float, default=0.02,
                        help="Delay between frames (seconds)")
    parser.add_argument("--save-stats", action="store_true",
                        help="Save evaluation statistics")
    parser.add_argument("--stats-dir", type=str, default="evaluation_stats",
                        help="Directory to save evaluation statistics")

    args = parser.parse_args()

    # Render unless explicitly disabled
    render = not args.no_render

    evaluate_model(
        model_path=args.model_path,
        render=render,
        max_episodes=args.max_episodes,
        exploration_rate=args.exploration_rate,
        delay=args.delay,
        save_stats=args.save_stats,
        stats_dir=args.stats_dir
    )


if __name__ == "__main__":
    main()
