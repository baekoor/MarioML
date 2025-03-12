import os
import argparse
import torch
import numpy as np
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit

from network import DuelingDQN, arrange
from wrappers import wrap_mario


def create_env(env_id='SuperMarioBros-v2', use_custom_rewards=True):
    """Create a Mario environment with specified settings"""
    env = gym_super_mario_bros.make(env_id)

    # Remove existing TimeLimit wrapper
    original_env = env
    while hasattr(env, 'env'):
        if isinstance(env, TimeLimit):
            env = env.env  # Unwrap the TimeLimit
        else:
            env = env.env  # Continue unwrapping

    # Reset back to original but without TimeLimit
    env = original_env
    while hasattr(env, 'env'):
        if isinstance(env, TimeLimit):
            env = env.env  # Remove TimeLimit
        else:
            if hasattr(env, 'env') and not isinstance(env.env, TimeLimit):
                env = env.env  # Unwrap further
            else:
                break

    # Apply new TimeLimit with much higher steps
    env = TimeLimit(env, max_episode_steps=25000)

    # Continue with other wrappers
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env, use_custom_rewards=use_custom_rewards)

    return env


def evaluate_model(model_path, env_id='SuperMarioBros-v2', num_episodes=5, use_custom_rewards=True, seed=None):
    """Evaluate a trained model by rendering gameplay"""
    # Set up device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create environment
    env = create_env(env_id, use_custom_rewards)

    # Set seed if provided
    if seed is not None:
        env.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Get action space size
    n_actions = env.action_space.n
    n_frame = 4  # Number of frames stacked

    # Load model
    print(f"Loading model from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Create and load network
        q_network = DuelingDQN(n_frame, n_actions, device).to(device)
        q_network.load_state_dict(checkpoint['q_network_state_dict'])
        q_network.eval()

        print("Model loaded successfully")
        if 'generation' in checkpoint:
            print(f"Model from generation: {checkpoint['generation']}")
        if 'fitness' in checkpoint:
            print(f"Model fitness: {checkpoint['fitness']}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run evaluation episodes
    for episode in range(num_episodes):
        print(
            f"\n=== Starting evaluation episode {episode+1}/{num_episodes} ===")

        state = env.reset()
        state = arrange(state)

        done = False
        total_reward = 0
        max_distance = 0
        steps = 0

        while not done:
            # Render environment
            env.render()

            # Select action using the network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = q_network(state_tensor).argmax(1).item()

            # Execute action
            next_state, reward, done, info = env.step(action)
            next_state = arrange(next_state)

            # Update tracking
            current_distance = info.get('x_pos', 0)
            max_distance = max(max_distance, current_distance)
            state = next_state
            total_reward += reward
            steps += 1

            # Print status periodically
            if steps % 100 == 0:
                print(
                    f"Step {steps}, Distance: {current_distance}, Reward: {total_reward:.2f}")

        # Episode complete
        print(f"Episode {episode+1} complete:")
        print(f"  Steps: {steps}")
        print(f"  Score: {total_reward:.2f}")
        print(f"  Max Distance: {max_distance}")

    # Clean up
    env.close()
    print("\nEvaluation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Mario AI model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the saved model")
    parser.add_argument("--env-id", type=str,
                        default="SuperMarioBros-v2", help="Environment ID")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to evaluate")
    parser.add_argument("--no-custom-rewards", action="store_true",
                        help="Disable custom reward wrapper")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        env_id=args.env_id,
        num_episodes=args.episodes,
        use_custom_rewards=not args.no_custom_rewards,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
