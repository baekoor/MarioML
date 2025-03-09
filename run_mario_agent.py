"""
Run a trained Mario agent from a checkpoint
"""
import os
import time
import torch
import argparse
from pathlib import Path

# Import needed components from your mario_dqn.py
from mario_dqn import (
    device, make_env, DuelingDQN, MarioAgent,
    EnhancedMarioEnv, evaluate_agent
)


def find_latest_checkpoint(checkpoint_dir=None):
    """Finds the latest checkpoint in the specified directory or all subdirectories"""
    if checkpoint_dir is None:
        # Look in mario_checkpoints directory
        base_dir = Path("mario_checkpoints")
        if not base_dir.exists():
            print("No checkpoints directory found.")
            return None

        # Find all checkpoint directories
        checkpoint_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if not checkpoint_dirs:
            print("No checkpoint directories found.")
            return None

        # Sort directories by creation time (most recent first)
        checkpoint_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        checkpoint_dir = checkpoint_dirs[0]
    else:
        checkpoint_dir = Path(checkpoint_dir)

    # Find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("mario_model_*.pt"))
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        return None

    # Sort by step number (extract from filename)
    def extract_step(filename):
        try:
            # Try to extract step number from 'mario_model_step_XXXXX_ep_YY.pt'
            parts = filename.stem.split('_')
            if 'step' in parts:
                step_idx = parts.index('step') + 1
                if step_idx < len(parts):
                    return int(parts[step_idx])
            # Fallback to just using the file modification time
            return int(filename.stat().st_mtime)
        except (ValueError, IndexError):
            # If anything goes wrong, use file modification time
            return int(filename.stat().st_mtime)

    checkpoint_files.sort(key=extract_step, reverse=True)

    # Return the highest step checkpoint
    latest_checkpoint = checkpoint_files[0]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    # Convert to string to avoid Path object issues
    return str(latest_checkpoint)


def run_agent(checkpoint_path=None, num_episodes=5, render=True, action_type="simple", stage="1-1"):
    """Run the agent from a checkpoint without training"""
    print(f"Using device: {device}")

    # Find latest checkpoint if not specified
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoint found. Please train an agent first.")
            return
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return

    print(f"Loading checkpoint: {checkpoint_path}")

    # Create temporary save directory (needed for agent initialization)
    temp_save_dir = Path("temp_run_dir")
    temp_save_dir.mkdir(exist_ok=True)

    # Create environment
    env, action_count = make_env(action_type, stage)
    wrapped_env = EnhancedMarioEnv(env)

    # Create agent
    agent = MarioAgent(
        input_channels=4,
        action_count=action_count,
        save_dir=temp_save_dir,
        batch_size=64,
        memory_size=10000,  # Smaller memory size for evaluation
    )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Set to very low exploration for demonstration
    agent.exploration_rate = 0.01

    total_reward = 0
    wins = 0
    max_distance = 0

    print(f"\n{'='*50}")
    print(f"Running Mario agent for {num_episodes} episodes")
    print(f"{'='*50}")

    for episode in range(num_episodes):
        state = wrapped_env.reset()
        done = False
        episode_reward = 0
        steps = 0
        start_time = time.time()

        print(f"\nEpisode {episode+1}/{num_episodes}")
        print(f"{'-'*30}")

        # Run episode
        while not done:
            # Select action with minimal exploration
            action = agent.act(state)

            # Take action
            next_state, reward, done, info = wrapped_env.step(action)

            # Render if enabled
            if render:
                wrapped_env.render()
                time.sleep(0.01)  # Slow down rendering for better viewing

            # Update state and counters
            state = next_state
            episode_reward += reward
            steps += 1

            # Print progress every 100 steps
            if steps % 100 == 0:
                print(
                    f"Step {steps}, Position: {info.get('x_pos', 0)}, Score: {info.get('score', 0)}")

        # Episode complete
        duration = time.time() - start_time
        x_pos = info.get('x_pos', 0)
        flag_get = info.get('flag_get', False)

        # Update stats
        total_reward += episode_reward
        if flag_get:
            wins += 1
        max_distance = max(max_distance, x_pos)

        print(f"Episode complete:")
        print(f"- Reward: {episode_reward:.2f}")
        print(f"- Distance: {x_pos}")
        print(f"- Steps: {steps}")
        print(f"- Duration: {duration:.2f}s")
        print(f"- Result: {'WIN!' if flag_get else 'Loss'}")

    # Final stats
    print(f"\n{'='*50}")
    print(f"Run complete!")
    print(f"- Total episodes: {num_episodes}")
    print(f"- Average reward: {total_reward/num_episodes:.2f}")
    print(f"- Win rate: {wins/num_episodes*100:.1f}%")
    print(f"- Max distance: {max_distance}")
    print(f"{'='*50}")

    # Clean up
    wrapped_env.close()

    # Remove temporary directory
    for file in temp_save_dir.glob("*"):
        try:
            file.unlink()
        except:
            pass
    try:
        temp_save_dir.rmdir()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a trained Mario agent')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--action-type', type=str, default='simple',
                        choices=['simple', 'right_only', 'custom'],
                        help='Action space to use')
    parser.add_argument('--stage', type=str, default='1-1',
                        help='Stage to play (e.g., 1-1, 1-2, etc.)')

    args = parser.parse_args()

    run_agent(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
        action_type=args.action_type,
        stage=args.stage
    )
