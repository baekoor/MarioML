import os
import sys
import time
import random
import argparse
from datetime import datetime
from collections import deque
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from network import DuelingDQN, arrange
from wrappers import wrap_mario


class MarioTrainer:
    def __init__(self,
                 env_id='SuperMarioBros-v3',
                 run_name='mario_run',
                 checkpoint_dir='checkpoints',
                 models_dir='models',
                 use_custom_rewards=True,
                 device=None,
                 gpu_memory_fraction=0.8):

        # Set up device
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device == 'cuda':
            # More precise GPU memory management
            import os
            # Set environment variable to limit memory growth
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

            # Optional: Use a fraction of available memory instead of a fixed size
            memory_fraction = 0.8  # Use 80% of available GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = int(total_memory * memory_fraction)

            print(f"Limiting GPU memory usage to {allocated_memory / (1024**3):.2f} GB "
                  f"({memory_fraction*100:.0f}% of available {total_memory / (1024**3):.2f} GB)")

            # Create a cache tensor to occupy space and prevent other processes from using it
            cache_tensor = torch.zeros((allocated_memory // 4,),
                                       dtype=torch.float32,
                                       device='cuda')
            # Keep a reference to the tensor to prevent it from being garbage collected
            self.cache_tensor = cache_tensor

            torch.backends.cudnn.benchmark = True
            print("CUDNN benchmark enabled for faster training")

        # Set up directories
        self.checkpoint_dir = checkpoint_dir
        self.models_dir = models_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Set up environment
        self.env = gym_super_mario_bros.make(env_id)
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        self.env = wrap_mario(self.env, use_custom_rewards=use_custom_rewards)

        # Print action space info
        self.n_actions = self.env.action_space.n
        print(f"Action space size: {self.n_actions}")
        print(f"Available actions: {COMPLEX_MOVEMENT}")

        # Set up networks
        self.n_frame = 4  # Number of frames stacked
        self.q_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)
        self.target_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Initialize weights
        self.q_network.init_weights()

        # Set up optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.00025)

        # Training parameters
        self.batch_size = 128
        self.gamma = 0.99
        self.eps_start = 1.0
        self.eps_end = 0.15
        self.eps_decay = 50000
        self.target_update = 1000  # Update target network every N steps
        self.memory_size = 100000
        self.run_name = run_name

        # Initialize replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Training metrics
        self.steps_done = 0
        self.episode = 0
        self.scores = []
        self.avg_scores = []
        self.max_distances = []
        self.best_score = 0

        # Set up mixed precision training if available
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)

        if training:
            self.steps_done += 1

        if sample > eps_threshold or not training:
            with torch.no_grad():
                # Use Q-network to select best action
                state_tensor = torch.FloatTensor(state).to(self.device)
                actions = self.q_network(state_tensor)
                return actions.max(1)[1].view(1, 1).item()
        else:
            # Random action
            return random.randrange(self.n_actions)

    def push_memory(self, state, action, next_state, reward, done):
        """Push transition to replay memory"""
        self.memory.append((state, action, next_state, reward, done))

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        transitions = random.sample(self.memory, self.batch_size)

        # Process batches more efficiently
        with torch.no_grad():  # Don't track gradients when preparing batches
            # Transpose batch
            batch = list(zip(*transitions))

            # Convert to tensors in a memory-efficient way
            states = torch.FloatTensor(
                np.array(batch[0], copy=False)).to(self.device)
            actions = torch.LongTensor(
                np.array(batch[1], copy=False)).view(-1, 1).to(self.device)
            next_states = torch.FloatTensor(
                np.array(batch[2], copy=False)).to(self.device)
            rewards = torch.FloatTensor(
                np.array(batch[3], copy=False)).view(-1, 1).to(self.device)
            dones = torch.BoolTensor(
                np.array(batch[4], copy=False)).view(-1, 1).to(self.device)

        # Transpose batch (see https://stackoverflow.com/a/19343/3343043)
        batch = list(zip(*transitions))

        # Convert to tensors
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(
            np.array(batch[1])).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[2])).to(self.device)
        rewards = torch.FloatTensor(
            np.array(batch[3])).view(-1, 1).to(self.device)
        dones = torch.BoolTensor(
            np.array(batch[4])).view(-1, 1).to(self.device)

        # Compute Q values
        if self.scaler:
            with torch.cuda.amp.autocast():
                current_q_values = self.q_network(states).gather(1, actions)
                next_q_values = self.target_network(next_states).max(1)[
                    0].detach().unsqueeze(1)
                expected_q_values = rewards + \
                    (self.gamma * next_q_values * (~dones))

                # Compute loss
                loss = F.smooth_l1_loss(current_q_values, expected_q_values)

            # Optimize with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path
            current_q_values = self.q_network(states).gather(1, actions)
            next_q_values = self.target_network(next_states).max(1)[
                0].detach().unsqueeze(1)
            expected_q_values = rewards + \
                (self.gamma * next_q_values * (~dones))

            # Compute loss
            loss = F.smooth_l1_loss(current_q_values, expected_q_values)

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update target network if needed
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, num_episodes=1000, render=False, save_interval=10):
        """Train the agent"""
        print("Starting training loop...")

        for episode in range(1, num_episodes + 1):
            self.episode = episode
            print(f"Starting episode {episode}...")

            # Reset environment
            print("Resetting environment...")
            start_time = time.time()
            state = self.env.reset()
            state = arrange(state)
            print(
                f"Environment reset took {time.time() - start_time:.2f} seconds")

            done = False
            total_reward = 0
            max_distance = 0
            steps = 0

            # Run episode
            print("Running episode...")
            episode_start_time = time.time()

            # Run episode
            while not done:
                if render:
                    self.env.render()

                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = arrange(next_state)

                # Update max distance
                max_distance = max(max_distance, info.get('x_pos', 0))

                distance_log = []  # To track distance over time

                distance_log.append((steps, info.get('x_pos', 0)))

                # Save transition to memory
                self.push_memory(state, action, next_state, reward, done)

                # Update state
                state = next_state

                # Perform optimization step
                if steps % 4 == 0:
                    optimize_start = time.time()
                    self.optimize_model()
                    if steps % 100 == 0:
                        print(
                            f"Optimization step took {time.time() - optimize_start:.4f} seconds")

                total_reward += reward
                steps += 1

                # Print periodic status during long episodes
                if steps % 500 == 0:
                    print(
                        f"Episode {episode} in progress: {steps} steps, current distance: {info.get('x_pos', 0)}, current reward: {total_reward:.2f}")

                if steps % 10000 == 0:
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

            # Record metrics
            self.scores.append(total_reward)
            self.max_distances.append(max_distance)

            if len(distance_log) > 1:
                print(
                    f"Distance progression: Start={distance_log[0][1]}, End={distance_log[-1][1]}, Max={max_distance}")
                # Optionally print intermediate checkpoints at 25%, 50%, 75% of the episode
                if len(distance_log) >= 4:
                    quarter_idx = len(distance_log) // 4
                    print(
                        f"Distance checkpoints: 25%={distance_log[quarter_idx][1]}, 50%={distance_log[2*quarter_idx][1]}, 75%={distance_log[3*quarter_idx][1]}")

            # Calculate average score
            avg_score = np.mean(
                self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores)
            self.avg_scores.append(avg_score)

            # Print episode info
            print(f"Episode {episode} - "
                  f"Score: {total_reward:.2f}, "
                  f"Avg Score: {avg_score:.2f}, "
                  f"Max Distance: {max_distance}, "
                  f"Steps: {steps}, "
                  f"Epsilon: {self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * self.steps_done / self.eps_decay):.2f}, "
                  f"Memory: {len(self.memory)}")

            # Save checkpoint
            if episode % save_interval == 0:
                self.save_checkpoint()

            # Save best model
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.save_best_model()
                print(
                    f"New best model saved with average score: {self.best_score:.2f}")

        # Close environment
        self.env.close()

        # Save final model
        self.save_checkpoint(final=True)
        print("Training complete!")

    def save_checkpoint(self, final=False):
        """Save checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f"{self.run_name}_episode_{self.episode}.pth" if not final else f"{self.run_name}_final.pth"
        )

        torch.save({
            'episode': self.episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'scores': self.scores,
            'max_distances': self.max_distances,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)

        print(f"Checkpoint saved to {checkpoint_path}")

    def save_best_model(self):
        """Save best model"""
        model_path = os.path.join(self.models_dir, f"{self.run_name}_best.pth")

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'best_score': self.best_score
        }, model_path)

        print(f"Best model saved to {model_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(
                checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.episode = checkpoint.get('episode', 0)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.scores = checkpoint.get('scores', [])
            self.max_distances = checkpoint.get('max_distances', [])

            if self.scores:
                self.best_score = max(np.mean(
                    self.scores[-100:]) if len(self.scores) >= 100 else np.mean(self.scores), self.best_score)

            print(f"Checkpoint loaded from {checkpoint_path}")
            print(
                f"Resuming from episode {self.episode}, steps done {self.steps_done}")

            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def plot_progress(self, save_path=None):
        """Plot training progress"""
        plt.figure(figsize=(15, 10))

        # Plot scores
        plt.subplot(2, 1, 1)
        plt.plot(self.scores, label='Score')
        if len(self.avg_scores) > 0:
            plt.plot(self.avg_scores,
                     label='Avg Score (100 episodes)', color='red')
        plt.title('Training Scores')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot distances
        plt.subplot(2, 1, 2)
        plt.plot(self.max_distances)
        plt.title('Max Distances')
        plt.xlabel('Episode')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if hasattr(self, 'distance_log') and self.distance_log:
            plt.subplot(2, 2, 3)
            steps, distances = zip(*self.distance_log)
            plt.plot(steps, distances)
            plt.title('Distance Progression (Latest Episode)')
            plt.xlabel('Steps')
            plt.ylabel('Distance')
            plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Progress plot saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train Mario AI with DQN")
    parser.add_argument("--env-id", type=str,
                        default="SuperMarioBros-v3", help="Environment ID")
    parser.add_argument("--run-name", type=str,
                        default="mario_run", help="Name of this run")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--models-dir", type=str,
                        default="models", help="Directory to save best models")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--resume", type=str,
                        help="Resume training from checkpoint")
    parser.add_argument("--no-custom-rewards", action="store_true",
                        help="Disable custom reward wrapper")

    args = parser.parse_args()

    # Create trainer
    trainer = MarioTrainer(
        env_id=args.env_id,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        models_dir=args.models_dir,
        use_custom_rewards=not args.no_custom_rewards
    )

    # Resume training if requested
    if args.resume:
        if not trainer.load_checkpoint(args.resume):
            print("Failed to load checkpoint. Starting from scratch.")

    # Train
    trainer.train(
        num_episodes=args.episodes,
        render=args.render,
        save_interval=args.save_interval
    )

    # Plot and save progress
    trainer.plot_progress(save_path=f"{args.run_name}_progress.png")


if __name__ == "__main__":
    main()
