# Optimized MarioTrainer.py

from gym.wrappers import TimeLimit
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


def __init__(self,
             env_id='SuperMarioBros-v3',
             run_name='mario_run',
             checkpoint_dir='checkpoints',
             models_dir='models',
             use_custom_rewards=True,
             device=None,
             memory_size=100000,
             batch_size=128,
             gamma=0.99,
             eps_start=1.0,
             eps_end=0.15,
             eps_decay=50000,
             target_update=1000,
             vram_gb=10):  # New parameter for VRAM allocation

    # Set up device
    self.device = device if device else (
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {self.device}")

    if self.device == 'cuda':
        # Configure for better CUDA performance
        torch.backends.cudnn.benchmark = True
        print("CUDNN benchmark enabled for faster training")

        # Hard allocate GPU memory
        self.cache_tensor = self.reserve_gpu_memory(vram_gb)

        # Set PyTorch memory allocator options for better efficiency
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Create a cache tensor to occupy space
        try:
            # Use a large contiguous block of memory (using float16 to double the size)
            print(
                f"Allocating {bytes_to_allocate / (1024**3):.2f}GB of VRAM...")
            cache_tensor = torch.zeros(bytes_to_allocate // 2,
                                       dtype=torch.float16,
                                       device=device)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"Successfully allocated {allocated:.2f}GB of VRAM")
            return cache_tensor
        except RuntimeError as e:
            print(f"Failed to allocate memory: {e}")
            print("Trying with a smaller allocation...")
            # Try with half the requested amount
            bytes_to_allocate = bytes_to_allocate // 2
            cache_tensor = torch.zeros(bytes_to_allocate // 2,
                                       dtype=torch.float16,
                                       device=device)
            allocated = torch.cuda.memory_allocated() / (1024**3)
            print(f"Successfully allocated {allocated:.2f}GB of VRAM")
            return cache_tensor
    else:
        print("CUDA not available. No VRAM allocated.")
        return None


class MarioTrainer:
    def __init__(self,
                 env_id='SuperMarioBros-v3',
                 run_name='mario_run',
                 checkpoint_dir='checkpoints',
                 models_dir='models',
                 use_custom_rewards=True,
                 device=None,
                 memory_size=100000,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=1.0,
                 eps_end=0.15,
                 eps_decay=50000,
                 target_update=1000,
                 vram_gb=9):  # New parameter for VRAM allocation

        # Set up device
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        if self.device == 'cuda':
            # Configure for better CUDA performance
            torch.backends.cudnn.benchmark = True
            print("CUDNN benchmark enabled for faster training")

            # Hard allocate GPU memory
            self.cache_tensor = self.reserve_gpu_memory(vram_gb)

            # Set PyTorch memory allocator options for better efficiency
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'memory_stats'):
                print(
                    f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        # Set up directories
        self.checkpoint_dir = checkpoint_dir
        self.models_dir = models_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Set up environment
        self.env = gym_super_mario_bros.make(env_id)

        # Check if there's a TimeLimit wrapper and remove it
        original_env = self.env
        while hasattr(self.env, 'env'):
            if isinstance(self.env, TimeLimit):
                print(
                    f"Found TimeLimit wrapper with max_steps: {self.env._max_episode_steps}")
                self.env = self.env.env  # Unwrap the TimeLimit
                print("Removed original TimeLimit wrapper")
            else:
                self.env = self.env.env  # Continue unwrapping

        # Reset back to original but without TimeLimit
        self.env = original_env
        while hasattr(self.env, 'env'):
            if isinstance(self.env, TimeLimit):
                self.env = self.env.env  # Remove TimeLimit
            else:
                if hasattr(self.env, 'env') and not isinstance(self.env.env, TimeLimit):
                    self.env = self.env.env  # Unwrap further
                else:
                    break

        # Apply a new TimeLimit with much higher steps
        # 100k steps limit
        self.env = TimeLimit(self.env, max_episode_steps=100000)
        print(f"Applied new TimeLimit with max_episode_steps=100000")

        # Continue with other wrappers
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
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.memory_size = memory_size
        self.run_name = run_name

        # Initialize replay memory more efficiently
        self.memory = deque(maxlen=self.memory_size)

        # Training metrics
        self.steps_done = 0
        self.episode = 0
        self.scores = []
        self.avg_scores = []
        self.max_distances = []
        self.best_score = 0
        self.distance_log = []  # Properly initialize the distance log

        # Set up mixed precision training if available
        self.scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    def reserve_gpu_memory(self, gb_to_allocate=10):
        """
        Hard allocate a specific amount of GPU memory to prevent other processes from using it
        and to ensure consistent performance during training.

        Args:
            gb_to_allocate (float): Amount of GPU memory to allocate in gigabytes

        Returns:
            cache_tensor: A reference to the allocated tensor (keep this to prevent garbage collection)
        """
        # Convert GB to bytes
        bytes_to_allocate = int(gb_to_allocate * 1024 * 1024 * 1024)

        # Check available GPU memory
        if torch.cuda.is_available():
            device = torch.device('cuda')
            total_memory = torch.cuda.get_device_properties(0).total_memory

            # Check if we're trying to allocate more than available
            if bytes_to_allocate > total_memory * 0.95:  # Leave 5% for overhead
                print(f"Warning: Attempting to allocate {gb_to_allocate}GB but only "
                      f"{total_memory / (1024**3):.2f}GB total VRAM available.")
                # Adjust to 90% of available memory
                bytes_to_allocate = int(total_memory * 0.9)
                print(
                    f"Adjusting allocation to {bytes_to_allocate / (1024**3):.2f}GB")

            # Create a cache tensor to occupy space
            try:
                # Use a large contiguous block of memory (using float16 to double the size)
                print(
                    f"Allocating {bytes_to_allocate / (1024**3):.2f}GB of VRAM...")
                cache_tensor = torch.zeros(bytes_to_allocate // 2,
                                           dtype=torch.float16,
                                           device=device)
                allocated = torch.cuda.memory_allocated() / (1024**3)
                print(f"Successfully allocated {allocated:.2f}GB of VRAM")
                return cache_tensor
            except RuntimeError as e:
                print(f"Failed to allocate memory: {e}")
                print("Trying with a smaller allocation...")
                # Try with half the requested amount
                bytes_to_allocate = bytes_to_allocate // 2
                cache_tensor = torch.zeros(bytes_to_allocate // 2,
                                           dtype=torch.float16,
                                           device=device)
                allocated = torch.cuda.memory_allocated() / (1024**3)
                print(f"Successfully allocated {allocated:.2f}GB of VRAM")
                return cache_tensor
        else:
            print("CUDA not available. No VRAM allocated.")
            return None

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        if training:
            self.steps_done += 1

        # Calculate epsilon threshold only if in training mode
        if training:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1. * self.steps_done / self.eps_decay)

            # Use random sampling for exploration
            if random.random() < eps_threshold:
                return random.randrange(self.n_actions)

        # Use network for exploitation (or evaluation)
        with torch.no_grad():
            # Convert state to tensor and get action in a single operation
            state_tensor = torch.FloatTensor(state).to(self.device)
            return self.q_network(state_tensor).argmax(1).item()

    def push_memory(self, state, action, next_state, reward, done):
        """Push transition to replay memory"""
        self.memory.append((state, action, next_state, reward, done))

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from memory
        transitions = random.sample(self.memory, self.batch_size)

        # Process batch once and efficiently
        batch = list(zip(*transitions))

        # Convert to tensors with proper memory reuse
        with torch.no_grad():
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

        # Compute Q values with mixed precision where available
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Get current Q values (only for taken actions)
                current_q_values = self.q_network(states).gather(1, actions)

                # Compute next state values using target network
                with torch.no_grad():  # Ensure no gradients flow through target network
                    next_q_values = self.target_network(
                        next_states).max(1, keepdim=True)[0]

                # Compute expected Q values
                target_q_values = rewards + \
                    (self.gamma * next_q_values * (~dones))

                # Compute loss
                loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize with mixed precision
            # More efficient than .zero_grad()
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Gradient clipping to prevent exploding gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), max_norm=10.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path
            current_q_values = self.q_network(states).gather(1, actions)

            with torch.no_grad():
                next_q_values = self.target_network(
                    next_states).max(1, keepdim=True)[0]

            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()

        # Update target network if needed
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Periodically clear CUDA cache on update
            if self.device == 'cuda' and self.steps_done % (self.target_update * 10) == 0:
                torch.cuda.empty_cache()

    def train(self, num_episodes=1000, render=False, save_interval=10):
        """Train the agent"""
        print("Starting training loop...")
        episode_distance_logs = []  # Store distance logs for all episodes

        for episode in range(1, num_episodes + 1):
            self.episode = episode
            print(f"Starting episode {episode}...")

            # Reset environment
            state = self.env.reset()
            state = arrange(state)

            done = False
            total_reward = 0
            max_distance = 0
            steps = 0

            # Clear distance log for this episode
            self.distance_log = []

            # Run episode
            while not done:
                if render:
                    self.env.render()

                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = arrange(next_state)

                # Update max distance
                current_distance = info.get('x_pos', 0)
                max_distance = max(max_distance, current_distance)

                # Track distance properly
                self.distance_log.append((steps, current_distance))

                # Save transition to memory
                self.push_memory(state, action, next_state, reward, done)

                # Update state
                state = next_state

                # Perform optimization step (less frequently for efficiency)
                if steps % 4 == 0:
                    self.optimize_model()

                total_reward += reward
                steps += 1

                # Print periodic status during long episodes (less frequently)
                if steps % 2500 == 0:
                    print(
                        f"Episode {episode} in progress: {steps} steps, current distance: {current_distance}, current reward: {total_reward:.2f}")

                # More efficient memory management
                if steps % 10000 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

            # Store the episode's distance log
            episode_distance_logs.append(self.distance_log)
            if len(episode_distance_logs) > 5:  # Keep only recent episodes
                episode_distance_logs.pop(0)

            # Record metrics
            self.scores.append(total_reward)
            self.max_distances.append(max_distance)

            # Print distance progression more efficiently
            if self.distance_log:
                start_distance = self.distance_log[0][1]
                end_distance = self.distance_log[-1][1]
                print(
                    f"Distance progression: Start={start_distance}, End={end_distance}, Max={max_distance}")

            # Calculate average score efficiently
            if len(self.scores) >= 100:
                avg_score = sum(self.scores[-100:]) / 100
            else:
                avg_score = sum(self.scores) / len(self.scores)

            self.avg_scores.append(avg_score)

            # Print episode info
            print(f"Episode {episode} - "
                  f"Score: {total_reward:.2f}, "
                  f"Avg Score: {avg_score:.2f}, "
                  f"Max Distance: {max_distance}, "
                  f"Steps: {steps}, "
                  f"Memory: {len(self.memory)}")

            # Save checkpoint (less frequently for better I/O performance)
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

        # Prepare checkpoint data efficiently
        checkpoint = {
            'episode': self.episode,
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'scores': self.scores,
            'max_distances': self.max_distances,
            'timestamp': datetime.now().isoformat()
        }

        # Save checkpoint with better error handling
        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def save_best_model(self):
        """Save best model"""
        model_path = os.path.join(self.models_dir, f"{self.run_name}_best.pth")

        # Only save necessary data for inference
        try:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'best_score': self.best_score
            }, model_path)
            print(f"Best model saved to {model_path}")
        except Exception as e:
            print(f"Error saving best model: {e}")

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

            # Update best score more efficiently
            if self.scores:
                if len(self.scores) >= 100:
                    avg_score = sum(self.scores[-100:]) / 100
                else:
                    avg_score = sum(self.scores) / len(self.scores)

                self.best_score = max(avg_score, self.best_score)

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

        # Plot distance progression for the last episode if available
        if self.distance_log:
            plt.figure(figsize=(10, 6))
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
                        default="SuperMarioBros-v2", help="Environment ID")
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
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--resume", type=str,
                        help="Resume training from checkpoint")
    parser.add_argument("--no-custom-rewards", action="store_true",
                        help="Disable custom reward wrapper")
    parser.add_argument("--memory-size", type=int, default=100000,
                        help="Size of replay memory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--vram-gb", type=float, default=10.0,
                        help="GB of VRAM to allocate (default: 10GB)")

    args = parser.parse_args()

    # Create trainer with additional configurable parameters
    trainer = MarioTrainer(
        env_id=args.env_id,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        models_dir=args.models_dir,
        use_custom_rewards=not args.no_custom_rewards,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        vram_gb=args.vram_gb  # Pass the VRAM allocation parameter
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
