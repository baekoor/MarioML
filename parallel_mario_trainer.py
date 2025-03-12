import os
import sys
import time
import random
import argparse
import multiprocessing as mp
from datetime import datetime
from collections import deque
import json
import copy
import threading
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import TimeLimit

from network import DuelingDQN, arrange
from wrappers import wrap_mario


class ParallelMarioTrainer:
    def __init__(self,
                 env_id='SuperMarioBros-v2',
                 run_name='mario_parallel',
                 checkpoint_dir='checkpoints',
                 models_dir='models',
                 n_agents=16,
                 use_custom_rewards=True,
                 device=None,
                 memory_size=100000,
                 batch_size=128,
                 gamma=0.99,
                 eps_start=1.0,
                 eps_end=0.15,
                 eps_decay=50000,
                 target_update=1000,
                 seed=None):

        # Set up device
        self.device = device if device else (
            'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Set up directories
        self.checkpoint_dir = checkpoint_dir
        self.models_dir = models_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

        # Training parameters
        self.env_id = env_id
        self.run_name = run_name
        self.n_agents = n_agents
        self.use_custom_rewards = use_custom_rewards
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.memory_size = memory_size

        # Set up seed for reproducibility
        self.seed = seed if seed is not None else random.randint(0, 10000)
        print(f"Using seed: {self.seed}")

        # Determine number of actions
        temp_env = self._create_env()
        self.n_actions = temp_env.action_space.n
        self.n_frame = 4  # Number of frames stacked
        temp_env.close()
        print(f"Action space size: {self.n_actions}")
        print(f"Available actions: {COMPLEX_MOVEMENT}")

        # Initialize agents
        self.agents = []

        # Global metrics for tracking progress
        self.generation = 0
        self.all_scores = []
        self.best_scores = []
        self.best_distances = []
        self.best_score_ever = 0
        self.best_distance_ever = 0

        # Shared replay memory
        self.shared_memory = deque(maxlen=self.memory_size)

        # Initialize the agents
        self._initialize_agents()

    def _create_env(self):
        """Create a Mario environment with our preferred settings"""
        env = gym_super_mario_bros.make(self.env_id)

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
        env = wrap_mario(env, use_custom_rewards=self.use_custom_rewards)

        return env

    def _initialize_agents(self):
        """Initialize the population of agents with different random seeds"""
        print(f"Initializing {self.n_agents} agents...")

        for i in range(self.n_agents):
            # Create a unique seed for each agent for diversity
            agent_seed = self.seed + i

            # Create a network for this agent
            q_network = DuelingDQN(
                self.n_frame, self.n_actions, self.device).to(self.device)

            # Initialize weights with small random variations
            q_network.init_weights()

            # Create a target network
            target_network = DuelingDQN(
                self.n_frame, self.n_actions, self.device).to(self.device)
            target_network.load_state_dict(q_network.state_dict())
            target_network.eval()

            # Create optimizer
            optimizer = optim.Adam(q_network.parameters(), lr=0.00025)

            # Store the agent
            agent = {
                'id': i,
                'seed': agent_seed,
                'q_network': q_network,
                'target_network': target_network,
                'optimizer': optimizer,
                'steps_done': 0,
                'scores': [],
                'distances': [],
                'fitness': 0  # Will be used to rank agents
            }

            self.agents.append(agent)

        print(f"All {self.n_agents} agents initialized")

    def select_action(self, state, agent, training=True):
        """Select an action using epsilon-greedy policy"""
        if training:
            agent['steps_done'] += 1

        # Calculate epsilon threshold
        if training:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                np.exp(-1. * agent['steps_done'] / self.eps_decay)

            # Use random sampling for exploration
            if random.random() < eps_threshold:
                return random.randrange(self.n_actions)

        # Use network for exploitation (or evaluation)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            return agent['q_network'](state_tensor).argmax(1).item()

    def optimize_model(self, agent):
        """Perform one step of optimization for an agent"""
        if len(self.shared_memory) < self.batch_size:
            return

        try:
            # Sample batch from memory
            transitions = random.sample(self.shared_memory, self.batch_size)

            # Process batch
            batch = list(zip(*transitions))

            # Convert to tensors
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

            # Standard training path
            current_q_values = agent['q_network'](states).gather(1, actions)

            with torch.no_grad():
                next_q_values = agent['target_network'](
                    next_states).max(1, keepdim=True)[0]

            target_q_values = rewards + (self.gamma * next_q_values * (~dones))
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

            # Optimize
            agent['optimizer'].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent['q_network'].parameters(), max_norm=10.0)
            agent['optimizer'].step()

            # Update target network if needed
            if agent['steps_done'] % self.target_update == 0:
                agent['target_network'].load_state_dict(
                    agent['q_network'].state_dict())

        except Exception as e:
            print(f"Error during optimization for agent {agent['id']}: {e}")

    def _evaluate_agent(self, agent, render=False):
        """Evaluate a single agent on a full episode"""
        env = self._create_env()

        eval_seed = self.seed + self.generation * 100 + agent['id']

        env.seed(eval_seed)
        random.seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)

        state = env.reset()
        state = arrange(state)

        # Track the most important metrics independently from the environment
        done = False
        total_reward = 0
        max_distance = 0
        final_distance = 0
        backward_steps = 0
        prev_x_pos = 0
        steps_without_progress = 0
        steps = 0

        # Run episode
        try:
            while not done:
                if render:
                    env.render()

                # Select and perform action
                action = self.select_action(state, agent, training=True)
                next_state, reward, done, info = env.step(action)
                next_state = arrange(next_state)

                # Get current position
                current_x_pos = info.get('x_pos', 0)

                # Track max distance
                if current_x_pos > max_distance:
                    max_distance = current_x_pos

                # Check for backward movement
                if current_x_pos < prev_x_pos:
                    backward_steps += 1
                else:
                    backward_steps = 0

                # Terminate if too many backward steps
                if backward_steps >= 15:  # Max allowed backward steps
                    print(
                        f"Terminating episode: Going backward for {backward_steps} steps")
                    done = True

                # Update final position for tracking
                final_distance = current_x_pos
                prev_x_pos = current_x_pos

                # Save transition to shared memory (for learning)
                self.shared_memory.append(
                    (state, action, next_state, reward, done))

                # Update state
                state = next_state

                # Optimize the model occasionally
                if steps % 4 == 0:
                    self.optimize_model(agent)

                total_reward += reward
                steps += 1

            # Calculate fitness independently from the reward
            # This creates a consistent fitness metric across generations
            fitness = self._calculate_fitness(
                max_distance, final_distance, total_reward, steps)

            # Update agent metrics
            agent['scores'].append(total_reward)
            agent['distances'].append(max_distance)
            agent['fitness'] = fitness

            print(f"Agent {agent['id']} completed evaluation: "
                  f"Score = {total_reward:.2f}, "
                  f"Max Distance = {max_distance}, "
                  f"Final Distance = {final_distance}, "
                  f"Fitness = {fitness:.2f}")

            return {
                'agent_id': agent['id'],
                'score': total_reward,
                'max_distance': max_distance,
                'final_distance': final_distance,
                'steps': steps,
                'fitness': fitness
            }

        except Exception as e:
            print(f"Error during evaluation for agent {agent['id']}: {e}")
            # Return minimal results on error
            return {
                'agent_id': agent['id'],
                'score': 0,
                'max_distance': 0,
                'final_distance': 0,
                'steps': 0,
                'fitness': 0
            }
        finally:
            env.close()

    def _calculate_fitness(self, max_distance, final_distance, score, steps):
        """
        Calculate fitness for Mario environment with proper handling of death scenarios.
        Milestones aligned with customRewardWrapper achievements.
        """
        # In Mario, max_distance is the primary goal
        distance_component = max_distance * 1  # Heavily weight maximum progress

        # In Mario, if final_distance is much lower than max_distance,
        # it's likely the agent died and respawned near position 40
        # Check if agent successfully remained at its maximum position
        completion_component = 0
        if final_distance > max_distance * 0.9:
            completion_component = max_distance * 3.0  # Increase from 2.0

        # Score component (includes coin collection, enemy defeats, etc.)
        score_component = score * 0.3

        # Speed component - faster completion is better
        speed_component = 0
        if steps > 0:
            # Reward for efficiency - how quickly max distance was reached
            speed_ratio = max_distance / steps
            speed_component = speed_ratio * 250

        # Milestone bonuses aligned with customRewardWrapper's achievements
        milestone_bonus = 0
        distance_milestones = {
            500: 500,    # Initial progress
            725: 1000,   # Good progress
            900: 1500,   # Excellent progress
            1000: 2500,  # Great progress
            1500: 3500,  # Outstanding progress
            1800: 4500,  # Exceptional progress
            2000: 5000,  # Amazing progress
            2200: 5500,  # Superior progress
            2800: 7500,  # Elite progress
            3000: 10000,  # Master progress
            3100: 15000,  # Legendary progress
        }

        for threshold, bonus in distance_milestones.items():
            if max_distance >= threshold:
                milestone_bonus += bonus

        # Combine all components
        fitness = (
            distance_component +
            completion_component +
            score_component +
            speed_component +
            milestone_bonus
        )

        return fitness

    def selection_and_reproduction(self, results):
        min_archive_threshold = 595
        # Keep track of best agent from each generation in a persistent archive
        if not hasattr(self, 'best_archive'):
            self.best_archive = []

        # Find current best agent
        current_best = max(self.agents, key=lambda a: a['fitness'])

        # Add to archive if it's good enough (maybe with some minimum fitness threshold)
        if current_best['fitness'] > min_archive_threshold:
            self.best_archive.append(self._clone_agent(current_best))

        # Create new generation - some percentage from each archived best
        new_agents = []

        # First add exact copies of archive agents (elitism)
        # Keep most recent 5
        for i, archived in enumerate(self.best_archive[-5:]):
            if len(new_agents) < self.n_agents:
                new_agents.append(self._clone_agent(
                    archived, new_id=len(new_agents)))

        # Fill remaining slots with mutations of archive agents
        while len(new_agents) < self.n_agents:
            # Select random archive agent with preference for better ones
            parent = random.choices(
                self.best_archive,
                # Weight toward newer
                weights=[i+1 for i in range(len(self.best_archive))],
                k=1
            )[0]
            child = self._mutate_agent(parent, len(new_agents))
            new_agents.append(child)

        self.agents = new_agents
        self.generation += 1

    def _clone_agent(self, agent, new_id=None):
        """Create an exact clone of an agent"""
        # Create new networks (true deep copies)
        q_network = DuelingDQN(self.n_frame, self.n_actions,
                               self.device).to(self.device)
        q_network.load_state_dict(agent['q_network'].state_dict())

        target_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)
        target_network.load_state_dict(agent['target_network'].state_dict())

        # Create optimizer
        optimizer = optim.Adam(q_network.parameters(), lr=0.00025)

        # Assign ID
        if new_id is None:
            new_id = agent['id']

        # Create a true deep copy of the agent
        cloned_agent = {
            'id': new_id,
            'seed': agent['seed'],
            'q_network': q_network,
            'target_network': target_network,
            'optimizer': optimizer,
            'steps_done': agent['steps_done'],
            'scores': agent['scores'].copy() if agent['scores'] else [],
            'distances': agent['distances'].copy() if agent['distances'] else [],
            'fitness': agent['fitness']
        }

        return cloned_agent

    def _mutate_agent(self, parent, new_id):
        """Create a mutated copy of a parent agent"""
        # Create new networks
        q_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)
        target_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)

        # Copy parent's weights
        q_network.load_state_dict(parent['q_network'].state_dict())
        target_network.load_state_dict(parent['target_network'].state_dict())

        # Apply random mutations
        mutation_strength = 0.05  # Controls how much we mutate

        with torch.no_grad():
            for param in q_network.parameters():
                # Add Gaussian noise to weights
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)

        # Sync target network with mutated Q network
        target_network.load_state_dict(q_network.state_dict())

        # Create optimizer for the new network
        optimizer = optim.Adam(q_network.parameters(), lr=0.00025)

        # Create new agent
        mutated_agent = {
            'id': new_id,
            'seed': self.seed + self.generation * 100 + new_id + 20,
            'q_network': q_network,
            'target_network': target_network,
            'optimizer': optimizer,
            'steps_done': 0,  # Reset steps
            'scores': [],  # Reset metrics
            'distances': [],
            'fitness': 0
        }

        return mutated_agent

    def _crossover_agents(self, parent1, parent2, new_id):
        """Create a child agent by crossing over two parent agents"""
        # Create new networks
        q_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)
        target_network = DuelingDQN(
            self.n_frame, self.n_actions, self.device).to(self.device)

        # Perform crossover of weights
        parent1_dict = parent1['q_network'].state_dict()
        parent2_dict = parent2['q_network'].state_dict()

        # For each layer, randomly choose weights from either parent
        with torch.no_grad():
            for name, param in q_network.named_parameters():
                # Randomly choose which parent to inherit from for this layer
                if random.random() < 0.5:
                    param.copy_(parent1_dict[name])
                else:
                    param.copy_(parent2_dict[name])

                # Add a small amount of noise for exploration
                noise = torch.randn_like(param) * 0.05
                param.add_(noise)

        # Sync target network
        target_network.load_state_dict(q_network.state_dict())

        # Create optimizer
        optimizer = optim.Adam(q_network.parameters(), lr=0.00025)

        # Create new agent
        child_agent = {
            'id': new_id,
            'seed': self.seed + self.generation * 100 + new_id + 60,
            'q_network': q_network,
            'target_network': target_network,
            'optimizer': optimizer,
            'steps_done': 0,
            'scores': [],
            'distances': [],
            'fitness': 0
        }

        return child_agent

    def save_best_model(self):
        """Save the best model from the current generation - but only if it's better than previous best"""
        if self.generation == 0:
            print("Skipping save for generation 0 to prevent unreliable initial results")
            return

        # Find the best agent
        best_agent = max(self.agents, key=lambda a: a['fitness'])

        # First check if this is indeed better than our previous best
        best_path = os.path.join(self.models_dir, f"{self.run_name}_best.pth")

        if os.path.exists(best_path):
            try:
                previous_best = torch.load(best_path, map_location=self.device)
                previous_fitness = previous_best.get('fitness', 0)

                if best_agent['fitness'] <= previous_fitness:
                    print(
                        f"Current best (fitness: {best_agent['fitness']:.2f}) not better than previous best (fitness: {previous_fitness:.2f})")
                    return
            except Exception as e:
                print(f"Error comparing with previous best, will save: {e}")

        # Still save the generation-specific model (if we're at a save interval)
        if self.generation % 10 == 0:
            model_path = os.path.join(
                self.models_dir, f"{self.run_name}_gen_{self.generation}_best.pth")

            try:
                torch.save({
                    'generation': self.generation,
                    'agent_id': best_agent['id'],
                    'q_network_state_dict': best_agent['q_network'].state_dict(),
                    'fitness': best_agent['fitness'],
                    'timestamp': datetime.now().isoformat()
                }, model_path)
                print(f"Generation best model saved to {model_path}")
            except Exception as e:
                print(f"Error saving generation best model: {e}")

        # Always save as overall best if it's better than previous
        try:
            torch.save({
                'generation': self.generation,
                'agent_id': best_agent['id'],
                'q_network_state_dict': best_agent['q_network'].state_dict(),
                'fitness': best_agent['fitness'],
                'timestamp': datetime.now().isoformat()
            }, best_path)
            print(f"New overall best model saved to {best_path}")
        except Exception as e:
            print(f"Error saving overall best model: {e}")

    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint to resume training"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load global metrics
            self.generation = checkpoint.get('generation', 0)
            self.best_scores = checkpoint.get('best_scores', [])
            self.best_distances = checkpoint.get('best_distances', [])
            self.best_score_ever = checkpoint.get('best_score_ever', 0)
            self.best_distance_ever = checkpoint.get('best_distance_ever', 0)

            # Load agents
            agent_states = checkpoint.get('agents', [])

            # Check if number of agents matches
            if len(agent_states) != self.n_agents:
                print(
                    f"Warning: Checkpoint has {len(agent_states)} agents, but we need {self.n_agents}")
                print("Will load as many as possible and create new ones if needed")

            # Load agents up to the available number
            self.agents = []
            for i, agent_state in enumerate(agent_states[:self.n_agents]):
                # Create network
                q_network = DuelingDQN(
                    self.n_frame, self.n_actions, self.device).to(self.device)
                q_network.load_state_dict(agent_state['q_network_state_dict'])

                target_network = DuelingDQN(
                    self.n_frame, self.n_actions, self.device).to(self.device)
                target_network.load_state_dict(
                    agent_state['target_network_state_dict'])

                # Create optimizer
                optimizer = optim.Adam(q_network.parameters(), lr=0.00025)
                if 'optimizer_state_dict' in agent_state:
                    optimizer.load_state_dict(
                        agent_state['optimizer_state_dict'])

                # Recreate agent
                agent = {
                    'id': i,
                    'seed': agent_state.get('seed', self.seed + i),
                    'q_network': q_network,
                    'target_network': target_network,
                    'optimizer': optimizer,
                    'steps_done': agent_state.get('steps_done', 0),
                    'scores': agent_state.get('scores', []),
                    'distances': agent_state.get('distances', []),
                    'fitness': agent_state.get('fitness', 0)
                }

                self.agents.append(agent)

            # Create any additional agents needed
            while len(self.agents) < self.n_agents:
                i = len(self.agents)

                # Create a new random agent
                q_network = DuelingDQN(
                    self.n_frame, self.n_actions, self.device).to(self.device)
                q_network.init_weights()

                target_network = DuelingDQN(
                    self.n_frame, self.n_actions, self.device).to(self.device)
                target_network.load_state_dict(q_network.state_dict())

                optimizer = optim.Adam(q_network.parameters(), lr=0.00025)

                agent = {
                    'id': i,
                    'seed': self.seed + 10000 + i,
                    'q_network': q_network,
                    'target_network': target_network,
                    'optimizer': optimizer,
                    'steps_done': 0,
                    'scores': [],
                    'distances': [],
                    'fitness': 0
                }

                self.agents.append(agent)

            print(f"Checkpoint loaded from {checkpoint_path}")
            print(f"Resuming from generation {self.generation}")

            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def save_checkpoint(self):
        """Save a checkpoint of the entire population and training state"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{self.run_name}_gen_{self.generation}.pth")

        # Prepare agent states
        agent_states = []
        for agent in self.agents:
            agent_state = {
                'id': agent['id'],
                'seed': agent['seed'],
                'q_network_state_dict': agent['q_network'].state_dict(),
                'target_network_state_dict': agent['target_network'].state_dict(),
                'optimizer_state_dict': agent['optimizer'].state_dict(),
                'steps_done': agent['steps_done'],
                'scores': agent['scores'],
                'distances': agent['distances'],
                'fitness': agent['fitness']
            }
            agent_states.append(agent_state)

        # Create checkpoint
        checkpoint = {
            'generation': self.generation,
            'agents': agent_states,
            'best_scores': self.best_scores,
            'best_distances': self.best_distances,
            'best_score_ever': self.best_score_ever,
            'best_distance_ever': self.best_distance_ever,
            'timestamp': datetime.now().isoformat()
        }

        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return False

    def plot_progress(self, save_path=None):
        """Plot training progress across generations"""
        if not self.best_scores or not self.best_distances:
            print("No training data to plot yet")
            return

        plt.figure(figsize=(15, 10))

        # Plot scores
        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.best_scores)), self.best_scores, marker='o')
        plt.title('Best Score by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)

        # Add a horizontal line for the best score ever
        plt.axhline(y=self.best_score_ever, color='r',
                    linestyle='--', alpha=0.5)
        plt.text(0, self.best_score_ever,
                 f' Best: {self.best_score_ever:.2f}', color='r')

        # Plot distances
        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.best_distances)),
                 self.best_distances, marker='o', color='g')
        plt.title('Best Distance by Generation')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.grid(True, alpha=0.3)

        # Add a horizontal line for the best distance ever
        plt.axhline(y=self.best_distance_ever,
                    color='r', linestyle='--', alpha=0.5)
        plt.text(0, self.best_distance_ever,
                 f' Best: {self.best_distance_ever}', color='r')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Progress plot saved to {save_path}")
        else:
            plt.show()

    def train(self, n_generations=50, save_interval=10, render_best=False):
        """Train the population for multiple generations"""
        try:
            for gen in range(self.generation, self.generation + n_generations):
                start_time = time.time()

                print(f"\n=== Starting Generation {gen} ===")

                # Evaluate all agents
                results = self.evaluate_generation(render=False)

                # Save the best model for this generation (will only save if it's better than previous best)
                self.save_best_model()

                # Create the next generation
                self.selection_and_reproduction(results)

                # Save checkpoint less frequently
                if gen % save_interval == 0 or gen == self.generation + n_generations - 1:
                    self.save_checkpoint()
                    # Only plot when we save
                    self.plot_progress(
                        save_path=f"{self.run_name}_gen_{gen}_progress.png")

                # Optionally render the best agent (do this less frequently too)
                if render_best and gen % 20 == 0:
                    best_agent = max(self.agents, key=lambda a: a['fitness'])
                    print(
                        f"\nRendering best agent (ID: {best_agent['id']})...")
                    self._evaluate_agent(best_agent, render=True)

                # Print generation time
                elapsed = time.time() - start_time
                print(f"Generation {gen} completed in {elapsed:.2f} seconds")

        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint()
        except Exception as e:
            print(f"Error during training: {e}")
            self.save_checkpoint()
        finally:
            print("\nTraining complete or interrupted.")
            # Final save
            self.save_checkpoint()
            self.plot_progress(save_path=f"{self.run_name}_final_progress.png")

    def evaluate_generation(self, render=False):
        """Evaluate all agents in the current generation in parallel"""
        print(f"\n=== Evaluating Generation {self.generation} ===")

        # We'll use ThreadPoolExecutor for parallel evaluation
        # This works well since most of the computation is on the GPU
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            # Submit all agents for evaluation
            future_to_agent = {
                executor.submit(self._evaluate_agent, agent, render): agent['id']
                for agent in self.agents
            }

            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent_id = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)

                    print(
                        f"Agent {agent_id} evaluation completed successfully")
                except Exception as e:
                    print(f"Agent {agent_id} evaluation failed: {e}")

        # Sort results by fitness (highest first)
        results.sort(key=lambda x: x['fitness'], reverse=True)

        # Update global metrics
        gen_scores = [r['score'] for r in results]
        gen_distances = [r['max_distance'] for r in results]

        best_result = results[0]
        self.best_scores.append(best_result['score'])
        self.best_distances.append(best_result['max_distance'])

        # Update all-time bests
        if best_result['score'] > self.best_score_ever:
            self.best_score_ever = best_result['score']
            print(f"New best score: {self.best_score_ever:.2f}")

        if best_result['max_distance'] > self.best_distance_ever:
            self.best_distance_ever = best_result['max_distance']
            print(f"New best distance: {self.best_distance_ever}")

        # Print generation summary
        print(f"\nGeneration {self.generation} Results:")
        print(f"Top agent: {best_result['agent_id']} - "
              f"Score: {best_result['score']:.2f}, "
              f"Max Distance: {best_result['max_distance']}")
        print(f"Average Score: {sum(gen_scores)/len(gen_scores):.2f}")
        print(f"Average Distance: {sum(gen_distances)/len(gen_distances):.2f}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Train Mario AI with Parallel Evolution")
    parser.add_argument("--env-id", type=str,
                        default="SuperMarioBros-v2", help="Environment ID")
    parser.add_argument("--run-name", type=str,
                        default="mario_parallel", help="Name of this run")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--models-dir", type=str,
                        default="models", help="Directory to save best models")
    parser.add_argument("--n-agents", type=int, default=16,
                        help="Number of parallel agents in the population")
    parser.add_argument("--generations", type=int, default=50,
                        help="Number of generations to train for")
    parser.add_argument("--render-best", action="store_true",
                        help="Render the best agent periodically")
    parser.add_argument("--save-interval", type=int, default=10,
                        help="Save checkpoint every N generations")
    parser.add_argument("--resume", type=str,
                        help="Resume training from checkpoint")
    parser.add_argument("--no-custom-rewards", action="store_true",
                        help="Disable custom reward wrapper")
    parser.add_argument("--memory-size", type=int, default=150000,
                        help="Size of shared replay memory")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    trainer = ParallelMarioTrainer(
        env_id=args.env_id,
        run_name=args.run_name,
        checkpoint_dir=args.checkpoint_dir,
        models_dir=args.models_dir,
        n_agents=args.n_agents,
        use_custom_rewards=not args.no_custom_rewards,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        seed=args.seed
    )

    if args.resume:
        if not trainer.load_checkpoint(args.resume):
            print("Failed to load checkpoint. Starting from scratch.")

    trainer.train(
        n_generations=args.generations,
        save_interval=args.save_interval,
        render_best=args.render_best
    )

    # Final plot
    trainer.plot_progress(save_path=f"{args.run_name}_final_progress.png")


if __name__ == "__main__":
    main()
