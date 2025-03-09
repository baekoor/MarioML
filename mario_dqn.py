import gc
from contextlib import nullcontext
import os
import time
import datetime
import random
import pickle
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.jit as jit

# Import for Super Mario Bros environment
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY

# Use regular gym for compatibility
import gym
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack

# Set up CUDA for GPU acceleration with optimizations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

state_tensor = torch.zeros((4, 84, 84), device=device, dtype=torch.uint8)
next_state_tensor = torch.zeros((4, 84, 84), device=device, dtype=torch.uint8)


def safe_step(env, action, max_retry=3):
    """Safely perform a step in the environment with error handling."""
    for attempt in range(max_retry):
        try:
            return env.step(action)
        except (OSError, RuntimeError) as e:
            if attempt < max_retry - 1:
                print(
                    f"Environment step error: {e}, retrying ({attempt+1}/{max_retry})...")
                time.sleep(0.1)  # Small delay before retry
                continue
            else:
                # If we've exhausted retries, reset the environment
                print(
                    f"Environment step failed after {max_retry} attempts. Resetting...")
                try:
                    state = env.reset()
                    # Return a safe default
                    return state, 0.0, True, {"x_pos": 0, "flag_get": False}
                except Exception as reset_error:
                    print(f"Environment reset also failed: {reset_error}")
                    # Create a minimal valid state if everything fails
                    dummy_state = np.zeros((4, 84, 84, 1), dtype=np.uint8)
                    dummy_info = {"x_pos": 0, "flag_get": False}
                    return dummy_state, 0.0, True, dummy_info


def cuda_error_handling(func):
    """Decorator to handle CUDA errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "CUDA" in str(e) or "cuda" in str(e) or "failed to execute" in str(e):
                print(f"CUDA ERROR in {func.__name__}: {e}")
                print("Attempting to recover...")
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Try one more time
                try:
                    return func(*args, **kwargs)
                except Exception as e2:
                    print(f"Recovery failed: {e2}")
                    # Return a safe default value based on the function
                    if func.__name__ == 'act':
                        # Return random action
                        return random.randrange(args[0].action_count)
                    return None
            else:
                raise  # Re-raise if not a CUDA error
    return wrapper


if torch.cuda.is_available():
    try:
        # Limit memory growth to avoid spikes
        # If using TensorFlow in background
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['CUDA_CACHE_DISABLE'] = '1'  # Disable the CUDA cache
        # Synchronous kernel launches for debugging
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # More explicit error reporting
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Explicitly select first GPU

    except Exception as e:
        print(f"Error setting up CUDA environment: {e}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # More aggressive CUDA optimizations
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TensorFloat-32 precision
    torch.backends.cudnn.allow_tf32 = True        # Allow TensorFloat-32 precision

    # Empty cache first
    torch.cuda.empty_cache()

    # Reserve memory for training
    reserved_gb = 8
    print(f"Reserving {reserved_gb}GB VRAM for training")

    # More efficient memory management approach using PyTorch's memory allocator
    # Instead of creating a huge tensor, we'll pre-allocate a pool for caching
    # This allows PyTorch to manage memory more efficiently

    # Create a series of smaller tensors that get cached in PyTorch's allocator
    # This primes the memory pool without using a single huge allocation
    cached_tensors = []
    allocation_size = 128 * 1024 * 1024  # 128MB per chunk
    num_chunks = (reserved_gb * 1024) // 128

    for i in range(num_chunks):
        tensor = torch.zeros(allocation_size // 4,
                             dtype=torch.float32, device=device)
        # Just access the tensor to force allocation
        tensor[0] = 1.0
        cached_tensors.append(tensor)
        if i % 10 == 0:  # Print status every 10 chunks
            current_gb = torch.cuda.memory_allocated(0) / (1024**3)
            print(
                f"Allocated {current_gb:.2f}GB VRAM ({i}/{num_chunks} chunks)")

    # Release references but leave memory in PyTorch's cache
    del cached_tensors

    # Set PyTorch memory management parameters for better performance
    # Empty cache but keep the memory pool reserved
    torch.cuda.empty_cache()

    # Enable tensor cores if available (for RTX cards)
    # This significantly speeds up certain operations
    # Volta or newer architecture
    if torch.cuda.get_device_capability(0)[0] >= 7:
        print("Tensor cores are available - enabling for faster performance")

    # Verify memory allocation
    allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
    reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"Memory allocated: {allocated_gb:.2f} GB")
    print(f"Memory reserved: {reserved_gb:.2f} GB")

    # Set additional environment variables for optimized CUDA performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches

    # Initialize other performance optimizations
    # Increase batch size if sufficient memory is available
    INCREASED_BATCH_SIZE = 64
    print(f"Using increased batch size: {INCREASED_BATCH_SIZE}")


# ==============================
# 1. Environment Setup
# ==============================
def make_env(action_type="simple", stage="1-1"):
    """Create the Mario environment with specified action set and stage"""
    env = gym_super_mario_bros.make(f"SuperMarioBros-{stage}-v3")

    # Choose action set based on parameter
    if action_type == "simple":
        actions = SIMPLE_MOVEMENT
    elif action_type == "right_only":
        actions = RIGHT_ONLY
    else:
        # Create a custom action set
        actions = [
            ['NOOP'],
            ['right'],
            ['right', 'A'],
            ['right', 'B'],
            ['right', 'A', 'B'],
            ['A'],
            ['left'],
        ]

    env = JoypadSpace(env, actions)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)

    # Debug info
    print(f"Environment observation space: {env.observation_space}")
    state = env.reset()
    state_array = np.array(state)
    print(f"Example state shape: {state_array.shape}, type: {type(state)}")

    return env, len(actions)


# =====================================
# Dueling DQN with Optimizations
# =====================================
class DuelingDQN(nn.Module):
    def __init__(self, input_channels, action_count):
        super(DuelingDQN, self).__init__()

        # Use more efficient layer configurations for GPU
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),  # Add BatchNorm for faster convergence
            nn.ReLU(inplace=True),  # inplace=True saves memory
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # Calculate feature size based on input dimensions and architecture
        # For 84x84 input with the above architecture
        feature_size = self._calculate_conv_output_size()

        # State value stream with improved layer sizes
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )

        # Action advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, action_count)
        )

        # Improved weight initialization for better convergence
        self._initialize_weights()

        # Move model to GPU and convert to optimized format
        if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
            self.half()

    def _calculate_conv_output_size(self):
        # Create a dummy input to calculate the feature map size
        x = torch.zeros(1, 4, 84, 84)

        # Forward pass through convolutional layers only
        for layer in self.features:
            if isinstance(layer, nn.Flatten):
                break
            x = layer(x)

        # Get the flattened size
        shape_values = torch.tensor(list(x.shape[1:]))
        product = 1
        for dim in shape_values:
            product *= dim
        return int(product)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming initialization for convolutional layers
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize batch norm layers
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                # Initialize linear layers
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                module.bias.data.zero_()

    def forward(self, x):
        """Forward pass through network with optimizations"""
        # Convert input to half precision for faster computation
        # We'll keep input normalization in float32 for stability then convert to float16
        x = x.float() / 255.0
        x = x.half()  # Convert to float16 after normalization

        # Apply feature extractor
        features = self.features(x)

        # Calculate state value and action advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages (Q = V + A - mean(A))
        # Using a more numerically stable implementation
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


# ============================
# Enhanced Mario Agent with GPU Optimizations
# ============================
class MarioAgent:
    def __init__(self, input_channels, action_count, save_dir,
                 lr=1e-4, memory_size=100000, batch_size=64,  # Increased batch size
                 gamma=0.99, exploration_max=1.0, exploration_min=0.05,
                 exploration_decay=0.9999):
        self.breakthrough_memory = []  # Store special experiences
        self.device = device
        self.action_count = action_count
        self.save_dir = save_dir
        self.input_channels = input_channels
        self.batch_size = batch_size
        self.gamma = gamma

        # Store exploration parameters as instance attributes
        self.exploration_max = exploration_max  # This was missing
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_max  # Initialize with max value

        # Create models with optimized architecture
        self.policy_net = DuelingDQN(
            input_channels, action_count).to(self.device)
        self.target_net = DuelingDQN(
            input_channels, action_count).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Disable gradient calculation for target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer with improved hyperparameters
        self.optimizer = torch.optim.AdamW(  # AdamW instead of Adam
            self.policy_net.parameters(),
            lr=lr,
            weight_decay=1e-5,  # Small weight decay for regularization
            eps=1e-5  # Smaller epsilon for better numerical stability
        )

        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=100,
            threshold=0.01, min_lr=1e-6, verbose=True
        )

        # Exploration parameters with better decay strategy
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay

        # Use a better exploration strategy - linear annealing
        self.exploration_steps = 100000  # Steps to reach minimum exploration
        self.exploration_step = (
            exploration_max - exploration_min) / self.exploration_steps

        # Metrics and counters
        self.curr_step = 0
        self.learn_step_counter = 0
        self.target_update_freq = 1000  # Update target network every N steps

        # More efficient memory implementation using torch tensors
        self.memory_size = memory_size
        self.memory_counter = 0

        # Pre-allocate memory on GPU for efficient access
        # This avoids CPU-GPU transfers during training
        self.state_memory = torch.zeros(
            (memory_size, input_channels, 84, 84),
            dtype=torch.uint8, device='cpu'
        )
        self.next_state_memory = torch.zeros(
            (memory_size, input_channels, 84, 84),
            dtype=torch.uint8, device='cpu'
        )
        self.action_memory = torch.zeros(
            (memory_size, 1),
            dtype=torch.long, device='cpu'
        )
        self.reward_memory = torch.zeros(
            (memory_size, 1),
            dtype=torch.float32, device='cpu'
        )
        self.done_memory = torch.zeros(
            (memory_size, 1),
            dtype=torch.bool, device='cpu'
        )

        # Prioritized experience replay components
        self.priorities = torch.zeros(
            (memory_size, 1),
            dtype=torch.float32, device='cpu'
        )
        self.priority_scale = 0.8  # Increased from 0.6 for more aggressive prioritization
        self.priority_epsilon = 0.01  # Small value to avoid zero priorities

        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

        # Print model info
        self._debug_model_info()

        # Prefetch some operations to optimize CUDA graph
        self._warmup_gpu()

    def _warmup_gpu(self):
        """Warmup GPU operations to optimize CUDA graph compilation"""
        print("Warming up GPU operations...")
        # Create sample inputs
        dummy_state = torch.zeros(
            (self.batch_size, 4, 84, 84), device=self.device)

        # Run a few forward and backward passes to compile CUDA graphs
        for _ in range(10):
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                output = self.policy_net(dummy_state)
                loss = output.sum()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.cuda.synchronize()
        print("GPU warmup complete")

    def _debug_model_info(self):
        """Print model information for debugging"""
        # Count parameters
        param_count = sum(p.numel() for p in self.policy_net.parameters())
        print(f"Model has {param_count:,} parameters")

        # Get model size in MB
        model_size_mb = sum(p.element_size() * p.nelement()
                            for p in self.policy_net.parameters()) / (1024 * 1024)
        print(f"Model size: {model_size_mb:.2f} MB")

        # Create profiler to measure performance
        print("Running model with profiler to measure performance...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_stack=True
        ) as prof:
            sample_input = torch.zeros((1, 4, 84, 84), device=self.device)
            _ = self.policy_net(sample_input)

        # Print top 5 most time-consuming operations
        print("Top 5 most time-consuming operations:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))

    def _lazy_frames_to_tensor(self, lazy_frames, reuse_tensor=None):
        """Convert LazyFrames to tensor with shape and type optimizations"""
        # Convert LazyFrames to numpy array
        array = np.array(lazy_frames)

        # Fix shape - from (4, 84, 84, 1) to (4, 84, 84)
        if array.shape == (4, 84, 84, 1):
            array = array.squeeze(-1)

        # Use pre-allocated tensor if provided
        if reuse_tensor is not None:
            reuse_tensor.copy_(torch.ByteTensor(array))
            return reuse_tensor

        # Otherwise create new tensor
        tensor = torch.ByteTensor(array).to(self.device)
        return tensor

    @cuda_error_handling
    def act(self, state, reuse_tensor=None):
        """Select action using epsilon-greedy policy with optimizations"""
        # Better epsilon annealing strategy
        if self.curr_step < self.exploration_steps:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_max - self.curr_step * self.exploration_step
            )
        else:
            self.exploration_rate = self.exploration_min

        # Special exploration handling for the 594 barrier
        # If we have access to position info
        # This assumes you pass x_pos to the agent somehow, possibly through the state
        # You might need to adapt this based on your exact implementation
        boost_exploration = False
        x_pos = 0

        # Try to extract position from state
        # This is a simplistic check - you might need to implement this differently
        # based on how your state representation works
        if hasattr(state, 'info') and hasattr(state.info, 'x_pos'):
            x_pos = state.info.x_pos
            if 590 <= x_pos <= 600:
                boost_exploration = True
            if 720 <= x_pos <= 730:
                boost_exploration = True
            if 890 <= x_pos <= 900:
                boost_exploration = True

        # If we are at the barrier, boost exploration temporarily
        effective_exploration_rate = self.exploration_rate
        if boost_exploration:
            effective_exploration_rate = min(
                0.5, self.exploration_rate * 2)  # Double exploration rate up to 50%
        else:
            effective_exploration_rate = self.exploration_rate

        # Epsilon-greedy action selection
        if random.random() < effective_exploration_rate:
            # Random action
            action = random.randrange(self.action_count)
        else:
            # Greedy action with optimized tensor handling
            state_tensor = self._lazy_frames_to_tensor(state, reuse_tensor)

            # Use inference mode and mixed precision for faster forward pass
            with torch.inference_mode(), torch.cuda.amp.autocast():
                # Don't call unsqueeze if using reuse_tensor since it already has batch dimension
                if reuse_tensor is not None:
                    q_values = self.policy_net(state_tensor)
                else:
                    # Otherwise add batch dimension for newly created tensor
                    q_values = self.policy_net(state_tensor.unsqueeze(0))

                action = torch.argmax(q_values, dim=1).item()

        # Update step counter
        self.curr_step += 1

        return action

    def cache(self, state, next_state, action, reward, done, info=None):
        """Add experience to memory with enhanced prioritization for breakthroughs"""
        # Get the index where we'll store the experience
        index = self.memory_counter % self.memory_size

        # Convert states to tensors and store directly
        state_tensor = self._lazy_frames_to_tensor(state)
        next_state_tensor = self._lazy_frames_to_tensor(next_state)

        # Store in memory tensors - moved to CPU
        self.state_memory[index].copy_(state_tensor.cpu())
        self.next_state_memory[index].copy_(next_state_tensor.cpu())
        self.action_memory[index, 0] = action
        self.reward_memory[index, 0] = reward
        self.done_memory[index, 0] = bool(done)

        # Set initial priority (max priority for new experiences)
        max_priority = min(self.priorities.max().item(
        ) if self.memory_counter > 0 else 1.0, 100.0)  # Cap max priority

        # Enhanced prioritization for breakthrough experiences
        priority_boost = 1.0

        # If info is provided (should be passed from step() in the training loop)
        if info is not None:
            x_pos = info.get('x_pos', 0)

            # Significant boosting for distance milestones
            if x_pos > 700:  # Far progression
                priority_boost = 10.0
                print(f"SUPER PRIORITY: Distance {x_pos} > 700")
            elif x_pos > 600:  # Beyond the 594 barrier
                priority_boost = 5.0
                print(f"HIGH PRIORITY: Distance {x_pos} > 600")
            elif x_pos > 500:
                priority_boost = 3.0
            elif x_pos > 400:
                priority_boost = 2.0

            # Additional boost for flag capture
            if info.get('flag_get', False):
                priority_boost = 20.0  # Use direct assignment instead of multiplication
                print("MAXIMUM PRIORITY: Level completed!")

        if info is not None:
            x_pos = info.get('x_pos', 0)
            # Calculate progress here
            x_progress = x_pos - getattr(self, 'last_x_pos', x_pos-1)

            # Store ONLY SUCCESSFUL breakthroughs - when there's positive progress
            if (((590 <= x_pos <= 610 and x_pos > 595 and x_progress > 0) or  # Only past 595 for the first barrier
                (720 <= x_pos <= 740 and x_pos > 722 and x_progress > 0) or
                    (895 <= x_pos <= 910 and x_pos > 898 and x_progress > 0))):
                self.breakthrough_memory.append(
                    (state, next_state, action, reward*2, done))
                print(f"Stored TRUE breakthrough at position {x_pos}!")

                # Limit size of breakthrough memory
                if len(self.breakthrough_memory) > 1000:  # Cap at 1000 experiences
                    self.breakthrough_memory.pop(0)  # Remove oldest

        # Apply the boost with a safety cap to prevent overflow
        try:
            # Cap the maximum priority
            self.priorities[index] = min(max_priority * priority_boost, 1000.0)
        except RuntimeError as e:
            print(f"Warning: Priority calculation error: {e}")
            print(
                f"max_priority = {max_priority}, priority_boost = {priority_boost}")
            # Fallback to a safe value
            self.priorities[index] = 1000.0

        # Update counter
        self.memory_counter += 1

    def recall(self):
        """Sample batch from memory with priority sampling and occasional breakthrough examples"""
        # Get number of experiences in memory
        n_experiences = min(self.memory_counter, self.memory_size)

        # Determine how many samples to take from regular memory vs breakthrough memory
        # Up to 20% from breakthroughs
        n_breakthrough = min(len(self.breakthrough_memory),
                             self.batch_size // 5)
        n_regular = self.batch_size - n_breakthrough

        # Sample from regular memory
        if n_experiences < n_regular:
            # Not enough experiences, sample with replacement
            indices = torch.randint(0, n_experiences, (n_regular,))
        else:
            # Use prioritized sampling
            priorities = self.priorities[:n_experiences]

            # Convert priorities to probabilities
            probs = priorities.pow(self.priority_scale)
            probs = probs / probs.sum()

            # Sample based on priorities
            indices = torch.multinomial(
                probs.flatten(), n_regular, replacement=True)

        # Get batch of experiences from regular memory
        state_batch = self.state_memory[indices].to(self.device).float()
        next_state_batch = self.next_state_memory[indices].to(
            self.device).float()
        action_batch = self.action_memory[indices].to(self.device)
        reward_batch = self.reward_memory[indices].to(self.device)
        done_batch = self.done_memory[indices].to(self.device).float()

        # If we have breakthrough examples, include them
        if n_breakthrough > 0:
            # Randomly sample from breakthrough memory
            breakthrough_indices = torch.randint(
                0, len(self.breakthrough_memory), (n_breakthrough,))

            # Convert breakthrough experiences to tensors
            for i, idx in enumerate(breakthrough_indices):
                # Get the experience
                bt_state, bt_next_state, bt_action, bt_reward, bt_done = self.breakthrough_memory[
                    idx]

                # Convert states to tensors
                bt_state_tensor = self._lazy_frames_to_tensor(
                    bt_state).unsqueeze(0)
                bt_next_state_tensor = self._lazy_frames_to_tensor(
                    bt_next_state).unsqueeze(0)

                # Append to the batch
                append_idx = n_regular + i
                # The key issue: make sure append_idx is in bounds
                # Check using size(0) instead of self.batch_size
                if append_idx < state_batch.size(0):
                    state_batch[append_idx] = bt_state_tensor.float()
                    next_state_batch[append_idx] = bt_next_state_tensor.float()
                    action_batch[append_idx, 0] = bt_action
                    reward_batch[append_idx, 0] = bt_reward
                    done_batch[append_idx, 0] = float(bt_done)

        # Save indices for priority updates (only for regular memory)
        self.batch_indices = indices

        return state_batch, next_state_batch, action_batch, reward_batch, done_batch

    @cuda_error_handling
    def learn(self):
        """Update model weights based on batch of experiences with GPU optimizations"""
        # Skip if not enough samples
        if self.memory_counter < self.batch_size:
            return None

        # Learn only every few steps for efficiency
        if self.curr_step % 2 != 0:
            return None

        # Sample batch from memory
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.recall()

        # Now check for barrier indices correctly - use tensor operations
        barrier_lr_adjustment = False

        # Check if any indices are in the critical barrier regions
        barrier_indices_720 = ((self.batch_indices >= 700) & (
            self.batch_indices <= 730)).any().item()
        barrier_indices_890 = ((self.batch_indices >= 890) & (
            self.batch_indices <= 910)).any().item()

        if barrier_indices_720 or barrier_indices_890:
            # Reduce learning rate temporarily when training on barrier experiences
            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = current_lr * 0.5
            barrier_lr_adjustment = True

        # Enable mixed precision for faster computation
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            amp_context = torch.cuda.amp.autocast()
        else:
            from contextlib import nullcontext
            amp_context = nullcontext()

        with amp_context:
            # Compute current Q values
            current_q_values = self.policy_net(
                state_batch).gather(1, action_batch)

            # Compute next Q values using Double DQN approach
            with torch.no_grad():
                # Get actions from policy network
                next_actions = self.policy_net(next_state_batch).max(1)[
                    1].unsqueeze(1)
                # Get Q-values from target network for those actions
                next_q_values = self.target_net(
                    next_state_batch).gather(1, next_actions)
                # Compute target Q values
                target_q_values = reward_batch + \
                    (1 - done_batch) * self.gamma * next_q_values

            # Calculate TD errors for prioritized replay
            td_errors = torch.abs(
                current_q_values - target_q_values).detach().cpu()

            # Update priorities
            for i, idx in enumerate(self.batch_indices):
                self.priorities[idx] = td_errors[i] + self.priority_epsilon

            # Calculate Huber loss (more stable than MSE)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)

        # Optimize the model with gradient scaling for mixed precision
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=10.0)

        # Step optimizer
        self.optimizer.step()

        # Restore learning rate if it was temporarily adjusted
        if barrier_lr_adjustment:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.param_groups[0]['lr'] = current_lr * 2.0

        # Update target network periodically - soft update for better stability
        self.learn_step_counter += 1
        if self.learn_step_counter % 10 == 0:  # More frequent but softer updates
            # Soft update: θ′ ← τθ + (1 − τ)θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * \
                    0.01 + target_net_state_dict[key] * 0.99

            self.target_net.load_state_dict(target_net_state_dict)

        # Full update less frequently
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Full target network update at step {self.curr_step}")

        # Log metrics
        self.writer.add_scalar('Training/Loss', loss.item(), self.curr_step)
        self.writer.add_scalar(
            'Training/Q_Value', current_q_values.mean().item(), self.curr_step)
        self.writer.add_scalar('Training/Exploration_Rate',
                               self.exploration_rate, self.curr_step)

        return loss.item(), current_q_values.mean().item()

    def adjust_learning_rate(self, avg_reward):
        """Adjust learning rate based on performance"""
        self.scheduler.step(avg_reward)
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('Training/Learning_Rate',
                               current_lr, self.curr_step)

    def save(self, episode=0):
        """Save model checkpoint with compression for smaller files"""
        save_path = self.save_dir / \
            f"mario_model_step_{self.curr_step}_ep_{episode}.pt"

        # Save in a background thread to avoid stalling training
        def save_model_thread():
            checkpoint = {
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'exploration_rate': self.exploration_rate,
                'curr_step': self.curr_step,
                'learn_step_counter': self.learn_step_counter
            }

            # Save with compression
            torch.save(checkpoint, save_path,
                       _use_new_zipfile_serialization=True)
            print(f"Model saved to {save_path}")

        import threading
        save_thread = threading.Thread(target=save_model_thread)
        save_thread.start()

    def load(self, filepath):
        """Load model checkpoint with enhancements"""
        print(f"Loading model from {filepath}...")

        # Load checkpoint using map_location to handle device changes
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load model state dicts
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler if it exists
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Load other parameters
        self.exploration_rate = checkpoint['exploration_rate']
        self.curr_step = checkpoint['curr_step']
        self.learn_step_counter = checkpoint.get('learn_step_counter', 0)

        print(f"Model loaded successfully from {filepath}")
        print(
            f"Resuming from step {self.curr_step} with exploration rate {self.exploration_rate:.4f}")

# ============================
# 4. Environment Wrapper with Advanced Reward Shaping
# ============================


class EnhancedMarioEnv:
    def __init__(self, env):
        self.env = env
        self.last_x_pos = 40
        self.max_x_pos = 40
        self.last_time = 400
        self.stuck_counter = 0
        self.max_stuck_steps = 50
        self.last_y_pos = 0
        self.jump_count = 0
        self.jump_heights = []
        self.stuck_at_594_count = 0  # Track attempts at the barrier
        self.last_actions = []

        # For tracking progress
        self.coin_count = 0
        self.score = 0
        self.last_score = 0

        # Performance metrics
        self.episode_step_count = 0
        self.total_reward = 0

        # Stage checkpoints for better rewards
        self.checkpoints = [100, 200, 300, 400, 500, 700, 1000, 1500, 2000]
        self.passed_checkpoints = set()

    def reset(self):
        """Reset environment and tracking variables"""
        state = self.env.reset()

        # Reset tracking variables
        self.last_x_pos = 40
        self.max_x_pos = 40
        self.last_time = 400
        self.stuck_counter = 0
        self.coin_count = 0
        self.score = 0
        self.last_score = 0
        self.episode_step_count = 0
        self.total_reward = 0
        self.passed_checkpoints = set()
        self.last_actions = []  # Reset action history

        # Reset jump tracking
        self.last_y_pos = 79  # Default ground level
        self.jump_count = 0
        self.jump_heights = []
        self.stuck_at_594_count = 0

        return state

    def step(self, action):
        """Take action and apply reward shaping"""
        # Increment step counter
        self.episode_step_count += 1

        # Take action in environment
        state, reward, done, info = self.env.step(action)

        # Extract info
        x_pos = info.get('x_pos', 0)
        y_pos = info.get('y_pos', 0)  # Track vertical position
        time_left = info.get('time', 0)
        score = info.get('score', 0)
        coins = info.get('coins', 0)
        status = info.get('status', 'small')  # small, tall, fireball

        # Start with environment reward (typically very small)
        shaped_reward = reward * 0.1

        # Progress reward - horizontal movement
        x_progress = x_pos - self.last_x_pos
        if x_pos > self.max_x_pos:
            # Bonus for furthest position yet
            shaped_reward += 0.5 * (x_pos - self.max_x_pos)
            self.max_x_pos = x_pos
            self.stuck_counter = 0

        self.last_actions.append(action)
        if len(self.last_actions) > 5:
            self.last_actions.pop(0)

        # SPECIFIC HANDLING FOR THE 594 BARRIER
        # If we're near the barrier, provide special incentives
        if 590 <= x_pos <= 605:
            # We're in the barrier region

            # Track if we're stuck at the barrier
            if 593 <= x_pos <= 595:
                self.stuck_at_594_count += 1
            else:
                self.stuck_at_594_count = 0

            # Jump detection and reward
            y_change = y_pos - self.last_y_pos

            # Reward for jumping/being airborne in the barrier region
            if y_pos < 79:  # If Mario is above ground level
                # The lower the y_pos, the higher Mario is (screen coordinates)
                jump_height = max(0, 79 - y_pos)

                # Exponential reward for jump height in barrier region
                jump_reward = 0.1 * (jump_height ** 1.5)
                shaped_reward += jump_reward

                # Extra reward for jumping actions specifically
                if action in [2, 3, 4, 5]:  # Jump actions in SIMPLE_MOVEMENT
                    shaped_reward += 0.2

                print(
                    f"Jump reward at position {x_pos}: +{jump_reward:.2f}, height: {jump_height}")

            # If stuck for too long at 594, add exploration boost
            if self.stuck_at_594_count > 20:
                # Encourage trying different actions
                # Reward actions different from the common ones
                shaped_reward += 0.5 * (1 - abs(action - 3) / 6)

                # Every 50 stuck steps, add a random reward to break patterns
                if self.stuck_at_594_count % 50 == 0:
                    import random
                    shaped_reward += random.uniform(0.5, 2.0)
                    print(
                        f"Adding randomized reward to break 594 barrier: +{shaped_reward:.2f}")

            if len(self.last_actions) >= 3:
                if self.last_actions[-1] in [2, 3, 4] and self.last_actions[-2] == 1 and self.last_actions[-3] == 1:
                    # Running (action 1) then jumping (action 2,3,4)
                    shaped_reward += 2.0
                    print("Sequence reward: run->jump pattern")

        # Even bigger breakthrough bonus for passing the barrier
        if x_pos > 600 and self.max_x_pos <= 600:
            shaped_reward += 25.0  # Increased from 15.0
            print(
                f"BREAKTHROUGH! Agent passed the 594 barrier! New position: {x_pos}")

        if 720 <= x_pos <= 730:
            # We're at the 722 barrier
            # Similar rewards for jumping in this region
            if y_pos < 79:  # If Mario is above ground level
                jump_height = max(0, 79 - y_pos)
                # Stronger reward curve
                jump_reward = 0.15 * (jump_height ** 1.8)
                shaped_reward += jump_reward
                print(
                    f"722 barrier jump reward: +{jump_reward:.2f}, height: {jump_height}")

        if x_progress > 0:
            # Reward for moving right
            shaped_reward += 0.2 * x_progress
            self.stuck_counter = 0
        else:
            # Small penalty for not moving forward
            shaped_reward -= 0.01
            self.stuck_counter += 1

        # Check for passing checkpoints
        for checkpoint in self.checkpoints:
            if x_pos >= checkpoint and checkpoint not in self.passed_checkpoints:
                shaped_reward += 5.0  # Substantial reward for passing a checkpoint
                self.passed_checkpoints.add(checkpoint)

        # Score/coin rewards
        score_increase = score - self.last_score
        if score_increase > 0:
            shaped_reward += 0.2 * score_increase / 100.0  # Scale down the score

        # Bonus for collecting coins
        if coins > self.coin_count:
            shaped_reward += 2.0 * (coins - self.coin_count)
            self.coin_count = coins

        # Bonus for power-ups
        if status == 'tall' and info.get('status', 'small') == 'small':
            shaped_reward += 5.0  # Significant reward for getting super mushroom
        elif status == 'fireball':
            shaped_reward += 10.0  # Major reward for getting fire flower

        # Time penalties
        time_penalty = 0.001 * (self.last_time - time_left)
        shaped_reward -= time_penalty

        # Completion bonus
        if info.get('flag_get', False):
            shaped_reward += 50.0  # Major reward for completing level
            done = True

        # Death/termination penalties
        if done and not info.get('flag_get', False):
            shaped_reward -= 5.0  # Penalty for dying

        # End episode if Mario is stuck
        if self.stuck_counter >= self.max_stuck_steps:
            shaped_reward -= 5.0  # Penalty for getting stuck
            done = True

        # Update tracking variables
        self.last_x_pos = x_pos
        self.last_y_pos = y_pos  # Track vertical position
        self.last_time = time_left
        self.last_score = score
        self.total_reward += shaped_reward

        return state, shaped_reward, done, info

    def render(self):
        """Render environment"""
        return self.env.render()

    def close(self):
        """Close environment"""
        return self.env.close()


def plot_metrics(save_dir, metrics):
    """Plot training metrics and save the figures"""
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    print("Plotting training metrics...")

    # Create figures directory
    figures_dir = save_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['rewards'], label='Episode Reward')
    if len(metrics['rewards']) > 100:
        # Add moving average if we have enough data
        rolling_mean = np.convolve(
            metrics['rewards'], np.ones(100)/100, mode='valid')
        plt.plot(range(99, 99+len(rolling_mean)), rolling_mean,
                 'r', label='100-episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(figures_dir / 'rewards.png'))
    plt.close()

    # Plot distances
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['distances'], label='Distance Reached')
    if len(metrics['distances']) > 100:
        # Add moving average if we have enough data
        rolling_mean = np.convolve(
            metrics['distances'], np.ones(100)/100, mode='valid')
        plt.plot(range(99, 99+len(rolling_mean)), rolling_mean,
                 'r', label='100-episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Distance')
    plt.title('Mario Distance per Episode')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(figures_dir / 'distances.png'))
    plt.close()

    # Plot episode lengths
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['episode_lengths'], label='Episode Length')
    if len(metrics['episode_lengths']) > 100:
        # Add moving average if we have enough data
        rolling_mean = np.convolve(
            metrics['episode_lengths'], np.ones(100)/100, mode='valid')
        plt.plot(range(99, 99+len(rolling_mean)), rolling_mean,
                 'r', label='100-episode Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(figures_dir / 'episode_lengths.png'))
    plt.close()

    # Plot FPS
    if metrics['fps']:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['fps'], label='FPS')
        plt.xlabel('Episode')
        plt.ylabel('Frames Per Second')
        plt.title('Training Performance (FPS)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'fps.png'))
        plt.close()

    # Plot losses and Q-values if available
    if metrics['losses'] and metrics['q_values']:
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['losses'], label='Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'losses.png'))
        plt.close()

        # Plot Q-values
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['q_values'], label='Average Q-Value')
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.title('Training Q-Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'q_values.png'))
        plt.close()

    # Plot evaluation scores if available
    if metrics['evaluation_scores']:
        episodes, scores = zip(*metrics['evaluation_scores'])
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, scores, 'o-', label='Evaluation Score')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Evaluation Scores')
        plt.legend()
        plt.tight_layout()
        plt.savefig(str(figures_dir / 'evaluation_scores.png'))
        plt.close()

    print(f"Metrics plots saved to {figures_dir}")

# ============================
# Training Loop with Performance Optimizations
# ============================


def train(num_episodes=1000, render=True, save_every=20, checkpoint_path=None,
          action_type="simple", eval_interval=50, use_amp=True):
    """Train the Mario agent with GPU optimizations"""
    # Create save directory
    save_dir = Path("mario_checkpoints") / \
        datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create environments with processing offloaded to CPU
    env, action_count = make_env(action_type)
    wrapped_env = EnhancedMarioEnv(env)

    # Create evaluation environment
    eval_env, _ = make_env(action_type)
    eval_wrapped_env = EnhancedMarioEnv(eval_env)

    # Create agent with optimized parameters
    agent = MarioAgent(
        input_channels=4,
        action_count=action_count,
        save_dir=save_dir,
        batch_size=INCREASED_BATCH_SIZE,  # Increased batch size
        memory_size=150000,  # Larger replay buffer
        lr=3e-4   # Slightly adjusted learning rate
    )

    # Load checkpoint if provided
    if checkpoint_path:
        agent.load(checkpoint_path)

    # Initialize metrics tracking
    metrics = {
        'rewards': [],
        'losses': [],
        'q_values': [],
        'distances': [],
        'episode_lengths': [],
        'evaluation_scores': [],
        'fps': []  # Track frames per second
    }

    # For smoother metric tracking
    recent_rewards = deque(maxlen=100)  # Increased window size
    recent_distances = deque(maxlen=100)
    recent_episode_lengths = deque(maxlen=100)
    recent_fps = deque(maxlen=20)

    # Create automatic mixed precision scaler if enabled
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Set process priority to high for better performance
    try:
        import psutil
        process = psutil.Process()
        process.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
        print("Process priority set to high")
    except:
        print("Could not set process priority")

    # Preload the first batch of frames to GPU
    preload_batch_size = 4
    preloaded_states = []

    # Create a state encoder for faster processing
    def encode_state(state):
        return agent._lazy_frames_to_tensor(state)

    # Pre-process initial states
    for _ in range(preload_batch_size):
        state = wrapped_env.reset()
        encoded_state = encode_state(state)
        preloaded_states.append((state, encoded_state))

    # Create environment worker for background frame processing
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=2)

    # Training loop
    print("Starting training with GPU optimizations...")
    try:
        for episode in range(num_episodes):
            last_checkpoint_path = None

            # Track episode start time
            start_time = time.time()

            # Get a pre-processed state or reset the environment
            if preloaded_states:
                state, encoded_state = preloaded_states.pop(0)
            else:
                state = wrapped_env.reset()
                encoded_state = encode_state(state)

            # Queue up a new environment reset in the background
            if len(preloaded_states) < preload_batch_size:
                executor.submit(lambda: preloaded_states.append(
                    (wrapped_env.reset(), encode_state(wrapped_env.reset()))))

            # Episode tracking
            total_reward = 0
            episode_loss = []
            episode_q = []
            step_times = []
            frame_count = 0

            done = False

            # Process each step in episode
            while not done:
                step_start = time.time()

                # Select action with reusable tensor
                action = agent.act(state)

                # Take action
                next_state, reward, done, info = safe_step(wrapped_env, action)

                # Important: Store this experience with info for priority boosting
                agent.cache(state, next_state, action, reward, done, info)

                # Update state and rewards
                total_reward += reward
                frame_count += 1

                # If done, break immediately without further processing
                if done:
                    break

                # Continue processing only if not done
                # Process learning only every 4 steps for efficiency
                if agent.curr_step % 2 == 0:
                    # Render if enabled - reduced frequency
                    if render and episode % 10 == 0 and agent.curr_step % 20 == 0:
                        wrapped_env.render()

                    # Learn from experience with AMP if enabled
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            learn_result = agent.learn()
                    else:
                        learn_result = agent.learn()

                    if learn_result:
                        loss, q_value = learn_result
                        episode_loss.append(loss)
                        episode_q.append(q_value)

                # Update state for next step
                state = next_state

                # Calculate step time
                step_time = time.time() - step_start
                step_times.append(step_time)

                # Calculate and log FPS periodically
                if len(step_times) >= 200:
                    avg_step_time = sum(step_times) / len(step_times)
                    fps = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    recent_fps.append(fps)
                    step_times = []

                    print(f"Episode {episode}, Step {agent.curr_step}: FPS={fps:.1f}, "
                          f"Exploration={agent.exploration_rate:.3f}")

                # Run very infrequent memory management
                if agent.curr_step % 3000 == 0:
                    torch.cuda.empty_cache()

            # After episode is done, ensure we reset the environment before the next episode
            if episode < num_episodes - 1:  # If not the last episode
                # Reset environment for next episode
                wrapped_env.reset()

            # Episode complete - gather metrics
            episode_time = time.time() - start_time
            episode_fps = frame_count / episode_time if episode_time > 0 else 0
            recent_fps.append(episode_fps)

            x_distance = info.get('x_pos', 0)
            flag_get = info.get('flag_get', False)
            episode_length = wrapped_env.episode_step_count

            # Update metrics
            metrics['rewards'].append(total_reward)
            metrics['distances'].append(x_distance)
            metrics['episode_lengths'].append(episode_length)
            metrics['fps'].append(episode_fps)

            recent_rewards.append(total_reward)
            recent_distances.append(x_distance)
            recent_episode_lengths.append(episode_length)

            # Track loss and Q-values if available
            if episode_loss:
                avg_loss = sum(episode_loss) / len(episode_loss)
                avg_q = sum(episode_q) / len(episode_q)
                metrics['losses'].append(avg_loss)
                metrics['q_values'].append(avg_q)

            # Calculate moving averages for metrics
            avg_reward_100 = sum(recent_rewards) / len(recent_rewards)
            avg_distance_100 = sum(recent_distances) / len(recent_distances)
            avg_length_100 = sum(recent_episode_lengths) / \
                len(recent_episode_lengths)
            avg_fps = sum(recent_fps) / len(recent_fps) if recent_fps else 0

            # Log to tensorboard
            agent.writer.add_scalar('Episode/reward', total_reward, episode)
            agent.writer.add_scalar('Episode/distance', x_distance, episode)
            agent.writer.add_scalar('Episode/length', episode_length, episode)
            agent.writer.add_scalar('Episode/flag_get', int(flag_get), episode)
            agent.writer.add_scalar(
                'Metrics/avg_reward_100', avg_reward_100, episode)
            agent.writer.add_scalar(
                'Metrics/avg_distance_100', avg_distance_100, episode)
            agent.writer.add_scalar(
                'Metrics/avg_length_100', avg_length_100, episode)
            agent.writer.add_scalar('Metrics/fps', avg_fps, episode)

            # Adjust learning rate based on performance
            agent.adjust_learning_rate(avg_reward_100)

            # Print episode summary with performance metrics
            print(f"Episode {episode}/{num_episodes}: "
                  f"Reward={total_reward:.2f}, "
                  f"Distance={x_distance}, "
                  f"Steps={episode_length}, "
                  f"{'WIN!' if flag_get else 'Loss'}, "
                  f"FPS={episode_fps:.1f}, "
                  f"Epsilon={agent.exploration_rate:.4f}, "
                  f"Time={episode_time:.2f}s")

            # Print memory usage every 10 episodes
            if episode % 10 == 0:
                allocated_mb = torch.cuda.memory_allocated(0)/1024**2
                reserved_mb = torch.cuda.memory_reserved(0)/1024**2
                print(f"GPU memory: {allocated_mb:.2f} MB allocated, "
                      f"{reserved_mb:.2f} MB reserved")

                # Get memory by category - helps identify memory leaks
                print(f"GPU memory: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB allocated, "
                      f"{torch.cuda.memory_reserved(0)/1024**2:.2f} MB reserved")

            # Evaluate periodically
            if episode % eval_interval == 0 or episode == num_episodes - 1 or flag_get:
                # Run evaluation with mixed precision
                with torch.cuda.amp.autocast() if use_amp else nullcontext():
                    eval_score = evaluate_agent(
                        agent, eval_wrapped_env, num_eval_episodes=3, render=(render and episode % 10 == 0))

                metrics['evaluation_scores'].append((episode, eval_score))
                agent.writer.add_scalar(
                    'Evaluation/score', eval_score, episode)

                # Save after evaluation
                agent.save(episode)

                # Plot metrics after evaluation
                plot_metrics(
                    save_dir=save_dir,
                    metrics=metrics
                )

            # Save periodically or on significant achievements
            if episode % save_every == 0 or flag_get or (episode > 0 and total_reward > max(metrics['rewards'][:-1], default=0) * 1.2):
                agent.save(episode)

            # Run garbage collection occasionally to prevent memory leaks
            if episode % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # With this more optimized approach:
            # Run minimal garbage collection only when needed (much less frequently)
            if episode % 50 == 0:  # Reduced from every 10 episodes to every 50
                # Only force a garbage collection cycle if memory usage is high
                allocated_mb = torch.cuda.memory_allocated(0)/1024**2
                if allocated_mb > 10000:  # Only collect if over 10GB used
                    print("Running targeted garbage collection...")
                    # Use less aggressive collection
                    gc.collect(0)  # Only collect youngest generation (faster)
                    if torch.cuda.is_available():
                        # Don't empty the cache completely - just free unused memory
                        torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Training interrupted by user")

    # Final save
    agent.save(num_episodes)

    # Clean up
    wrapped_env.close()
    eval_wrapped_env.close()
    agent.writer.close()
    executor.shutdown()

    print("Training complete!")

    # Save final metrics
    with open(save_dir / 'final_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    return metrics


# ============================
# Enhanced Evaluation Function
# ============================

def evaluate_agent(agent, env, num_eval_episodes=5, render=False):
    """Evaluate agent performance with GPU optimizations"""
    total_scores = []
    distances = []
    completion_rate = 0

    # Store original exploration rate
    original_exploration = agent.exploration_rate
    # Set to minimum for evaluation (mostly greedy)
    agent.exploration_rate = 0.05

    try:
        for i in range(num_eval_episodes):
            state = env.reset()
            done = False
            total_reward = 0

            # Create frame buffer to skip frames during evaluation (speeds it up)
            frame_skip = 1  # Can be increased to speed up evaluation
            frame_counter = 0

            # For measuring fps during evaluation
            start_time = time.time()
            step_count = 0

            while not done:
                # Skip frames for faster evaluation (optional)
                frame_counter += 1
                if frame_counter % frame_skip != 0 and frame_counter > 1:
                    # Take the same action without rendering
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    state = next_state
                    step_count += 1
                    continue

                # Select action (mostly greedy)
                with torch.cuda.amp.autocast():  # Use mixed precision for evaluation too
                    action = agent.act(state)

                # Take action
                next_state, reward, done, info = env.step(action)

                # Render if enabled
                if render:
                    env.render()

                # Update state and reward
                state = next_state
                total_reward += reward
                step_count += 1

            # Calculate fps
            duration = time.time() - start_time
            fps = step_count / duration if duration > 0 else 0

            # Get final distance and completion status
            distance = info.get('x_pos', 0)
            win = info.get('flag_get', False)
            completion_rate += int(win)

            # Save final score
            total_scores.append(total_reward)
            distances.append(distance)

            print(f"Evaluation episode {i+1}/{num_eval_episodes}: "
                  f"Score={total_reward:.2f}, "
                  f"Distance={distance}, "
                  f"FPS={fps:.1f}, "
                  f"{'WIN!' if win else 'Loss'}")

    finally:
        # Restore original exploration rate
        agent.exploration_rate = original_exploration

    # Calculate comprehensive metrics
    avg_score = sum(total_scores) / len(total_scores)
    avg_distance = sum(distances) / len(distances)
    completion_percentage = (completion_rate / num_eval_episodes) * 100

    print(f"Evaluation complete - "
          f"Average score: {avg_score:.2f}, "
          f"Average distance: {avg_distance:.1f}, "
          f"Completion rate: {completion_percentage:.1f}%")

    return avg_score


# Import for nullcontext when not using AMP

# ============================
# 8. Enhanced Main Function
# ============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mario agent with Dueling DQN and GPU optimizations')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of episodes to train')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering')
    parser.add_argument('--save-every', type=int, default=20,
                        help='Save model every N episodes')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint file to load')
    parser.add_argument('--action-type', type=str, default='simple',
                        choices=['simple', 'right_only', 'custom'], help='Action space to use')
    parser.add_argument('--eval-interval', type=int, default=50,
                        help='Evaluate model every N episodes')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--reserve-memory', type=int, default=8,
                        help='Amount of VRAM to reserve in GB')

    args = parser.parse_args()

    # Set VRAM reservation if specified
    if args.reserve_memory > 0 and torch.cuda.is_available():
        print(f"Reserving {args.reserve_memory}GB of VRAM")
        reserved_gb = args.reserve_memory
        bytes_per_element = 4  # float32
        elements_needed = reserved_gb * 1024 * 1024 * 1024 // bytes_per_element

        # Create square dimensions for the tensor
        import math
        dim_size = int(math.sqrt(elements_needed))

        # Create tensors to allocate memory
        print(f"Creating tensor with size {dim_size}x{dim_size}")
        x = torch.ones((dim_size, dim_size),
                       device=device, dtype=torch.float32)
        x.normal_()

        # Keep this tensor in global scope
        global reserved_memory
        reserved_memory = x

        # Check how much was actually reserved
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_gb_actual = torch.cuda.memory_reserved(0) / (1024**3)
        print(f"Memory allocated: {allocated_gb:.2f} GB")
        print(f"Memory reserved: {reserved_gb_actual:.2f} GB")

    train(
        num_episodes=args.episodes,
        render=not args.no_render,
        save_every=50,
        checkpoint_path=args.checkpoint,
        action_type=args.action_type,
        eval_interval=args.eval_interval,
        use_amp=not args.no_amp
    )
