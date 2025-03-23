# MARIOML

To train a new model use `./mario_train.ipynb`.
To run, use `./tester.py`.

## Requirements

    Module                Version 
    gym                   0.21.0  
    gym-super-mario-bros  7.3.0   
    nes-py                8.2.1   
    pyglet                1.5.21  
    stable-baselines3     1.5.0   
    torch                 1.11.1  
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

This project uses the gym-super-mario-bros environment with a custom observation method that reads directly from the game’s RAM. You can find the code in ./marioUtils.py and check out the references in ./mario_train.ipynb.

--------- Features ---------

Tile Grid:
The game tiles (blocks, items, pipes, etc.) are stored in memory as a 32×13 grid (effectively two 16×13 grids). The displayed screen is a 16×13 portion that scrolls through this grid. Each tile represents a 16×16 pixel area, and the full rendered screen is 256×240 pixels (including areas like score display that aren’t in the RAM grid).

Positions of Mario and Enemies:
Mario and enemies have their positions stored in pixels. These positions are converted to grid coordinates by dividing by 16 and rounding.

Simplified Representation:
To keep it simple, each element is given an integer value:

2: Mario
1: Non-empty tile
0: Empty tile
–1: Enemy
This works well for World 1-1, though later levels with different enemy types might cause issues.
Observation Wrapper and Frame Stacking:
The RAM reading function is wrapped in an ObservationWrapper that also stacks a sequence of recent frames (of shape (13, 16, n_stack)) to provide temporal context.

Why Use the RAM Grid Instead of Pixels?

Efficiency: Processing raw pixel images would require deep convolutional networks, which are computationally expensive—especially.
Focus on Game Mechanics: Using the RAM grid lets the agent learn the game’s rules and level layout directly, similar to how an experienced player might ignore visual details that are irrelevant for gameplay.
Reward Function: The default reward is based on how far Mario travels, adjusted by the time taken and penalizing deaths. Changing the reward function would add extra complexity.

Training & Results
Trained a PPO agent with SB3’s MlpPolicy (using two Dense(64) Neural layers) and default hyperparameters, along with a linear learning rate scheduler that decreases the rate to 0 over time. Each model was trained for 10 million steps, taking about 4.5 hours to complete.

#### INFORMATION BELOW REGARDING TRAINING, OBSERVATION, TESTING AND DEPLOYMENT

Observation Space and Environment Setup
Our implementation uses a grid-based representation of the game state rather than raw pixels, which is a smart optimization:

Grid Representation: We convert the game's RAM state into a simplified 16x13 grid where:

0 represents empty space
1 represents tiles/obstacles
-1 represents enemies
2 represents Mario

Frame Stacking: We stack 4 frames with a skip of 4, which helps the agent understand movement and dynamics. This provides temporal context so the agent can perceive velocity and direction.
Cropping: We focus only on the relevant play area [0:16, 0:13], removing unnecessary visual information.

Reward Structure
While not explicitly defined in our code (it uses the default rewards from the gym_super_mario_bros environment), the standard reward function includes:

Forward Progress: Positive rewards for moving right (making progress in the level)
Time Penalty: Small negative rewards for each timestep to encourage faster completion
Enemy Defeat: Rewards for defeating enemies
Coin Collection: Rewards for collecting coins
Level Completion: Large reward for finishing the level

The environment typically combines these into a single reward signal that the agent tries to maximize.
Training Process
We're using Proximal Policy Optimization (PPO) which is a policy gradient method that's known for stability and good performance:

Linear Learning Rate Schedule: We implemented a learning rate that decreases linearly from 3e-4 to 0 as training progresses:
pythonCopylearning_rate=linear_learning_rate_schedule(3e-4)
This helps with early exploration and later fine-tuning.
Model Checkpointing: We save the model every 100,000 steps with our custom callback:
pythonCopycallback = ModelCheckpointCallback(check_freq=1e5, starting_steps=0, save_path=MODEL)
This ensures we don't lose progress if training is interrupted.
Extensive Training: We set the training to run for 10 million timesteps, which is substantial for learning complex behaviors.
pythonCopymodel.learn(total_timesteps=10e6, callback=callback)

Monitoring: We use TensorBoard logging to track performance metrics over time.

Key Components for Effective Learning

Custom RAM Wrapper: Our MarioGrid and RamWrapper classes provide a compact state representation by directly accessing the game's RAM rather than using pixel data. This dramatically speeds up learning by:

Simplifying the state space
Explicitly representing important game elements (Mario, enemies, tiles)
Reducing input dimensionality

Vectorized Environment: We use DummyVecEnv which allows for future parallelization of environments if needed.
MLP Policy: The 'MlpPolicy' is used since we're working with a simplified grid representation rather than images (which would require a CNN).
Action Space: We're using the SIMPLE_MOVEMENT action space which gives the agent 7 possible actions instead of all possible button combinations, simplifying the learning task.
Evaluation: We periodically evaluate the model's performance using deterministic actions (no exploration).

Deployment and Testing
Our MarioUtils class provides a convenient way to deploy and visualize the trained agent:
marioUtils = load_and_play_mario(model_name='mario-1-1', episodes=1)
This handles model loading, renders the game visually, and returns performance metrics.
Importance of Each Component

State Representation: Perhaps the most critical aspect - by converting the raw game state into a meaningful grid, we drastically reduce the complexity of the learning task.
Frame Stacking: Essential for teaching the agent about motion and dynamics, allowing it to infer velocity from multiple frames.
PPO Algorithm: A robust, sample-efficient algorithm that avoids the catastrophic policy updates that can occur with other methods.
Learning Rate Schedule: Helps balance exploration and exploitation throughout training.
Extended Training: 10 million steps gives the agent sufficient experience to learn complex behaviors in this environment.

Our implementation shows a sophisticated approach to RL for Mario, focusing on efficient state representation and modern training techniques. The agent learns to navigate the level, avoid or defeat enemies, and make progress toward the goal flag.
