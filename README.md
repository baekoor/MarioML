# MARIOML

The pre-trained models are located under `./models`. To run these models use `./ppo-play.ipynb`.
To train a new model use `./ppo-train.ipynb`.

## Requirements (tested)

    | Module               | Version |
    | -------------------- | ------- |
    | gym                  | 0.21.0  |
    | gym-super-mario-bros | 7.3.0   |
    | nes-py               | 8.2.1   |
    | pyglet               | 1.5.21  |
    | stable-baselines3    | 1.5.0   |
    | torch                | 1.11.1  |

    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

This project uses the gym-super-mario-bros environment with a custom observation method that reads directly from the game’s RAM. You can find the code in ./smb_utils.py and check out the references in ./ppo-train.ipynb.

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
Trained a PPO agent with SB3’s MlpPolicy (using two Dense(64) layers) and default hyperparameters, along with a linear learning rate scheduler that decreases the rate to 0 over time. Each model was trained for 10 million steps, taking about 4.5 hours to complete.
