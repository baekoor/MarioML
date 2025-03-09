Mario DQN - Teaching an AI to Play Super Mario Bros
A Deep Q-Network (DQN) implementation that teaches an AI agent to play Super Mario Bros using PyTorch and reinforcement learning.
Features

Dueling DQN Architecture: Superior learning performance with separate value and advantage streams
GPU Optimization: CUDA acceleration with memory management for faster training
Advanced Reward Shaping: Custom incentives for game progression and obstacle clearing
Visualization: TensorBoard integration and automatic performance graphs
Progress Tracking: Periodic evaluation and checkpointing
Breakthrough Detection: Special reward boosting for difficult level sections

Requirements

Python 3.7+
PyTorch 1.13.1 with CUDA support (recommended)
NES-Py and gym-super-mario-bros
CUDA-compatible GPU

Quick Start
Installation
bashCopy# Setup virtual environment
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# Install dependencies

pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install numpy==1.24.3 matplotlib==3.7.2 psutil==5.9.5 tensorboard==2.11.2
pip install opencv-python
pip install gym==0.23.1 nes-py==8.2.1 gym-super-mario-bros==7.4.0

# Or use the setup script

python uninstall_and_setup.py
Training
bashCopy# Basic training
python mario_dqn.py --episodes 100

# Training with 8GB VRAM reservation (adjust for your GPU)

python mario_dqn.py --episodes 100 --reserve-memory 8

# Resume from checkpoint

python mario_dqn.py --checkpoint "path/to/checkpoint.pt"
Using a Trained Agent
bashCopy# Run agent from the most recent checkpoint
python run_mario_agent.py

# Run agent with specific checkpoint

python run_mario_agent.py --checkpoint "mario_checkpoints/xxx/mario_model_step_xxxx_ep_xx.pt"
Command Line Arguments
ArgumentDescriptionDefault--episodesNumber of training episodes500--no-renderDisable rendering for faster trainingRendering enabled--save-everySave interval (episodes)20--checkpointPath to load checkpoint fromNone--action-typeAction space ('simple', 'right_only', 'custom')'simple'--eval-intervalEvaluation interval (episodes)50--no-ampDisable automatic mixed precisionAMP enabled--reserve-memoryVRAM to reserve (GB)8
Monitoring Training
Training metrics are saved to mario_checkpoints/[timestamp]/:

Model checkpoints (.pt files)
TensorBoard logs
Performance graphs (rewards, distances, etc.)

bashCopy# View training metrics
tensorboard --logdir mario_checkpoints
Troubleshooting

GPU Memory Issues: Reduce --reserve-memory to match your GPU capacity (e.g., 4GB)
NES Emulator Errors: These usually don't stop training but indicate issues with the emulator
Slow Training: Use --no-render for faster training
"Stuck at 594": The agent may struggle at certain obstacles - try longer training or use the checkpoint from episode 59+

Demo Scripts
Run a demo with your trained agent:
bashCopy# Windows
run_mario.bat

# Manual run

python run_mario_agent.py --episodes 5
