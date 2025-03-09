
import torch
import gym_super_mario_bros
import sys

print(f"Python: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available - check your PyTorch installation")

try:
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v2')
    print("Mario environment created successfully")
    env.close()
except Exception as e:
    print(f"Error creating Mario environment: {e}")
