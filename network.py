import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def arrange(s):
    """Rearrange the image state for the network input"""
    if not isinstance(s, np.ndarray):
        try:
            # Handle LazyFrames or other types by converting to numpy array
            s = np.array(s)
        except Exception as e:
            print(f"Error converting state to numpy array: {e}")
            print(f"State type: {type(s)}")
            # Emergency fallback
            return np.zeros((4, 84, 84), dtype=np.float32)

    # Ensure we have a 3D array
    if len(s.shape) != 3:
        print(f"Warning: Expected 3D state, got shape {s.shape}")
        # Return a zero tensor of expected shape
        return np.zeros((4, 84, 84), dtype=np.float32)

    # Transpose from HWC to CHW format
    # The input should be (H, W, C) where C is the stacked frames
    # We need to transpose to (C, H, W) for PyTorch
    ret = np.transpose(s, (2, 0, 1))

    # No need to add batch dimension - this will be handled by the model
    return ret


class DuelingDQN(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(DuelingDQN, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 16, 8, 4)
        self.layer2 = nn.Conv2d(16, 32, 3, 1)
        self.flatten_size = 10368  # Hardcoded value for 84x84 input
        self.fc = nn.Linear(self.flatten_size, 512)
        self.advantage = nn.Linear(512, n_action)
        self.value = nn.Linear(512, 1)
        self.device = device

    def forward(self, x):
        with torch.cuda.amp.autocast():
            if not isinstance(x, torch.Tensor):
                # Convert to tensor and ensure proper shape
                # If x has shape (batch_size, 1, channels, height, width), reshape it
                if isinstance(x, np.ndarray) and len(x.shape) == 5:
                    x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
                x = torch.FloatTensor(x).to(self.device)

            # If tensor has shape [batch, 1, channels, height, width], reshape it
            if len(x.shape) == 5:
                x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])

            # Add batch dimension if missing
            if len(x.shape) == 3:
                x = x.unsqueeze(0)

            # Apply convolutional layers
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))

            # Flatten the tensor
            x = x.view(x.size(0), -1)

            # If flatten size doesn't match, print debug info and adjust
            if x.shape[1] != self.flatten_size:
                print(
                    f"Warning: Expected flatten size {self.flatten_size}, got {x.shape[1]}")
                # Adapt for this input
                if not hasattr(self, 'fc_adapted') or self.fc_adapted.in_features != x.shape[1]:
                    self.fc_adapted = nn.Linear(
                        x.shape[1], 512).to(self.device)
                    # Initialize weights
                    nn.init.xavier_uniform_(self.fc_adapted.weight)
                    if self.fc_adapted.bias is not None:
                        self.fc_adapted.bias.data.fill_(0.01)
                x = F.relu(self.fc_adapted(x))
            else:
                x = F.relu(self.fc(x))

            advantage = self.advantage(x)
            value = self.value(x)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))

            return q

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.01)
