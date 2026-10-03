# training_env_wake/model.py

import torch
import torch.nn as nn
import torchaudio

class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_p=0.15):
        super().__init__()
        # Depthwise layer: Scans each channel individually
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=kernel_size//2, groups=in_channels, bias=False
        )
        # Pointwise layer: Combines channels
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        # Dropout randomly turns off portions of the network during training
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Learned Front-End Stem (Replaces Mel-Spectrogram)
        # Aggressively compresses raw audio samples into 40 feature channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=40, kernel_size=256, stride=160, bias=False),
            nn.BatchNorm1d(40),
            nn.ReLU()
        )
        
        # 2. Depthwise-Separable Convolutional Stack
        self.network = nn.Sequential(
            DSConvBlock(40, 64, kernel_size=5, dropout_p=0.15),
            DSConvBlock(64, 100, kernel_size=5, dropout_p=0.15),
            DSConvBlock(100, 128, kernel_size=3, dropout_p=0.15),
            
            # 3. Global Average Pooling & Classification Head
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Ensure input has a channel dimension: (Batch, 1, Time)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        return self.network(x)