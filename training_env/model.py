# model.py

import torch
import torch.nn as nn

class DepthwiseSeparableConv1d(nn.Module):
    """A lightweight visual scanner that splits processing into two smaller steps to save memory."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Sub-layer 1: Convolution -> Batch Normalization (stabilizes math) -> ReLU -> Dropout (15%)
        # Dropout randomly turns off 15% of the network during training so it doesn't just memorize the audio.
        self.sub_layer_1 = nn.Sequential(
            DepthwiseSeparableConv1d(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(p=0.15) 
        )
        
        # Sub-layer 2: Convolution -> Batch Normalization[cite: 2]
        self.sub_layer_2 = nn.Sequential(
            DepthwiseSeparableConv1d(channels, channels),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.sub_layer_1(x)
        out = self.sub_layer_2(out)
        
        # Skip connection: adds the raw input directly to the output before the final activation[cite: 2].
        # This helps the network remember long-term phonetic context[cite: 2].
        out += residual
        return self.relu(out)

class CNNVoiceCommandModel(nn.Module):
    def __init__(self, num_classes=57):
        super().__init__()
        # 1. Feature Extractor (Stem Layer)
        # Aggressively compresses the 80,000 raw audio samples down to roughly 500 temporal frames.
        # This acts as an ultra-fast, learned spectrogram that skips heavy visual conversions.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=256, stride=160, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. The Encoder (3 Residual Blocks)
        self.encoder = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        # 3. Classification Head (CTC Projector)
        # A 1x1 Pointwise Convolution (Linear mapping) that maps the features to output channels.
        self.head = nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        # Expects a 1D tensor shape of (1, 80000) representing 5.0 seconds of raw audio.
        x = self.stem(x)
        x = self.encoder(x)
        x = self.head(x)
        return x