"""UNET_OutputLayer: This module contains the implementation 
of the output layer of the UNet architecture used in Stable Diffusion."""
import torch
from torch import nn
from torch.nn import functional as F

class UNET_OutputLayer(nn.Module):
    """UNET output layer.
    Args:
        in_channels (int): Number of input channels. Default: 320.
        out_channels (int): Number of output channels. Default: 4.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm  = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv       = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        """Forward pass of the UNET output layer.
        Args:
              x (torch.Tensor): Input tensor of shape (Batch_Size, 320, Height / 8, Width / 8).
        Returns:
            torch.FloatTensor: Output tensor of shape (Batch_Size, 4, Height / 8, Width / 8).
        """
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        x = F.silu(x)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)

        # (Batch_Size, 4, Height / 8, Width / 8)
        return x
