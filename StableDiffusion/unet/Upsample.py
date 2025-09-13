"""Upsample module for UNet architecture in Stable Diffusion."""
import torch
from torch import nn
from torch.nn import functional as F

class Upsample(nn.Module):
    """Upsample module using nearest neighbor interpolation followed by a convolution.
    Args:
        channels (int): Number of input and output channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Upsample module.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).
        Returns:
            torch.Tensor: Upsampled tensor of shape (Batch_Size, Channels, Height*2, Width*2).
        """
        # (Batch_Size, Channels, Height, Width) -> (Batch_Size, Channels, Height*2, Width*2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv(x)
        return x
