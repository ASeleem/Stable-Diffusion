"""UNET Residual Block
"""
import torch
from torch import nn
import torch.nn.functional as F

class UNET_ResidualBlock(nn.Module):
    """UNET Residual Block.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n_time (int): Dimension of the time embedding. Default: 1280.
    """
    def __init__(self, in_channels: int, out_channels: int, n_time=1280):
        super().__init__()
        self.groupnorm_feature  = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_feature       = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time        = nn.Linear(n_time, out_channels)

        self.groupnorm_merged   = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_merged        = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)


    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNET residual block.
        Args:
            feature (torch.Tensor): Input feature tensor of shape (Batch_Size, In_Channels, Height, Width).
            time (torch.Tensor): Time embedding tensor of shape (Batch_Size, N_Time).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Out_Channels, Height, Width).
        """
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (Batch_Size, N_Time) (e.g., (1, 1280) )
        residue = feature

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)

        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)

        # Add width and height dimensions to time
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        # If in_channels != out_channels, apply a 1x1 convolution to the residue
        output = merged + self.residual_layer(residue)

        return output
