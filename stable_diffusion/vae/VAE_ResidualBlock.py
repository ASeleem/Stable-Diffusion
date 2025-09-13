"""""VAE Residual Block 
"""
from torch import nn
from torch.nn import functional as F

class VAE_ResidualBlock(nn.Module):
    """A residual block used in the VAE encoder and decoder.
    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(num_groups=32, num_channels=in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        """Forward pass of the residual block.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, In_Channels, Height, Width).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Out_Channels, Height, Width).
        """
        # x: (Batch_Size, In_Channels, Height, Width)
        residue = x

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        x = self.groupnorm_1(x)
        x = F.silu(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.conv_1(x)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        residue = self.residual_layer(residue)
        x += residue

        # (Batch_Size, Out_Channels, Height, Width)
        return x
