"""VAE Attention Block
Attention block used in the VAE encoder and decoder.
"""
from torch import nn

from stable_diffusion.attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    """An attention block used in the VAE encoder and decoder.
    Args:
        channels (int): The number of input channels.
    """
    def __init__(self, channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x):
        """Forward pass of the attention block.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Features, Height, Width).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Features, Height, Width).
        """
        # x: (Batch_Size, Features, Height, Width)

        residue = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        n, c, h, w = x.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features).
        # Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention(x)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
        x += residue

        # (Batch_Size, Features, Height, Width)
        return x
