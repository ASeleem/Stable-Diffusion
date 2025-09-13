"""UNet architecture for Stable Diffusion."""
import torch
from torch import nn

from stable_diffusion.unet import (
    UNET_ResidualBlock,
    UNET_AttentionBlock,
    SwitchSequential,
    Upsample
)

class UNET(nn.Module):
    """UNET architecture for Stable Diffusion.
    The architecture consists of an encoder, a bottleneck, and a decoder.
    It uses residual blocks and attention blocks at various stages.
    """
    def __init__(self):
        super().__init__()

        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(
                nn.Conv2d(4, 320, kernel_size=3, padding=1)
            ),
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(
                UNET_ResidualBlock(320, 320),
                UNET_AttentionBlock(8, 40)
            ),
            SwitchSequential(
                UNET_ResidualBlock(320, 320),
                UNET_AttentionBlock(8, 40)
            ),

            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)
            ),
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                UNET_ResidualBlock(320, 640),
                UNET_AttentionBlock(8, 80)
            ),
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                UNET_ResidualBlock(640, 640),
                UNET_AttentionBlock(8, 80)
            ),

            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(
                nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)
            ),
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                UNET_ResidualBlock(640, 1280),
                UNET_AttentionBlock(8, 160)
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280),
                UNET_AttentionBlock(8, 160)
            ),

            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(
                nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)
            ),
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280)
            ),
            SwitchSequential(
                UNET_ResidualBlock(1280, 1280)
            ),
        ])
        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280)
        )

        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280)
            ),

            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                Upsample(1280)
            ),

            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                UNET_AttentionBlock(8, 160)
            ),
            SwitchSequential(
                UNET_ResidualBlock(2560, 1280),
                UNET_AttentionBlock(8, 160)
            ),
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(
                UNET_ResidualBlock(1920, 1280),
                UNET_AttentionBlock(8, 160),
                Upsample(1280)
            ),

            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                UNET_ResidualBlock(1920, 640),
                UNET_AttentionBlock(8, 80)
            ),
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(
                UNET_ResidualBlock(1280, 640),
                UNET_AttentionBlock(8, 80)
            ),
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(
                UNET_ResidualBlock(960, 640),
                UNET_AttentionBlock(8, 80),
                Upsample(640)
            ),

            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(
                UNET_ResidualBlock(960, 320),
                UNET_AttentionBlock(8, 40)
            ),

            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(
                UNET_ResidualBlock(640, 320),
                UNET_AttentionBlock(8, 40)
            ),
            SwitchSequential(
                UNET_ResidualBlock(640, 320),
                UNET_AttentionBlock(8, 40)
            )
        ])

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 4, Height / 8, Width / 8).
            context (torch.Tensor): Context tensor for cross-attention of shape (Batch_Size, Seq_Len, Dim).
            time (torch.Tensor): Time tensor for time embeddings of shape (1, 1280).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, 320, Height / 8, Width / 8).
        """
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 1280)

        skip_connections = []
        for layer in self.encoders:
            x = layer(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layer in self.decoders:
             # Since we always concat with the skip connection of the encoder,
             # the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x, context, time)
        # (Batch_Size, 320, Height / 8, Width / 8)
        return x
