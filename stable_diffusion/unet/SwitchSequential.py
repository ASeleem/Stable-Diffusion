"""A sequential module that can switch between different types of layers based on their requirements."""
from torch import nn

from stable_diffusion.unet import UNET_AttentionBlock
from stable_diffusion.unet import UNET_ResidualBlock

class SwitchSequential(nn.Sequential):
    """A sequential container that can handle layers with different forward method signatures.
    It supports layers that require additional arguments such as context or time embeddings.
    """
    def forward(self, x, context, time):
        """Forward pass through the sequential container.
        Args:
            x (torch.Tensor): Input tensor.
            context (torch.Tensor): Context tensor for layers that require it.
            time (torch.Tensor): Time tensor for layers that require it.
        Returns:
            torch.Tensor: Output tensor after passing through all layers.
        """
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
