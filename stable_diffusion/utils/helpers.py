"""
Helper functions for Stable Diffusion models.
"""
import torch

def rescale(x, old_range, new_range, clamp=False):
    """Rescale a tensor from an old range to a new range.
    Args:
        x (torch.Tensor): Input tensor.
        old_range (tuple): Tuple of (min, max) for the old range.
        new_range (tuple): Tuple of (min, max) for the new range.
        clamp (bool): Whether to clamp the output to the new range.
    Returns:
        torch.Tensor: Rescaled tensor.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    """Get the time embedding for a given timestep.
    Args:
        timestep (int): Timestep value.
    Returns:
        torch.Tensor: Time embedding tensor of shape (1, 320).
    """
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
