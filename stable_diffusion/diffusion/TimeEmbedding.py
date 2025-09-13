"""
Time Embedding for Stable Diffusion
"""
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    """ Time Embedding Module
    Args:
        n_embd (int): Dimension of the input time embedding.
    """
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        """ Forward function of Time Embedding
        Args:
            x (torch.Tensor): Input time embedding of shape (1, n_embd).
        Returns:
            torch.Tensor: Output time embedding of shape (1, 4 * n_embd).
        """
        # x: (1, 320)
        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)

        # (1, 1280) -> (1, 1280)
        x = F.silu(x)

        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x
