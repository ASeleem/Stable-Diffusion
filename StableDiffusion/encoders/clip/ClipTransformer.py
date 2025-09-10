"""Clip Transformer module for Stable Diffusion."""
import torch
from torch import nn

from StableDiffusion.attention import SelfAttention

class ClipTransformer(nn.Module):
    """Transformer module used in CLIP text encoder.
    Args:
        n_head (int): Number of attention heads.
        n_embd (int): Embedding dimension.
    """
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        # Pre-attention norm
        self.layernorm_1 = nn.LayerNorm(n_embd)

        # Self-Attention
        self.attention = SelfAttention(n_head, n_embd)

        # Pre-FNN norm
        self.layernorm_2 = nn.LayerNorm(n_embd)

        # Feed-forward network
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        """Forward pass of the transformer module.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Seq_Len, Dim).
        Returns:
            torch.Tensor: Output tensor of shape (Batch_Size, Seq_Len, Dim).
        """
        # (Batch_Size, Seq_Len, Dim)
        residue = x

        ## SELF ATTENTION BLOCK ##

        # (batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_1(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.attention(x, causal_mask=True)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x + residue

        ## FEED-FORWARD BLOCK ##
        # Apply a feed-forward layer where the hidden dimention is 4 times the input dimension

        residue = x
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.layernorm_2(x)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 4*Dim)
        x = self.linear_1(x)

        # GELU activation
        # (Batch_Size, Seq_Len, 4*Dim) -> (Batch_Size, Seq_Len, 4*Dim)
        x = x * torch.sigmoid(1.702 * x)

        # (Batch_Size, Seq_Len, 4*Dim) -> (Batch_Size, Seq_Len, Dim)
        x = self.linear_2(x)

        # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x = x + residue

        return x
