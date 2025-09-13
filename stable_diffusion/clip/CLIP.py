"""CLIP text encoder module.
"""
import torch
from torch import nn

from stable_diffusion.clip import CLIPEmbedding
from stable_diffusion.clip import CLIPLayer

class CLIP(nn.Module):
    """CLIP text encoder.
    Args:
        vocab_size (int): Size of the vocabulary. Default: 49408.
        embed_dim (int): Dimension of the embedding space. Default: 768.
        max_seq_len (int): Maximum sequence length. Default: 77.
        n_heads (int): Number of attention heads. Default: 12.
        n_layers (int): Number of transformer layers. Default: 12.
    """
    def __init__(self, vocab_size: int = 49408, embed_dim: int = 768, max_seq_len: int = 77,
                 n_heads: int = 12, n_layers: int = 12):
        super().__init__()
        self.embedding = CLIPEmbedding(vocab_size, embed_dim, max_seq_len)

        self.transformer = nn.ModuleList([
            CLIPLayer(n_heads, embed_dim) for _ in range(n_layers)
            ])

        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.FloatTensor:
        """Forward pass of the CLIP text encoder.
        Args:
            tokens (torch.Tensor): Input token tensor of shape (Batch_Size, Seq_Len).
        Returns:
            torch.FloatTensor: Output tensor of shape (Batch_Size, Seq_Len, Dim).
        """
        # Ensure tokens are of type long
        tokens = tokens.type(torch.long)

        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        state = self.embedding(tokens)

        # Apply transformer blocks
        for block in self.transformer:
            # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
            state = block(state)
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.layernorm(state)

        # (Batch_Size, Seq_Len, Dim)
        return output
