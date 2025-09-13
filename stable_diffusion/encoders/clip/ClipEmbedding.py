"""CLIP text embedding layer."""
import torch
from torch import nn

class ClipEmbedding(nn.Module):
    """CLIP text embedding layer.
    Args:
        n_vocab (int): Size of the vocabulary.
        n_embed (int): Dimensionality of the embeddings.
        n_token (int): Maximum number of tokens in the input sequence.
    """
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)

        # A learnable weight matrix encodes the postion information for each token
        self.position_embedding = nn.Parameter(torch.zeros(1, n_token, n_embed))

    def forward(self, token):
        """Forward pass of the CLIP text embedding layer.
        Args:
            token (torch.Tensor): Input token ids. Shape: (Batch_Size, Seq_Len)
        Returns:
            torch.Tensor: Embedded tokens with positional encoding. Shape: (Batch_Size, Seq_Len, Dim)
        """
        # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        x = self.token_embedding(token)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        x += self.position_embedding
        return x
