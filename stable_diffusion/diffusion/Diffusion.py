"""Diffusion Model for Stable Diffusion"""
import torch.nn as nn

from stable_diffusion.diffusion import TimeEmbedding
from stable_diffusion.unet import UNET, UNET_OutputLayer

class Diffusion(nn.Module):
    """ Diffusion Model for Stable Diffusion
    """
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)

    def forward(self, latent, context, time):
        """ Forward function of Diffusion Model
        Args:
            latent (torch.Tensor): Noisy latent tensor of shape (Batch_Size, 4, Height / 8, Width / 8).
            context (torch.Tensor): Text encoder output of shape (Batch_Size, Seq_Len, Dim).
            time (torch.Tensor): Time embedding of shape (1, 320).
        Returns:
            torch.Tensor: Denoised latent tensor of shape (Batch_Size, 4, Height / 8, Width / 8).
        """
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)

        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)

        # (Batch, 4, Height / 8, Width / 8)
        return output
