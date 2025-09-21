""" Utility functions for loading Stable Diffusion models. """
from stable_diffusion.clip import CLIP
from stable_diffusion.vae import VAE_Encoder
from stable_diffusion.vae import VAE_Decoder
from stable_diffusion.diffusion import Diffusion

from stable_diffusion.utils import load_from_standard_weights

def preload_models_from_standard_weights(ckpt_path, device):
    """ Preload Stable Diffusion models from standard weights.
    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (str): Device to load the models onto.
    Returns:
        dict: Dictionary containing the pre-loaded models: "clip", "diffusion", "encoder", "decoder".
    """
    state_dict = load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }
