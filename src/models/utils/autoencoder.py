from src.models.vq_gan_3d_module import VQGAN
from src.models.vq_gan_3d_seg_head import VQGANSegHead
from src.models.vq_gan_2d_seg_module import VQGANSeg
from src.models.swin_transformer_ae_module import SwinVQGAN
from src.models.multihead_swin_transformer_ae_module import MultiheadSwinVQGAN
from torch import nn
from IPython import embed

# def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
#     try:
#         ae = VQGAN.load_from_checkpoint(ckpt_path, map_location=map_location)
#         if ae.use_ema:
#             ae.model_ema.store(ae.parameters())
#             ae.model_ema.copy_to(ae)
#         if disable_decoder:
#             ae.decoder = None
#     except Exception as e:
#         print(f"Failed to load autoencoder from {ckpt_path}: {e}")
#     if eval:
#         ae.eval()
#         ae.freeze()
#     return ae

def load_autoencoder_seg_head(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
    class Autoencoder(nn.Module):
        def __init__(self, pre_vq_conv, post_vq_conv, encoder, decoder, codebook):
            super().__init__()
            self.pre_vq_conv = pre_vq_conv
            self.post_vq_conv = post_vq_conv
            self.encoder = encoder
            self.decoder = decoder
            self.codebook = codebook
        def forward(self, x):
            z = self.pre_vq_conv(self.encoder(x))
            vq_output = self.codebook(z)
            x = self.decoder(self.post_vq_conv(vq_output['embeddings']))
            return x
    try:
        ae = VQGANSegHead.load_from_checkpoint(ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)
        if disable_decoder:
            ae.decoder = None
    except Exception as e:
        print(f"Failed to load autoencoder from {ckpt_path}: {e}")
    if eval:
        ae.eval()
        ae.freeze()
    return Autoencoder(ae.pre_vq_conv, ae.post_vq_conv, ae.encoder, ae.decoder, ae.codebook)

def load_encoder(ckpt_path, map_location="cuda", disable_decoder=False, eval_mode=False):
    """
    Loads the encoder part of the autoencoder model from the given checkpoint.

    Parameters:
    - ckpt_path (str): Path to the model checkpoint.
    - map_location (str): The device to load the model onto (default is 'cuda').
    - disable_decoder (bool): If True, disables the decoder part of the model (default is False).
    - eval_mode (bool): If True, sets the model to evaluation mode and freezes its parameters (default is False).

    Returns:
    - The encoder part of the loaded model.
    """
    try:
        # Attempt to load the VQGANSeg model from the checkpoint
        ae = VQGANSeg.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)

        # Disable the decoder if requested
        if disable_decoder:
            ae.decoder = None

    except Exception as e:
        print(f"Failed to load VQGANSeg from {ckpt_path}: {e}")
        # Attempt to load the SwinVQGAN model as a fallback
        try:
            ae = SwinVQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
            if ae.use_ema:
                ae.model_ema.store(ae.parameters())
                ae.model_ema.copy_to(ae)
            
            # Disable the decoder if requested
            if disable_decoder:
                ae.decoder = None

        except Exception as e:
            print(f"Failed to load SwinVQGAN from {ckpt_path}: {e}")
            try:
                ae = MultiheadSwinVQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
                if ae.use_ema:
                    ae.model_ema.store(ae.parameters())
                    ae.model_ema.copy_to(ae)
                
                # Disable the decoder if requested
                if disable_decoder:
                    ae.decoder = None

            except Exception as e:
                print(f"Failed to load MultiheadSwinVQGAN from {ckpt_path}: {e}")
                return None

    # Set model to evaluation mode if requested
    if eval_mode:
        ae.eval()
        ae.freeze()

    return ae.encoder

def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval_mode=False):
    """
    Loads the encoder part of the autoencoder model from the given checkpoint.

    Parameters:
    - ckpt_path (str): Path to the model checkpoint.
    - map_location (str): The device to load the model onto (default is 'cuda').
    - disable_decoder (bool): If True, disables the decoder part of the model (default is False).
    - eval_mode (bool): If True, sets the model to evaluation mode and freezes its parameters (default is False).

    Returns:
    - The encoder part of the loaded model.
    """
    try:
        # Attempt to load the VQGANSeg model from the checkpoint
        ae = VQGANSeg.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)

        # Disable the decoder if requested
        if disable_decoder:
            ae.decoder = None

    except Exception as e:
        print(f"Failed to load VQGANSeg from {ckpt_path}: {e}")
        # Attempt to load the SwinVQGAN model as a fallback
        try:
            ae = SwinVQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
            if ae.use_ema:
                ae.model_ema.store(ae.parameters())
                ae.model_ema.copy_to(ae)
            
            # Disable the decoder if requested
            if disable_decoder:
                ae.decoder = None

        except Exception as e:
            print(f"Failed to load SwinVQGAN from {ckpt_path}: {e}")
            try:
                ae = MultiheadSwinVQGAN.load_from_checkpoint(checkpoint_path=ckpt_path, map_location=map_location)
                if ae.use_ema:
                    ae.model_ema.store(ae.parameters())
                    ae.model_ema.copy_to(ae)
                
                # Disable the decoder if requested
                if disable_decoder:
                    ae.decoder = None

            except Exception as e:
                print(f"Failed to load MultiheadSwinVQGAN from {ckpt_path}: {e}")
                return None

    # Set model to evaluation mode if requested
    if eval_mode:
        ae.eval()
        ae.freeze()

    return ae