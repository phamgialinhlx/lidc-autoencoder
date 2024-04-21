from src.models.vq_gan_3d_module import VQGAN
from src.models.vq_gan_3d_seg_head import VQGANSegHead
from torch import nn
from IPython import embed

def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
    try:
        ae = VQGAN.load_from_checkpoint(ckpt_path, map_location=map_location)
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
    return ae

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
