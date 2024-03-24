from .vq_gan_3d_module import VQGAN
from .diffusion_module import DiffusionModule
from .classification_module import ClassificationModule
# from .segmentation_module import SegmentationModule

def load_autoencoder(ckpt_path, map_location="cuda", disable_decoder=False, eval=True):
    try:
        ae = VQGAN.load_from_checkpoint(ckpt_path, map_location=map_location)
        if ae.use_ema:
            ae.model_ema.store(ae.parameters())
            ae.model_ema.copy_to(ae)
        if disable_decoder:
            ae.decoder = None
        if eval:
            ae.eval()
            ae.freeze()
    except Exception as e:
        print(f"Failed to load autoencoder from {ckpt_path}: {e}")
    return ae
