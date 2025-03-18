from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from PIL import Image
import torch


# config model path
clip_path = '/workspace/sd_models/CLIP-ViT-H-14-laion2B-s32B-b79K'
mapper_path = '/workspace/sd_models/my_mapper/mapper_024000.pt'
sd_path = '/workspace/sd_models/stable-diffusion-2-1-base'
controlnet_path = None



# init clip model
CLIPVisionModel.from_pretrained(clip_path)

# init sd components
vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")


# init controlnet
if controlnet_path is not None:
# if controlnet_path is not None, load controlnet from controlnet_path
    controlnet = ControlNetModel.from_pretrained(controlnet_path)
else:
    # init controlnet by unet, if controlnet_path is None
    controlnet = ControlNetModel.from_unet(unet)