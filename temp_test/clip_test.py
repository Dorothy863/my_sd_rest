
from transformers import CLIPVisionModel
import torch
import torch.nn.functional as F
from PIL import Image
from processor.image_processor import process_image

clip_path = '/workspace/sd_models/CLIP-ViT-H-14-laion2B-s32B-b79K'
clip_vision = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=torch.float16).to('cuda')

clip_vision.eval()

# 读取LQ文件
lq_path = "/workspace/datasets/SD_Rest/val/motion-blurry/LQ/GOPR0384_11_00_000001.png"
lq_image = process_image(lq_path).to('cuda')

image = F.interpolate(lq_image, (224, 224), mode='bilinear')

# encode image by clip
image_features = clip_vision(image, output_hidden_states=True)

image_embeddings = [image_features[0]]
image_embeddings = [emb.detach() for emb in image_embeddings]

print("done.")