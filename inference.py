from diffusers import StableDiffusionPipeline, ControlNetModel, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel
import torch
from typing import Optional, Dict, Any, Union, List, Callable
from PIL import Image
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

class CustomStableDiffusion(StableDiffusionPipeline):
    """继承官方Pipeline并进行定制化扩展"""
    
    def add_extension(
        self,
        controlnet: Optional[ControlNetModel] = None,
        precompiled_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        enable_xformers: bool = True,  # 新增xformers开关
        enable_slicing: bool = False,    # 新增attention slicing开关
        enable_vae_slicing: bool = False, # 新增VAE切片支持
        enable_model_cpu_offload: bool = False, # 新增模型卸载支持
    ):
        # 显存优化配置
        if enable_slicing:
            self.enable_attention_slicing()
        
        if enable_vae_slicing:
            self.enable_vae_slicing()
            
        if enable_model_cpu_offload:
            self.enable_model_cpu_offload()
            
        try:
            if enable_xformers:
                self.enable_xformers_memory_efficient_attention()
        except ImportError:
            print("xformers未安装，跳过初始化。建议安装以获得更好性能：pip install xformers")

        # ControlNet扩展
        if controlnet:
            self.controlnet = controlnet.to(self.device)
            # 在此添加ControlNet与UNet的融合逻辑
        
        # 预编译embedding初始化
        self.precompiled_embeddings = precompiled_embeddings or {}

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        cache_key: Optional[str] = None,  # 新增缓存键参数
        **kwargs,
    ):
        """扩展文本编码支持预编译embedding"""
        if cache_key and cache_key in self.precompiled_embeddings:
            return self.precompiled_embeddings[cache_key]
            
        # 调用父类原始编码逻辑
        prompt_embeds, negative_prompt_embeds = super().encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            lora_scale,
            clip_skip,
            **kwargs,
        )
        
        # 缓存计算结果
        if cache_key:
            self.precompiled_embeddings[cache_key] = (prompt_embeds, negative_prompt_embeds)
            
        return prompt_embeds, negative_prompt_embeds

    def save_text_embeddings(self, cache_path: str):
        """保存预编译embedding"""
        torch.save(self.precompiled_embeddings, cache_path)

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        latents: Optional[torch.Tensor] = None,
        # 新增扩展参数
        control_image: Optional[torch.Tensor] = None,
        embedding_cache_key: Optional[str] = None,
        **kwargs,
    ):
        
        """扩展调用方法，兼容原生参数与扩展参数"""
        
        # 处理ControlNet输入
        if hasattr(self, 'controlnet') and self.controlnet and control_image is not None:
            control_image = control_image.to(self.device)
            # 在此注入ControlNet处理逻辑
            kwargs.update({"controlnet_cond": control_image})
        
        with autocast(enabled=True):
            # 调用父类实现
            return super().__call__(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                cross_attention_kwargs=cross_attention_kwargs,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
                latents=latents,
                **kwargs
            )

# seed = 114514
# generator = torch.Generator(device='cuda').manual_seed(seed)


sd_path = "/data/coding/upload-data/data/adrive/stable-diffusion-2-1-base"
clip_path = "/data/coding/upload-data/data/adrive/CLIP-ViT-H-14-laion2B-s32B-b79K"
# DIY
vae_path = "/data/coding/ori_checkpoing-26000/vae/"


vae = AutoencoderKL.from_pretrained(vae_path).to("cuda")
unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet").to("cuda")

text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to("cuda")

mapper_path = "/data/coding/mapper_024000.pt"
from my_modules.mapper import Mapper
mapper = Mapper(input_dim=1280, output_dim=1024, num_words=20).to('cuda')
mapper = mapper.prepare_mapper_with_unet(unet)
mapper.load_state_dict(torch.load(mapper_path))

from use_mapper_test import inj_forward_text
# 替换前向的更稳健方式
orig_text_forward = text_encoder.text_model.forward
def custom_forward(*args, **kwargs):
    if isinstance(kwargs['input_ids'], dict):
        if "inj_embedding" in kwargs['input_ids']:
            return inj_forward_text(*args, **kwargs)
        else:
            raise ValueError("inj_embedding not found in input_ids")
    else:
        return orig_text_forward(**kwargs)
for _module in text_encoder.modules():
    if _module.__class__.__name__ == "CLIPTextTransformer":
        _module.__class__.__call__ = custom_forward

# 使用半精度优化
pipeline = CustomStableDiffusion.from_pretrained(
    sd_path,
    vae=vae,
    unet=unet,
    text_encoder=text_encoder,
    torch_dtype=torch.float32  
).to("cuda")

# 启用所有优化
pipeline.add_extension(
    controlnet=None,
    enable_xformers=True,
)


batch = {}

# 读取LQ文件
from processor.image_processor import process_image
lq_path = "0327.png"
# lq_path = "/workspace/datasets/SD_Rest/val/motion-blurry/GT/GOPR0384_11_00_000001.png"
lq_image = process_image(lq_path).to('cuda')

# save the tensor image
# convert tensor to PIL image
input_image = pipeline.image_processor.postprocess(lq_image.detach(), output_type="pil", do_denormalize=[True] * lq_image.shape[0])[0]
input_image.save("0327_input.png")

# 生成LQ的VAE嵌入
vae_embedding = pipeline.vae.encode(
    lq_image,
)
begin_latents = vae_embedding[0].mode() * pipeline.vae.config.scaling_factor

begin_image = pipeline.vae.decode(begin_latents / pipeline.vae.config.scaling_factor, return_dict=False)[0]
begin_image = pipeline.image_processor.postprocess(begin_image.detach(), output_type="pil", do_denormalize=[True] * begin_image.shape[0])[0]
begin_image = begin_image.save("begin_image.jpg")



clip_image_encoder = CLIPVisionModel.from_pretrained(clip_path).to("cuda")
tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
with torch.no_grad():
    # 提取图像特征
    batch['pixel_values_clip'] = (lq_image + 1) / 2
    batch['pixel_values_clip'] = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])(batch['pixel_values_clip'])
    image_features = [clip_image_encoder(batch['pixel_values_clip'], output_hidden_states=True).last_hidden_state]
    image_embeddings = [emb.detach() for emb in image_features]
    # 通过mapper生成嵌入
    inj_embedding = mapper(image_embeddings)

    batch["input_ids"] = tokenizer(
        "a photo of S",
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    ).input_ids

# 确定占位符位置 (新增部分)
words = "a photo of S".strip().split(' ')
placeholder_index = words.index('S') + 1  # +1 因为CLIP添加了开始token

# 构造注入输入
text_input = {
    "input_ids": batch["input_ids"].to('cuda'),
    "inj_embedding": inj_embedding.to('cuda'),
    "inj_index": torch.tensor([placeholder_index]).to('cuda')  # 假设固定注入位置
}

# 替换原来的文本编码器调用
prompt_embeds = text_encoder(text_input, return_dict=False)[0]

# 生成时进一步优化
image = pipeline(
    # "A seal roars in the sea",
    num_inference_steps=50,
    guidance_scale=5,
    # latents=begin_latents,
    prompt_embeds=prompt_embeds
)
# 保存生成结果
image.images[0].save("generated_image.jpg")

