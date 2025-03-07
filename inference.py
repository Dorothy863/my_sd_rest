from diffusers import StableDiffusionPipeline, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer
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
        
        # 处理预编译embedding
        if embedding_cache_key:
            prompt_embeds, negative_prompt_embeds = self.precompiled_embeddings[embedding_cache_key]
            prompt = None  # 必须清空原始prompt

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

seed = 114514
generator = torch.Generator(device='cuda').manual_seed(seed)

# 使用半精度优化
pipeline = CustomStableDiffusion.from_pretrained(
    "/workspace/sd_models/stable-diffusion-2-1-base",
    torch_dtype=torch.float32  
).to("cuda")

# 启用所有优化
pipeline.add_extension(
    controlnet=None,
    enable_xformers=True,
)


# 读取LQ文件
from processor.image_processor import process_image
lq_path = "/workspace/datasets/SD_Rest/val/motion-blurry/LQ/GOPR0384_11_00_000001.png"
# lq_path = "/workspace/datasets/SD_Rest/val/motion-blurry/GT/GOPR0384_11_00_000001.png"
lq_image = process_image(lq_path).to('cuda')

# save the tensor image
# convert tensor to PIL image
input_image = pipeline.image_processor.postprocess(lq_image.detach(), output_type="pil", do_denormalize=[True] * lq_image.shape[0])[0]
input_image.save("input_image.jpg")

# 生成LQ的VAE嵌入
vae_embedding = pipeline.vae.encode(
    lq_image,
)
begin_latents = vae_embedding[0].sample() * pipeline.vae.config.scaling_factor

begin_image = pipeline.vae.decode(begin_latents / pipeline.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
begin_image = pipeline.image_processor.postprocess(begin_image.detach(), output_type="pil", do_denormalize=[True] * begin_image.shape[0])[0]
begin_image = begin_image.save("begin_image.jpg")



# 生成时进一步优化
image = pipeline(
    "people walking on a city street with a lot of buildings",
    num_inference_steps=28,
    guidance_scale=5,
    latents=begin_latents,
)
# 保存生成结果
image.images[0].save("generated_image.jpg")

