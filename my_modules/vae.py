from diffusers.models import AutoencoderKL
import torch.nn as nn
import torch

# 这是一个修改过的VAE模型，添加了跳跃连接以增强特征融合 
class ModifiedAutoencoderKL(AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 在编解码器间建立连接通道
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(enc_ch, dec_ch, 1)
            for enc_ch, dec_ch in zip([128, 256, 512], [512, 256, 128])  # 需根据实际通道数调整
        ])

    def get_encoder_features(self, x):
        # 编码器阶段收集特征
        encoder_features = []
        x = self.encoder.conv_in(x)
        for down_block in self.encoder.down_blocks:
            x = down_block(x)
            encoder_features.append(x)
        return encoder_features
    
    def decode_by_middle_and_skip(self, x):

        x = self.encoder.mid_block(x)

        return x

    def forward(self, x):
        # 编码器阶段收集特征
        encoder_features = []
        x = self.encoder.conv_in(x)
        for down_block in self.encoder.down_blocks:
            x = down_block(x)
            encoder_features.append(x)
        
        # 中间模块
        x = self.encoder.mid_block(x)
        
        # 解码器阶段融合特征
        x = self.decoder.conv_in(x)
        for i, up_block in enumerate(self.decoder.up_blocks):
            # 跳跃连接融合（需匹配维度）
            if i < len(encoder_features):
                skip = self.skip_connections[-i-1](encoder_features[-i-1])
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        
        return self.decoder.conv_out(x)
    
if __name__ == "__main__":
    # 加载原始模型
    original_vae = AutoencoderKL.from_pretrained("/workspace/sd_models/stable-diffusion-2-1-base", subfolder="vae")
    # 创建改进模型
    modified_vae = ModifiedAutoencoderKL(**original_vae.config)
    # 智能权重加载（官方推荐方式）
    status = modified_vae.load_state_dict(
        original_vae.state_dict(),
        strict=False  # 允许部分载入
    )
    print(f"成功加载参数：{len(status.missing_keys)}缺失，{len(status.unexpected_keys)}冗余")

    # 参数分组优化（预训练参数小学习率，新增参数大学习率）
    pretrained_params = []
    new_params = []
    for name, param in modified_vae.named_parameters():
        if "skip_connections" in name:
            new_params.append(param)
        else:
            pretrained_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": 1e-6},
        {"params": new_params, "lr": 1e-4}
    ])

    # 混合监督策略示例（结合L1和SSIM）
    loss_fn = lambda x, y: 0.8*torch.nn.L1Loss()(x, y) + 0.2*(1 - ssim(x, y))

    # 特征可视化调试
    with torch.no_grad():
        demo_input = torch.randn(1,3,512,512)
        output = modified_vae(demo_input)
        print(f"特征图尺寸变化：{[f.shape for f in encoder_features]}")

    scaler = torch.cuda.amp.GradScaler()
    with torch.autocast(device_type='cuda'):
        output = modified_vae(demo_input)
        loss = loss_fn(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)




r"""
Args:
    sample (`torch.Tensor`): Input sample.
    sample_posterior (`bool`, *optional*, defaults to `False`):
        Whether to sample from the posterior.
    return_dict (`bool`, *optional*, defaults to `True`):
        Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
"""


r"""
# diffusion vae train step

    # Convert images to latent space
    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(
        dtype=weight_dtype
    )

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

    down_block_res_samples, mid_block_res_sample = controlnet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=controlnet_image,
        return_dict=False,
    )

    # Predict the noise residual
    model_pred = unet(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        down_block_additional_residuals=[
            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
        ],
        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
        return_dict=False,
    )[0]

    # Get the target for loss depending on the prediction type
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    accelerator.backward(loss)
    if accelerator.sync_gradients:
        params_to_clip = controlnet.parameters()
        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad(set_to_none=args.set_grads_to_none)


"""