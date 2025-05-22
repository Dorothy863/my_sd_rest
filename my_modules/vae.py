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
