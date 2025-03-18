import torch
from torch import nn
from typing import Tuple, Dict
from diffusers import ModelMixin, ConfigMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.controlnet import ControlNetConditioningEmbedding, zero_module
from diffusers.models.autoencoders.vae import Decoder 

class ControlNetDecoder(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    def __init__(
        self,
        in_channels: int = 3,
        conditioning_channels: int = 3,
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256),
    ):
        super().__init__()
        
        # 条件嵌入层
        self.cond_embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=block_out_channels[0],
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels
        )
        
        # 复制原始Decoder结构
        self.up_blocks = nn.ModuleList()
        self.controlnet_up_blocks = nn.ModuleList()
        
        # 创建上采样块
        for i, up_block_type in enumerate(up_block_types):
            in_channel = block_out_channels[i]
            out_channel = block_out_channels[i] if i == 0 else block_out_channels[i-1]
            
            up_block = get_up_block(
                up_block_type,
                in_channels=in_channel,
                out_channels=out_channel,
                num_layers=layers_per_block,
                norm_num_groups=norm_num_groups,
                act_fn=act_fn
            )
            self.up_blocks.append(up_block)
            
            # 添加控制块
            control_block = zero_module(nn.Conv2d(out_channel, out_channel, kernel_size=1))
            self.controlnet_up_blocks.append(control_block)
    @classmethod
    def from_decoder(
        cls,
        decoder: Decoder,
        conditioning_channels: int = 3,
        conditioning_embedding_out_channels: Tuple[int] = (16, 32, 96, 256)
    ):
        controlnet = cls(
            in_channels=decoder.config.in_channels,
            up_block_types=decoder.config.up_block_types,
            block_out_channels=decoder.config.block_out_channels,
            layers_per_block=decoder.config.layers_per_block,
            norm_num_groups=decoder.config.norm_num_groups,
            act_fn=decoder.config.act_fn,
            conditioning_channels=conditioning_channels,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels
        )
        
        # 复制原始Decoder权重
        controlnet.up_blocks.load_state_dict(decoder.up_blocks.state_dict())
        return controlnet
    def forward(
        self,
        z: torch.Tensor,
        conditioning: torch.Tensor,
        conditioning_scale: float = 1.0
    ) -> Tuple[Tuple[torch.Tensor], torch.Tensor]:
        # 条件嵌入
        cond_emb = self.cond_embedding(conditioning)
        z = z + cond_emb  # 将条件嵌入合并到输入
        
        # 存储各阶段输出
        up_block_res_samples = ()
        
        # 逐层处理
        for up_block, control_block in zip(self.up_blocks, self.controlnet_up_blocks):
            z = up_block(z)
            control_signal = control_block(z)
            up_block_res_samples += (control_signal * conditioning_scale,)
        
        # 中间块处理
        mid_block_res = self.mid_block(z) * conditioning_scale
        
        return up_block_res_samples, mid_block_res


class ControlNetConditionEncoder(nn.Module):
    def __init__(self, conditioning_channels, embedding_channels=256, num_blocks=(2,2,2), channel_mults=(1,2,4)):
        super().__init__()
        channels = [embedding_channels * mult for mult in channel_mults]
        
        self.down_blocks = nn.ModuleList()
        in_ch = conditioning_channels
        for ch in channels:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, ch, 3, padding=1),
                    nn.GroupNorm(ch//16, ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, ch, 3, padding=1, stride=2),
                    ResnetBlock(ch, ch, temb_channels=None, groups=ch//16)
                )
            )
            in_ch = ch

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = {}
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            features[f"level_{i}"] = x
        return features
