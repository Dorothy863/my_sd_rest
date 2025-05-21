import torch
from torch import nn
import os
from torch.nn import functional as F

class Mapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
                 adapter_dim: int = 512,
                 gate_type: str = 'softmax'  # 'sigmoid' or 'softmax'
                ):
        super().__init__()
        self._is_mapper = True
        self.num_words = num_words
        self.gate_type = gate_type
        
        # 兼容性适配器参数
        self.adapter_dim = adapter_dim
        
        # 门控初始化参数
        self.gate_alpha = nn.Parameter(torch.ones(num_words))
        self.gate_beta = nn.Parameter(torch.ones(num_words))
        # 通道对齐适配层
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, input_dim)
        )
        # 模块初始化结构改进
        for i in range(self.num_words):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, output_dim)))

        for i in range(self.num_words):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, output_dim)))

    def forward(self, embs):
        hidden_states = ()

        for i in range(self.num_words):
            hidden_state = getattr(self, f"mapping_{i}")(embs[:, i].unsqueeze(1))
            hidden_states += (hidden_state, )

        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states

class EnhanceMapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
                 adapter_dim: int = 256,
                 gate_type: str = 'softmax'  # 'sigmoid' or 'softmax'
                ):
        super().__init__()
        self._is_mapper = True
        self.num_words = num_words
        self.gate_type = gate_type
        
        # 兼容性适配器参数
        self.adapter_dim = adapter_dim
        
        # 门控初始化参数
        self.gate_alpha = nn.Parameter(torch.ones(num_words))
        self.gate_beta = nn.Parameter(torch.ones(num_words))
        # 通道对齐适配层
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, input_dim)
        )
        # 模块初始化结构改进
        for i in range(self.num_words):
            # 主特征路径
            setattr(self, f'mapping_{i}', self._build_mapping_block(input_dim, output_dim))
            # Patch特征路径
            setattr(self, f'mapping_patch_{i}', self._build_mapping_block(input_dim, output_dim))
            
    def _build_mapping_block(self, in_dim, out_dim):
        return nn.Sequential(
            GateUnit(in_dim, 1280),  # 带门控的初始转换
            AdapterLayer(1280, self.adapter_dim),
            nn.LayerNorm(1280),
            nn.LeakyReLU(),
            GateUnit(1280, 1280),
            AdapterLayer(1280, self.adapter_dim),
            nn.LayerNorm(1280),
            nn.LeakyReLU(),
            GateUnit(1280, 1280),
            AdapterLayer(1280, self.adapter_dim),
            nn.LayerNorm(1280),
            nn.LeakyReLU(),
            nn.Linear(1280, out_dim)
        )
    def _gate_mechanism(self, main, patch, idx):
        # 动态门控权重生成
        if self.gate_type == 'sigmoid':
            gate = torch.sigmoid(self.gate_alpha[idx] * main + self.gate_beta[idx] * patch)
            return gate * main + (1 - gate) * patch
        elif self.gate_type == 'softmax':
            combined = torch.stack([main, patch], dim=1)
            weights = F.softmax(torch.stack([self.gate_alpha[idx], self.gate_beta[idx]]), dim=0)
            return (weights[0] * main + weights[1] * patch)
            
    def forward(self, embs):
        # 特征适配阶段
        embs = embs[0].to(next(self.parameters()).device)
        embs = self.feature_adapter(embs)  # 全局特征适配
        
        hidden_states = []
        for i in range(self.num_words):
            main_feat = getattr(self, f'mapping_{i}')(embs[:, :1])
            patch_feat = getattr(self, f'mapping_patch_{i}')(embs[:, 1:]).mean(dim=1, keepdim=True)
            
            # 门控融合
            gated_feat = self._gate_mechanism(main_feat, patch_feat, i)
            
            hidden_states.append(gated_feat)
            
        return torch.cat(hidden_states, dim=1)
# 门控单元组件
class GateUnit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        return self.linear(x) * self.sigmoid(self.gate(x))
# 兼容性适配层
class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim, adapter_dim):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, adapter_dim)
        self.up_proj = nn.Linear(adapter_dim, hidden_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + x  # 残差连接保持稳定性
    
class FineMapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
    ):
        super(FineMapper, self).__init__()

        self.num_words = num_words

        for i in range(self.num_words):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, 1280),
                                                        nn.LayerNorm(1280),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1280, output_dim)))

    def forward(self, embs):
        hidden_states = ()

        for i in range(self.num_words):
            hidden_state = getattr(self, f"mapping_{i}")(embs[:, i].unsqueeze(1))
            hidden_states += (hidden_state, )

        hidden_states = torch.cat(hidden_states, dim=1)

        return hidden_states

if __name__ == "__main__":
    from transformers import CLIPVisionModel
    from PIL import Image

    clip_path = '/workspace/sd_models/CLIP-ViT-H-14-laion2B-s32B-b79K'
    mapper_path = '/workspace/sd_models/my_mapper/mapper_024000.pt'
    sd_path = '/workspace/sd_models/stable-diffusion-2-1-base'

    image_path = './input_image.jpg'
    from processor.image_processor import clip_process
    image = clip_process(image_path)

    image_encoder = CLIPVisionModel.from_pretrained(clip_path)
    image_features = image_encoder(image, output_hidden_states=True)
    image_embeddings = [image_features[0]]
    image_embeddings = [emb.detach() for emb in image_embeddings]

    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
    mapper = Mapper(input_dim=1280, output_dim=1024, num_words=20)
    # 将UNet的交叉注意力层添加到Mapper
    mapper = mapper.prepare_mapper_with_unet(unet)
    mapper.load_state_dict(torch.load(mapper_path, map_location='cuda'))

    inj_embedding = mapper(image_embeddings)
    print("Injected embedding shape:", inj_embedding.shape)