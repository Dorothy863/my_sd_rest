import torch
from torch import nn
import os
from torch.nn import functional as F

# modified
class Mapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
    ):
        super(Mapper, self).__init__()

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

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1280),
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
        embs = embs[0]

        for i in range(self.num_words):
            hidden_state = getattr(self, f'mapping_{i}')(embs[:, :1]) + getattr(self, f'mapping_patch_{i}')(embs[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states

    
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

class FeatureProjector(nn.Module):
    def __init__(self,
                 vis_dim=1280,
                 txt_dim=1024,
                 num_queries=20,
                 n_heads=8,
                 dropout=0.1):
        super().__init__()
        self.d_model = txt_dim
        
        # 视觉特征预处理
        self.vis_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, txt_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 可学习的查询向量
        self.query_embeds = nn.Parameter(
            torch.randn(1, num_queries, txt_dim)  # 初始化为随机张量
        )
        
        # Transformer解码器层
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=txt_dim,
                nhead=n_heads,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )
        
        # 输出适配器
        self.output_norm = nn.LayerNorm(txt_dim)
        self.output_proj = nn.Linear(txt_dim, txt_dim)
    def forward(self, visual_features):
        """
        Args:
            visual_features: [B, 257, 1280]
        Returns:
            text_embeddings: [B, N, 1024]
        """
        if isinstance(visual_features, list) : # if visual_features is List:
            visual_features = visual_features[0]

        B = visual_features.size(0)
        
        # 1. 视觉特征预处理
        vis_features = self.vis_proj(visual_features)  # [B, 257, 1024]
        
        # 2. 准备查询向量
        query = self.query_embeds.expand(B, -1, -1)  # [B, 20, 1024]
        
        # 3. 生成文本控制嵌入
        text_embeds = self.decoder(
            tgt=query,
            memory=vis_features
        )  # [B, 20, 1024]
        
        # 4. 后处理
        text_embeds = self.output_proj(
            self.output_norm(text_embeds)
        )
        
        return text_embeds

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