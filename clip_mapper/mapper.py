import torch
from torch import nn
import os

class Mapper(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_words: int,
    ):
        """
        init use the following code:
            mapper = Mapper(input_dim=1280, output_dim=1024, num_words=args.num_words) # for clip-vit-h is (input_dim=1280, output_dim=1024, num_words=args.num_words), normally num_words=20
            mapper = prepare_mapper_with_unet(mapper, unet) # unet from diffuser
            mapper.load_state_dict(torch.load(args.i2t_mapper_path, map_location='cpu')) # if is pretrained, then use this line to load the pretrained mapper
        """
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

    def prepare_mapper_with_unet(self, unet, device='cuda'):
        # mapper = self
        # 遍历UNet模块，添加cross attention层到Mapper
        for name, module in unet.named_modules():
            if isinstance(module, torch.nn.Module) and 'Attention' in module.__class__.__name__:
                if 'attn1' in name:  # 跳过自注意力层
                    continue
                # 添加to_k_global和to_v_global到mapper
                in_features = module.to_k.in_features
                out_features = module.to_k.out_features
                
                to_k_global = nn.Linear(in_features, out_features, bias=False)
                to_v_global = nn.Linear(in_features, out_features, bias=False)
                
                # 转换为有效的属性名
                layer_name = name.replace('.', '_')
                self.add_module(f'{layer_name}_to_k', to_k_global)
                self.add_module(f'{layer_name}_to_v', to_v_global)
        return self.to(device)

    def forward(self, embs):
        """
        input: embs: torch.Tensor of shape (batch_size, num_patches, emb_dim)
                the input embs from the CLIP Vision model
                such as:
                    image_encoder = CLIPVisionModel.from_pretrained(args.pretrained_clip_model_path)
                    image_features = image_encoder(image, output_hidden_states=True)
                    image_embeddings = [image_features[0]]
                    image_embeddings = [emb.detach() for emb in image_embeddings]
                    inj_embedding = mapper(image_embeddings)

        output: hidden_states: torch.Tensor of shape (batch_size, num_words, output_dim) # for clip-vit-h is (batch_size, num_words, 1024)
        """
        hidden_states = ()
        embs = embs[0].to('cuda')

        for i in range(self.num_words):
            hidden_state = getattr(self, f'mapping_{i}')(embs[:, :1]) + getattr(self, f'mapping_patch_{i}')(embs[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state,)
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
    

def save_progress(mapper, accelerator, args, step=None):
    """
    save the mapper model, use accelerator 
    """
    print("Saving embeddings")

    state_dict = accelerator.unwrap_model(mapper).state_dict()

    if step is not None:
        torch.save(state_dict, os.path.join(args.output_dir, f"mapper_{str(step).zfill(6)}.pt"))
    else:
        torch.save(state_dict, os.path.join(args.output_dir, "mapper.pt"))

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