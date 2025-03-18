from clip_mapper.mapper import Mapper
import torch

from typing import Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.models.clip.configuration_clip import CLIPTextConfig
from transformers.models.clip.modeling_clip import CLIP_TEXT_INPUTS_DOCSTRING, _expand_mask


@add_start_docstrings_to_model_forward(CLIP_TEXT_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=CLIPTextConfig)
def inj_forward_text(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPooling]:
    r"""
    Returns:
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is None:
        raise ValueError("You have to specify either input_ids")

    r_input_ids = input_ids['input_ids']
    if 'inj_embedding' in input_ids:
        inj_embedding = input_ids['inj_embedding']
        inj_index = input_ids['inj_index']
    else:
        inj_embedding = None
        inj_index = None

    input_shape = r_input_ids.size()
    r_input_ids = r_input_ids.view(-1, input_shape[-1])

    inputs_embeds = self.embeddings.token_embedding(r_input_ids)
    new_inputs_embeds = inputs_embeds.clone()
    if inj_embedding is not None:
        emb_length = inj_embedding.shape[1]
        for bsz, idx in enumerate(inj_index):
            lll = new_inputs_embeds[bsz, idx+emb_length:].shape[0]
            new_inputs_embeds[bsz, idx+emb_length:] = inputs_embeds[bsz, idx+1:idx+1+lll]
            new_inputs_embeds[bsz, idx:idx+emb_length] = inj_embedding[bsz]

    hidden_states = self.embeddings(input_ids=r_input_ids, position_ids=position_ids, inputs_embeds=new_inputs_embeds)

    bsz, seq_len = input_shape
    # CLIP's text model uses causal mask, prepare it here.
    # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
    causal_attention_mask = _build_causal_attention_mask(bsz, seq_len, hidden_states.dtype).to(
        hidden_states.device
    )
    # expand attention_mask
    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

    encoder_outputs = self.encoder(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = encoder_outputs[0]
    last_hidden_state = self.final_layer_norm(last_hidden_state)

    # text_embeds.shape = [batch_size, sequence_length, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=r_input_ids.device), r_input_ids.to(torch.int).argmax(dim=-1)
    ]

    if not return_dict:
        return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,
        pooler_output=pooled_output,
        hidden_states=encoder_outputs.hidden_states,
        attentions=encoder_outputs.attentions,
    )

def _build_causal_attention_mask(bsz, seq_len, dtype):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype)
    mask.fill_(torch.tensor(torch.finfo(dtype).min))
    mask.triu_(1)  # zero out the lower diagonal
    mask = mask.unsqueeze(1)  # expand mask
    return mask

if __name__ == "__main__":
    from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
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

    del image_encoder

    from diffusers import UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet")
    mapper = Mapper(input_dim=1280, output_dim=1024, num_words=20)
    # 将UNet的交叉注意力层添加到Mapper
    mapper = mapper.prepare_mapper_with_unet(unet)
    mapper.load_state_dict(torch.load(mapper_path, map_location='cuda'))

    inj_embedding = mapper(image_embeddings)

    del mapper

    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    """# replace the forward method of the text encoder to inject the word embedding
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = inj_forward_text"""

    # 替换前向的更稳健方式
    orig_forward = text_encoder.text_model.forward
    def custom_forward(*args, **kwargs):
        if isinstance(kwargs['input_ids'], dict):
            if "inj_embedding" in kwargs['input_ids']:
                return inj_forward_text(*args, **kwargs)
            else:
                raise ValueError("inj_embedding not found in input_ids")
        else:
            return orig_forward(**kwargs)
    for _module in text_encoder.modules():
        if _module.__class__.__name__ == "CLIPTextTransformer":
            _module.__class__.__call__ = custom_forward
    # text_encoder.text_model.forward = custom_forward


    batch = {}

    text = "a photo of {}"
    placeholder = "S"
    text = text.format(placeholder)

    placeholder_index = 0
    words = text.strip().split(' ')
    for idx, word in enumerate(words):
        if word == placeholder:
            placeholder_index = idx + 1

    batch["index"] = torch.tensor(placeholder_index).unsqueeze(0)
    batch["input_ids"] = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0].unsqueeze(0)

    encoder_hidden_states = text_encoder({'input_ids': batch["input_ids"],
                                            "inj_embedding": inj_embedding,
                                            "inj_index": batch["index"].detach()})[0]

    print("Injected embedding shape:", inj_embedding.shape)

    from diffusers import StableDiffusionPipeline, ControlNetModel
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder")
    pipeline = StableDiffusionPipeline.from_pretrained(sd_path,
                                                       # vae=vae,
                                                       text_encoder=text_encoder,
                                                       unet=unet
                                                       ).to("cuda")

    print('hello')

    # 显存优化配置
    if False:
        pipeline.enable_attention_slicing()
    
    if False:
        pipeline.enable_vae_slicing()
        
    if False:
        pipeline.enable_model_cpu_offload()
        
    try:
        if True:
            pipeline.enable_xformers_memory_efficient_attention()
    except ImportError:
        print("xformers未安装，跳过初始化。建议安装以获得更好性能：pip install xformers")

    image = pipeline(
    # prompt = "people walking on a city street with a lot of buildings",
    num_inference_steps=28,
    guidance_scale=5,
    prompt_embeds=encoder_hidden_states,
    )

    # 保存生成结果
    image.images[0].save("generated_image.jpg")


    print("Generated image saved as 'generated_image.jpg'")
