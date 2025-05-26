import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time  # 新增时间模块用于生成唯一文件名

from typing import Union, Optional

def show(image: Union[Image.Image, torch.Tensor, np.ndarray], 
        caption: Optional[str] = None,
        weight=512, height=512) -> None:
    # check if the image is a tensor or a PIL image
    if isinstance(image, torch.Tensor):
        print("🖼️ 传入的是一个张量")
        # if the tensor is on gpu, move it to cpu
        if image.is_cuda:
            image = image.cpu()

        # if it have over 3 channel, squeeze it
        if image.shape[1] > 3:
            image = image[:,:3,:,:]

        # check if the tensor is not float, convert it to float
        if image.dtype != torch.float32 and image.dtype != torch.float64:
            image = image.float()

        if image.min() < -0.7: # if the image is [-1, 1]
            image = (image + 1) / 2

        # convert the tensor to numpy
        try:
            image = image.numpy()
        except:
            image = image.detach().numpy()

    elif isinstance(image, Image.Image):
        print("🖼️ 传入的是一个PIL图像")
        image = np.array(image)
    else:
        print("🖼️ 传入的是一个numpy数组")

    # Remove single-dimensional entries from the shape
    image = np.squeeze(image)

    # if the image shape is (c, h, w), change it to (h, w, c)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]

    # display the image
    dpi = 100
    figsize = (weight / dpi, height / dpi)
    fig, ax = plt.subplots(
        figsize=figsize, 
        dpi=dpi,
        frameon=False,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 绘图
    ax.imshow(image)
    ax.axis("off")
    if caption:
        ax.set_title(caption, fontsize=20, y=0.98, pad=20)

    # 自动生成唯一文件名（时间戳 + caption简写）
    timestamp = int(time.time() * 1000)  # 毫秒级时间戳
    if caption:
        safe_caption = caption.replace(" ", "_").replace("/", "_")[:20]  # 清理特殊字符
        filename = f"./{safe_caption}_{timestamp}.png"
    else:
        filename = f"./plot_{timestamp}.png"
    
    # 保存图像并清理内存
    plt.savefig(filename, bbox_inches="tight", pad_inches=0, transparent=True, dpi=dpi)
    plt.close(fig)
    print(f"🎉 图像已保存到: {filename}")  # 输出提示

def print_model_dtypes(model):
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            print(f"Module: {name:<40} | Parameter: {param_name:<15} | dtype: {param.dtype}")

def count_parameters(model, format=False):
    """统计模型总参数和可训练参数
    
    Args:
        model: PyTorch模型
        format: 是否自动格式化为M/B单位
    
    Returns:
        total (int): 总参数数量
        trainable (int): 可训练参数数量
        或根据format返回格式化字符串
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not format:
        return total, trainable
    else:
        def _format(num):
            if num >= 1e9:
                return f"{num/1e9:.2f}B"
            return f"{num/1e6:.2f}M" if num >= 1e6 else f"{num/1e3:.0f}K"
        
        return _format(total), _format(trainable)

def analyze_model_channels(model):
    architecture = {'encoder': {'down_blocks': []}, 
                    'decoder': {'up_blocks': []}}

    # 分析编码器
    for block_idx, down_block in enumerate(model.encoder.down_blocks):
        block_info = {
            'output_channels': None,
            'resnets': [],
            'downsampler': None
        }
        
        # 检测下采样块最终输出通道
        if hasattr(down_block, 'resnets'):
            for resnet in down_block.resnets:
                conv2 = resnet.conv2
                block_info['resnets'].append(conv2.out_channels if hasattr(conv2, 'out_channels') else conv2.weight.shape[0])
        
        if hasattr(down_block, 'downsamplers') and down_block.downsamplers:
            downsample_conv = down_block.downsamplers[0].conv
            block_info['downsampler'] = downsample_conv.out_channels if hasattr(downsample_conv, 'out_channels') else downsample_conv.weight.shape[0]
            block_info['output_channels'] = block_info['downsampler'] if block_info['downsampler'] else block_info['resnets'][-1]
        
        architecture['encoder']['down_blocks'].append(block_info)

    # 分析解码器 
    for block_idx, up_block in enumerate(model.decoder.up_blocks):
        block_info = {
            'input_channels': None,
            'resnets': [],
            'upsampler': None
        }
        
        # 获取第一个resnet的第一个卷积输入通道作为该block输入
        first_resnet_conv = up_block.resnets[0].conv1
        block_info['input_channels'] = first_resnet_conv.in_channels if hasattr(first_resnet_conv, 'in_channels') else first_resnet_conv.weight.shape[1]
        
        architecture['decoder']['up_blocks'].append(block_info)

    return architecture