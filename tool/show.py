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