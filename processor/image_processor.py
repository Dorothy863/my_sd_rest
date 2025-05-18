import PIL
from PIL import Image
import torchvision.transforms as transforms

def process_image(input_path: str, target_size=512):
    """
    改进后的图像处理流程，对齐训练数据集预处理
    
    参数：
        input_path: 输入图片路径
        target_size: 目标尺寸（默认512）
    返回：
        torch.Tensor 格式图像数据，形状为 (3, H, W)
    """
    # 对齐数据集的预处理流程
    image_transforms = transforms.Compose([
        
        # 保持长宽比的短边缩放（与原始功能不同但符合数据集逻辑）
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BILINEAR),
        # 中心裁剪到目标尺寸
        transforms.CenterCrop(target_size),
        # 转换为Tensor并归一化到[-1,1]
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 输入范围从[0,1]映射到[-1,1]
    ])
    
    with Image.open(input_path) as img:
        # 强制转换为RGB模式（与dataset预处理一致）
        img = img.convert("RGB")
        
        # 应用完整转换流程
        tensor_img = image_transforms(img)
        
    return tensor_img.unsqueeze(0)  # 添加batch维度


def clip_process(input_path):
    """
    for the clip model, image should be a [224, 224] tensor
    """
    image = Image.open(input_path)
    image = image.convert("RGB")

    size = 224
    W, H = image.size
    interpolation = transforms.InterpolationMode.BILINEAR
    # 定义通用预处理（排除 RandomCrop）
    process = []

    """# 需要裁剪时
    if H >= size and W >= size:
        # 获取随机裁剪参数
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(size, size)
        )
        
        # 对两个图像应用相同的裁剪坐标
        process += transforms.Lambda(lambda x: transforms.functional.crop(x, i, j, h, w)),
    else:"""

    # 小图直接中心裁剪+缩放
    process += transforms.Resize(size, interpolation=interpolation),
    process += transforms.CenterCrop(size),
    
    process += transforms.ToTensor(),
    

    process += transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ),
    process = transforms.Compose(process)


    image = process(image).unsqueeze(0)
    return image
