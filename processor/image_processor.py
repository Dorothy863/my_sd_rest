from PIL import Image
import torchvision.transforms as transforms

def process_image(input_path, target_size=512):
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
