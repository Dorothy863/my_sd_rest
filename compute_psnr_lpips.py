import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
from tqdm import tqdm  # 新增tqdm导入

# 初始化LPIPS模型
lpips_loss_fn = lpips.LPIPS(net='alex').to('cuda')

# 数据集路径和退化类型配置
dataset_path = "/workspace/sd_models/my_controlnet/validation_20250317_130223"
degration_type = ['hazy', 'low-light', 'blurry', 'raindrop', 'rainy', 'shadowed', 'snowy']

# 图像预处理流程
transform = transforms.Compose([
    transforms.ToTensor()  # 转换为[0,1]范围的Tensor
])

for dt in degration_type:
    deg_dir = os.path.join(dataset_path, dt)
    gt_dir = os.path.join(deg_dir, 'gt')
    
    # 获取所有退化图像路径
    deg_files = glob.glob(os.path.join(deg_dir, '*_0.png'))
    
    # 添加类型进度条
    if not deg_files:
        print(f"[{dt.upper()}] 未找到样本，跳过该类型")
        continue
    
    total_psnr = 0.0
    total_lpips = 0.0
    valid_count = 0
    
    # 为当前退化类型创建进度条
    progress_bar = tqdm(
        deg_files,
        desc=f"Processing {dt.ljust(10)}",  # 对齐显示更美观
        unit='image',
        ncols=100  # 控制进度条宽度
    )
    
    for deg_path in progress_bar:
        # 构造对应的GT路径
        basename = os.path.basename(deg_path).replace('_0.png', '.png')
        gt_path = os.path.join(gt_dir, basename)
        
        # 确保GT文件存在
        if not os.path.exists(gt_path):
            progress_bar.write(f"警告：{gt_path} 不存在，跳过该样本")
            continue
        
        # 加载图像并转换格式
        try:
            img_deg = Image.open(deg_path).convert('RGB')
            img_gt = Image.open(gt_path).convert('RGB')
        except Exception as e:
            progress_bar.write(f"加载 {deg_path} 或 {gt_path} 失败: {e}")
            continue
        
        # 转换为Tensor并添加batch维度
        img_deg_tensor = transform(img_deg).unsqueeze(0).to('cuda')
        img_gt_tensor = transform(img_gt).unsqueeze(0).to('cuda')
        
        # 检查尺寸一致性
        if img_deg_tensor.shape != img_gt_tensor.shape:
            progress_bar.write(f"尺寸不匹配: {deg_path} ({img_deg_tensor.shape}) vs {gt_path} ({img_gt_tensor.shape})")
            continue
        
        # 计算PSNR
        mse = torch.mean((img_deg_tensor * 255 - img_gt_tensor * 255) ** 2)
        psnr = 10 * torch.log10((255.0 ** 2) / mse) if mse != 0 else float('inf')
        total_psnr += psnr.item()
        
        # 计算LPIPS
        with torch.no_grad():
            img_deg_lpips = img_deg_tensor * 2 - 1
            img_gt_lpips = img_gt_tensor * 2 - 1
            lpips_val = lpips_loss_fn(img_deg_lpips, img_gt_lpips)
        total_lpips += lpips_val.item()
        
        valid_count += 1
        
        # 实时更新进度条附加信息
        progress_bar.set_postfix({
            'PSNR': f"{total_psnr/valid_count:.1f}" if valid_count else 'N/A',
            'LPIPS': f"{total_lpips/valid_count:.3f}" if valid_count else 'N/A'
        })
    
    # 关闭当前类型进度条
    progress_bar.close()
    
    # 输出统计结果
    if valid_count > 0:
        avg_psnr = total_psnr / valid_count
        avg_lpips = total_lpips / valid_count
        print(f"\n[{dt.upper()}] 最终结果:")
        print(f"  有效样本数: {valid_count}/{len(deg_files)}")
        print(f"  PSNR: {avg_psnr:.2f} dB")
        print(f"  LPIPS: {avg_lpips:.4f}\n")
    else:
        print(f"\n[{dt.upper()}] 没有找到有效样本\n")
