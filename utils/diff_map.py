import torch
import numpy as np
from PIL import Image

def compute_diff_map(original_tensor, recon_tensor):
    """
    计算原图与重建图的差异图(这里以像素MSE作为示例),
    返回一个可以用PIL显示的单通道灰度差异图。
    
    参数:
        original_tensor: (1, C, H, W) 归一化后的原图tensor
        recon_tensor:    (1, C, H, W) 重建或预测输出的tensor
    返回:
        diff_img: PIL Image格式的灰度图, 用于差异可视化
    """
    # 1) 计算逐像素差异，可使用MSE或绝对误差
    diff_map = (original_tensor - recon_tensor) ** 2  # (1, C, H, W)
    # 2) 对通道取均值，得到(1, 1, H, W)
    diff_map = torch.mean(diff_map, dim=1, keepdim=True)
    # 3) 转为CPU数组 (H, W)
    diff_map = diff_map.squeeze(0).squeeze(0).detach().cpu().numpy()
    # 4) 归一化到[0,1]，防止差异过大导致全白
    max_val = diff_map.max()
    if max_val > 0:
        diff_map = diff_map / max_val
    # 5) 转为0~255的灰度图
    diff_map = (diff_map * 255).astype(np.uint8)
    # 6) 将灰度图转为RGB图像，并将差异显示为蓝色
    diff_img = Image.fromarray(diff_map, mode='L').convert('RGB')
    diff_img_np = np.array(diff_img)
    diff_img_np[:, :, 0] = 0  # R channel
    diff_img_np[:, :, 2] = 0  # G channel
    diff_img_np[:, :, 0] = diff_map  # B channel
    diff_img = Image.fromarray(diff_img_np)
    return diff_img
