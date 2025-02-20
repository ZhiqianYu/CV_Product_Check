"""
文件: train_pipeline.py
功能: 从 config.yaml 中读取训练超参和数据路径, 
     调用 datasets/mvtec.py 加载数据,
     构建Autoencoder, 选择损失函数, 完成训练并保存
被引用: app.py 中 "开始训练" 按钮后调用
"""

import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.gpu_info import get_gpu_info

from datasets.mvtec import load_screw_dataset
from models.m_autoencoder import build_autoencoder
import torchvision.transforms.functional as F_tv

try:
    from pytorch_ssim import SSIM
except ImportError:
    SSIM = None

def get_loss_function(loss_type="MSE"):
    if loss_type.upper() == "MSE":
        return nn.MSELoss()
    elif loss_type.upper() == "L1":
        return nn.L1Loss()
    elif loss_type.upper() == "SSIM":
        if SSIM is not None:
            return SSIM()
        else:
            raise ValueError("SSIM库未安装, 无法使用SSIM")
    elif loss_type.upper() == "MIXED":
        if SSIM is None:
            raise ValueError("SSIM库未安装, 无法使用Mixed中SSIM")
        mse_loss = nn.MSELoss()
        ssim_loss = SSIM()

        def mixed_loss(recon, target):
            return 0.5 * mse_loss(recon, target) + 0.5 * (1 - ssim_loss(recon, target))
        return mixed_loss
    else:
        return nn.MSELoss()  # 默认用MSE

def train_autoencoder(config_path="configs/config.yaml", progress_callback=None):
    """
    主训练函数, 读取config.yaml, 训练Autoencoder
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1) 从 config 中解析相关参数
    dataset_path = config["data"]["dataset_path"]
    image_size   = config["data"]["image_size"]# 若存在回调函数, 则把进度、时间、loss等信息传给app.py
    batch_size   = config["training"]["batch_size"]
    lr           = config["training"]["learning_rate"]
    num_epochs   = config["training"]["num_epochs"]
    loss_type    = config["training"]["loss_function"]

    # 模型参数
    model_name       = config["model"]["name"]
    pretrained       = config["model"]["pretrained"]
    pretrained_path  = config["model"]["pretrained_path"]
    save_path        = config["model"]["save_path"]
    latent_channels  = config["autoencoder"]["latent_channels"]
    decoder_conf     = config["autoencoder"]["decoder"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) 加载数据
    augment_conf = config.get("augment", None)
    train_loader, _, _ = load_screw_dataset(
        dataset_path=dataset_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        augment_config=augment_conf,
        normalize_config={"mean": [0.5]*3, "std":[0.5]*3}
    )

    # 3) 构建Autoencoder
    model = build_autoencoder(
        model_name=model_name,
        pretrained=pretrained,
        latent_channels=latent_channels,
        decoder_config=decoder_conf
    ).to(device)

    # 4) 选择损失与优化器
    criterion = get_loss_function(loss_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 5) 训练循环
    os.makedirs(save_path, exist_ok=True)
    best_loss = float("inf")

    # 用于时间统计
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            recon = model(images)
            loss = criterion(recon, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 2 == 0:
                # === 1) Pick 4 images from this batch (or fewer if batch < 4) ===
                n_show = min(4, images.size(0))
                sample_imgs = images[:n_show]  # shape: (4,3,H,W)

                # === 2) Convert these images back to CPU and undo normalization ===
                # Since your original normalization is mean=0.5, std=0.5,
                # we can invert it by: x_img = x_norm * std + mean
                # Also convert them to PIL or numpy so that Streamlit can display
                sample_list = []
                for i in range(n_show):
                    # each image[i]: shape (3,H,W)
                    img_cpu = sample_imgs[i].detach().cpu()
                    # undo normalization
                    img_cpu = img_cpu * 0.5 + 0.5  # because your mean=0.5,std=0.5
                    # clamp to [0,1] to avoid float rounding
                    img_cpu = img_cpu.clamp(0,1)
                    # convert to PIL
                    pil_img = F_tv.to_pil_image(img_cpu)
                    sample_list.append(pil_img)

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")# 若存在回调函数, 则把进度、时间、loss等信息传给app.py

        gpu_info = get_gpu_info()

        # 计算已经花费的时间
        elapsed_time = time.time() - start_time
        # 粗略估计剩余时间: 
        # (已用时 / 已完成轮数) * (剩余轮数)
        remain_time = (elapsed_time / (epoch+1)) * (num_epochs - (epoch+1))

        gpu_info = get_gpu_info()
        if progress_callback is not None:
            progress_callback(
                epoch+1,      # 当前epoch(从1开始)
                num_epochs,   # 总epoch数
                avg_loss,     # 当前epoch平均loss
                elapsed_time, # 已用时间
                remain_time,   # 预计剩余时间
                gpu_info,     # GPU信息
                aug_images=sample_list,
            )
      
        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, "autoencoder_best.pth"))