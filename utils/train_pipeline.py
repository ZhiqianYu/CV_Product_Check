"""
文件: train_pipeline.py
功能: 从 config.yaml 中读取训练超参和数据路径, 
     调用 datasets/mvtec.py 加载数据,
     构建Autoencoder, 选择损失函数, 完成训练并保存
被引用: app.py 中 "开始训练" 按钮后调用
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from utils.gpu_info import get_gpu_info

from datasets.mvtec import load_screw_dataset
from models.m_autoencoder import build_autoencoder

try:
    from pytorch_ssim import SSIM
except ImportError:
    SSIM = None

def get_loss_function(loss_type="MSE"):
    """
    根据字符串类型创建并返回对应的损失函数:
      MSE, L1, SSIM, Mixed等
    """
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

def train_autoencoder(config_path="configs/config.yaml"):
    """
    主训练函数, 读取config.yaml, 训练Autoencoder
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1) 从 config 中解析相关参数
    dataset_path = config["data"]["dataset_path"]
    image_size   = config["data"]["image_size"]
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
    train_loader, _, _ = load_screw_dataset(
        dataset_path=dataset_path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        augment_config={"horizontal_flip": True, "rotation": 15},
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

        avg_loss = running_loss / len(train_loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")
        gpu_info = get_gpu_info()
        print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx} GPU: {gpu_info}")

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_path, "autoencoder_best.pth"))

    print(f"✅ 训练完成! 最佳Loss: {best_loss:.6f}")
    print(f"✅ 模型已保存至: {os.path.join(save_path, 'autoencoder_best.pth')}")
