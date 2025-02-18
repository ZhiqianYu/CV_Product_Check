"""
文件: models/m_autoencoder.py
功能: 定义一个基于EfficientNet的Autoencoder, 
     并根据 config 里的 decoder 配置动态构建 Decoder 部分。
被引用: train_pipeline.py (训练时), app.py (推理时)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

def get_activation(activation_name: str):
    """
    根据字符串名称返回对应的激活函数模块
    """
    if activation_name.lower() == "relu":
        return nn.ReLU(inplace=True)
    elif activation_name.lower() == "leakyrelu":
        return nn.LeakyReLU(0.2, inplace=True)
    elif activation_name.lower() == "elu":
        return nn.ELU(inplace=True)
    else:
        return nn.ReLU(inplace=True)  # 默认ReLU

class EfficientNetAutoencoder(nn.Module):
    """
    基于EfficientNet骨干的自编码器:
    1) encoder: 使用timm.create_model(...) 构建EfficientNet, 去掉分类层
    2) decoder: 根据 config 里指定的 num_layers, kernel_size, activation 等动态构建多层反卷积
    """
    def __init__(self,
                 model_name="efficientnetv2_s",
                 pretrained=False,
                 pretrained_path=None,
                 latent_channels=1280,
                 decoder_config=None):
        super(EfficientNetAutoencoder, self).__init__()
        if decoder_config is None:
            decoder_config = {
                "num_layers": 3,
                "kernel_size": 4,
                "activation": "ReLU",
                "use_skip_connection": False
            }

        # 1) EfficientNet作为Encoder
        self.encoder = create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 去掉分类层
            global_pool="avg"
        )
        # 输出: (B, latent_channels)

        # 2) 如果需要本地预训练
        if pretrained and pretrained_path is not None and os.path.exists(pretrained_path):
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # 有的预训练权重可能包含一些多余key或形状不匹配，需要做兼容处理
            missing, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
            print(f"[build_autoencoder] Loaded local pretrained weights from {pretrained_path}")
            if len(missing) > 0:
                print("Missing keys:", missing)
            if len(unexpected) > 0:
                print("Unexpected keys:", unexpected)

        # 3) Decoder parms
        self.num_layers = decoder_config["num_layers"]
        self.kernel_size = decoder_config["kernel_size"]
        self.use_skip = decoder_config["use_skip_connection"]
        act_name = decoder_config["activation"]

        # 构建多层反卷积
        layers = []
        in_channels = latent_channels
        for i in range(self.num_layers):
            out_channels = in_channels // 2 if in_channels > 3 else 3
            layers.append(nn.ConvTranspose2d(in_channels, out_channels,
                                             kernel_size=self.kernel_size,
                                             stride=2))
            layers.append(get_activation(act_name))
            in_channels = out_channels

        # 若最后一层通道 != 3，就再补一层让其=3
        if in_channels != 3:
            layers.append(nn.ConvTranspose2d(in_channels, 3,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1))
            layers.append(nn.Sigmoid())
        else:
            # 如果已经3通道，就加个Sigmoid
            layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B,3,H,W)
        B, C, H, W = x.shape
        # 1) encoder => (B, latent_channels)
        latent = self.encoder(x)
        # 2) reshape => (B, latent_channels, 1, 1)
        latent = latent.view(B, -1, 1, 1)
        # 3) decoder => (B, 3, H', W'), H'W'由反卷积层决定
        out_small = self.decoder(latent)
        # 4) 若需严格回到(H,W), 用插值
        out = F.interpolate(out_small, size=(H, W), mode="bilinear",
                            align_corners=False)
        return out

def build_autoencoder(model_name="efficientnetv2_s",
                      pretrained=False,
                      latent_channels=1280,
                      pretrained_path=None,
                      decoder_config=None):
    return EfficientNetAutoencoder(
        model_name=model_name,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        latent_channels=latent_channels,
        decoder_config=decoder_config
    )
