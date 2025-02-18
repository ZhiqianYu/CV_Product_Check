import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientNetAutoencoder(nn.Module):
    def __init__(self, model_name="efficientnetv2_s"):
        super(EfficientNetAutoencoder, self).__init__()
        from timm import create_model
        self.encoder = create_model(model_name, pretrained=True, num_classes=0)  # 去掉分类层
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1280, 512, kernel_size=3, stride=2, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, output_padding=1),  
            nn.Sigmoid()  # 归一化到 0-1 之间
        )

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), 1280, 1, 1)  # 变成 4D
        output = self.decoder(features)
        return output

def get_autoencoder(model_name="efficientnetv2_s"):
    return EfficientNetAutoencoder(model_name)
