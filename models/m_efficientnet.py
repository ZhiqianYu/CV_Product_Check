import torch
import torch.nn as nn
import os
from timm import create_model

class EfficientNetModel(nn.Module):
    def __init__(self, model_name="efficientnetv2_s", num_classes=6, pretrained=False):
        super(EfficientNetModel, self).__init__()
        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

def get_efficientnet(model_name, num_classes, pretrained=False):
    """
    创建 EfficientNet 模型
    """
    return EfficientNetModel(model_name, num_classes, pretrained)

def save_model(model, save_path):
    """
    保存模型到 .pth 文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到: {save_path}")

def load_model(model_name, num_classes, model_path, device):
    """
    加载训练好的 EfficientNet 模型
    """
    model = get_efficientnet(model_name, num_classes, pretrained=False).to(device)
    print(model)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 已加载模型: {model_path}")
    return model

def export_onnx(model, save_path, image_size):
    """
    将 PyTorch 模型转换为 ONNX
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    dummy_input = torch.randn(1, 3, image_size, image_size).to(next(model.parameters()).device)
    torch.onnx.export(
        model, dummy_input, save_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"✅ ONNX 模型已导出: {save_path}")
