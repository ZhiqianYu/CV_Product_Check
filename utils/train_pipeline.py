import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from models.m_efficientnet import get_efficientnet, save_model, export_onnx
from datasets.mvtec import load_screw_dataset
from models.trainer import train_model

def train_and_save_config(new_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 解析新配置
    dataset_path = new_config["data"]["dataset_path"]
    image_size = tuple(new_config["data"]["image_size"])
    batch_size = new_config["training"]["batch_size"]
    learning_rate = new_config["training"]["learning_rate"]
    num_epochs = new_config["training"]["num_epochs"]
    model_name = new_config["model"]["name"]
    save_path = new_config["model"]["save_path"]

    # 读取数据集
    train_loader, test_loader, num_classes = load_screw_dataset(
        dataset_path=dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        augment_config={"horizontal_flip": True, "rotation": 15},
        normalize_config={"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}
    )

    # 创建模型
    model = get_efficientnet(model_name, num_classes=num_classes, pretrained=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 训练模型
    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # 导出 PyTorch 模型
    model_save_path = os.path.join(save_path, model_name + ".pth")
    save_model(model, model_save_path)

    # 导出 ONNX 模型
    onnx_save_path = os.path.join(save_path, model_name + ".onnx")
    export_onnx(model, onnx_save_path, image_size[0])

    # 保存新的 config
    with open("configs/config.yaml", "w") as file:
        yaml.dump(new_config, file)

    print(f"✅ 训练完成！模型保存在 {model_save_path}")
    print(f"✅ ONNX 模型保存在 {onnx_save_path}")
    print(f"✅ 配置已更新并保存到 configs/config.yaml")
