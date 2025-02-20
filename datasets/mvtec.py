"""
文件: datasets/mvtec.py
功能: 加载MVTec或类似结构的数据集. 
     返回 train_loader, test_loader, 以及类别数num_classes
被引用: train_pipeline.py -> load_screw_dataset(...)
"""

import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def load_screw_dataset(dataset_path, image_size, batch_size,
                       augment_config=None, normalize_config=None):
    """
    根据给定路径和参数, 构造训练/测试DataLoader
    你可以在这里过滤只加载 'good' 或者全类别
    """
    if augment_config is None:
        augment_config = {"horizontal_flip": True, "rotation": 15}
    if normalize_config is None:
        normalize_config = {"mean":[0.5]*3, "std":[0.5]*3}

    # 这里根据 augment_config 动态添加数据增强操作
    transform_list = [
        transforms.Resize(image_size)
    ]
    # 简化写法, 你可按需加 RandomHorizontalFlip / RandomRotation 等
    # if augment_config["horizontal_flip"]:
    #     transform_list.append(transforms.RandomHorizontalFlip())
    # if augment_config["rotation"] > 0:
    #     transform_list.append(transforms.RandomRotation(augment_config["rotation"]))

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalize_config["mean"], normalize_config["std"])
    ])
    data_transform = transforms.Compose(transform_list)

    # 构造ImageFolder
    train_dir = os.path.join(dataset_path, "train")
    test_dir  = os.path.join(dataset_path, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=data_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=8)

    num_classes = len(train_dataset.classes)
    print(f"[Dataset] 训练集类别: {train_dataset.classes}")
    print(f"[Dataset] 测试集类别: {test_dataset.classes}")
    return train_loader, test_loader, num_classes
