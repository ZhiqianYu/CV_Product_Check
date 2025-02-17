# =========================
# 2️⃣ 数据集模块（datasets/mvtec.py）
# =========================

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

def load_mvtec_dataset(data_dir, batch_size=32):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=data_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, test_loader