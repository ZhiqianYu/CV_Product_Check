import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from datasets.aug import build_transforms

def load_screw_dataset(dataset_path, image_size, batch_size,
                       augment_config=None, normalize_config=None):
    if augment_config is None:
        augment_config = {"horizontal_flip": True, "rotation": 15}  # default

    if normalize_config is None:
        normalize_config = {"mean":[0.5]*3, "std":[0.5]*3}

    transform_list = [transforms.Resize(image_size)]

    if "rotation" in augment_config and augment_config["rotation"] > 0:
        rot_deg = augment_config["rotation"]
        transform_list.append(transforms.RandomRotation(rot_deg))

    if "horizontal_flip" in augment_config and augment_config["horizontal_flip"]:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if "color_jitter" in augment_config:
        cj = augment_config["color_jitter"]
        # assume subfields: brightness, contrast, saturation, hue
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0)
            )
        )

    # Then convert to Tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(normalize_config["mean"], normalize_config["std"])
    ])

    # 构造 transforms（训练集和测试集可根据需求分开）
    train_transform = build_transforms(image_size,
                                       augment_config,
                                       normalize_config)
    test_transform  = build_transforms(image_size,
                                       None,  # 测试集一般不做或做最少的增强
                                       normalize_config)

    # 构造ImageFolder
    train_dir = os.path.join(dataset_path, "train")
    test_dir  = os.path.join(dataset_path, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=8)

    num_classes = len(train_dataset.classes)
    print(f"[Dataset] 训练集类别: {train_dataset.classes}")
    print(f"[Dataset] 测试集类别: {test_dataset.classes}")
    return train_loader, test_loader, num_classes
