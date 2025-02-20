import torchvision.transforms as transforms

def build_transforms(image_size,
                     augment_config=None,
                     normalize_config=None):
    """
    根据给定的 augment_config 动态构造图像变换流水线并返回.
    参数:
        image_size: (H, W) 或者 int, 用于统一图像大小
        augment_config: dict, 包含是否使用旋转、颜色变换等标记和范围值
            示例:{
               "rotation": 30,          # 旋转范围
               "color_jitter": True,
               "brightness": 0.2,
               "contrast": 0.2,
               "saturation": 0.2,
               "hue": 0.1,
               "horizontal_flip": True
            }
        normalize_config: dict, 包含 mean, std 例如 {"mean":[0.5,0.5,0.5], "std":[0.5,0.5,0.5]}

    返回:
        transforms.Compose([...])  # 用于 DataLoader 中
    """
    if augment_config is None:
        augment_config = {}
    if normalize_config is None:
        normalize_config = {"mean": [0.5]*3, "std": [0.5]*3}

    transform_list = []

    # 1) 调整图像尺寸
    transform_list.append(transforms.Resize(image_size))

    # 2) 根据 augment_config 决定是否进行旋转、翻转、颜色等增强
    if "rotation" in augment_config and augment_config["rotation"] > 0:
        # 随机旋转
        transform_list.append(transforms.RandomRotation(degrees=augment_config["rotation"]))

    if augment_config.get("horizontal_flip", False):
        transform_list.append(transforms.RandomHorizontalFlip())

    if augment_config.get("color_jitter", False):
        # 随机颜色抖动
        brightness = augment_config.get("brightness", 0.3)
        contrast   = augment_config.get("contrast", 0.3)
        saturation = augment_config.get("saturation", 0.3)
        hue        = augment_config.get("hue", 0.2)
        transform_list.append(
            transforms.ColorJitter(brightness=brightness,
                                   contrast=contrast,
                                   saturation=saturation,
                                   hue=hue)
        )

    # 3) 转为 Tensor 并归一化
    transform_list.append(transforms.ToTensor())
    transform_list.append(
        transforms.Normalize(normalize_config["mean"],
                             normalize_config["std"])
    )

    return transforms.Compose(transform_list)