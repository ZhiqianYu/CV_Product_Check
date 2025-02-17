import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

def load_screw_dataset(dataset_path, image_size, batch_size, augment_config, normalize_config):
    data_transform = transforms.Compose([
        transforms.Resize(tuple(image_size)),
        #transforms.RandomHorizontalFlip() if augment_config['horizontal_flip'] else transforms.Lambda(lambda x: x),
        #transforms.RandomRotation(augment_config['rotation']) if augment_config['rotation'] > 0 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(normalize_config['mean'], normalize_config['std'])
    ])

    # 训练集：只包含 `good`
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=data_transform)

    # 测试集：包含 `good` 和所有缺陷类别
    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=data_transform)

    # 打印类别信息
    print(f"✅ 训练集类别: {train_dataset.classes}")
    print(f"✅ 类别映射: {train_dataset.class_to_idx}")  # {'good': 0, 'bent': 1, 'scratch': 2, 'other_defect': 3}
    print(f"✅ 测试集类别: {test_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    return train_loader, test_loader, num_classes