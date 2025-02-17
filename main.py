import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from timm import create_model
from models.trainer import train_model
from inference.evaluator import evaluate_model
from inference.export import export_model
from datasets.mvtec import load_mvtec_dataset

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
    data_dir = "./mvtec_ad"
    train_loader, test_loader = load_mvtec_dataset(data_dir)
    
    model = create_model("efficientnetv2_s", pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    evaluate_model(model, test_loader, device)
    export_model(model, device)