import os
import torch
import onnxruntime as ort
import torchvision.transforms as transforms
from PIL import Image
import yaml

# 读取配置
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_inference(image_file, model_name, image_size):
    # 载入模型路径
    model_path = os.path.join(config["model"]["save_path"], model_name + ".onnx")
    
    if not os.path.exists(model_path):
        return "模型未找到，请先训练模型！"

    # 加载 ONNX 模型
    ort_session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider" if get_device() == "cuda" else "CPUExecutionProvider"])

    # 预处理图片
    image = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config["data"]["normalize"]["mean"], std=config["data"]["normalize"]["std"])
    ])
    image_tensor = transform(image).unsqueeze(0).numpy()

    # 运行推理
    outputs = ort_session.run(None, {"input": image_tensor})
    pred = torch.tensor(outputs[0]).argmax(dim=1).item()

    # 适配 `screw` 类别
    class_names = ["good"] + ["bent", "scratch", "other_defect"]
    return class_names[pred]
