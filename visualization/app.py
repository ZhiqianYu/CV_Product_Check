"""
文件: app.py
功能: 提供 Streamlit 界面, 在侧栏可调训练参数, 点击按钮执行训练,
     并可上传图片进行推理, 用Autoencoder重建误差来判定是否缺陷。
被引用: 直接通过: streamlit run app.py
"""

import os
import yaml
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, config_path="configs/config.yaml"):
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

def run_visualization():
    st.title("工业缺陷检测系统 - 自编码器")

    # 1) 读取原有config
    config_path = "configs/config.yaml"
    config = load_config(config_path)

    st.sidebar.header("1. 训练参数设置")
    # 2) 在侧栏中让用户调参
    dataset_path = st.sidebar.text_input("数据集路径", value=config["data"]["dataset_path"])
    image_size   = st.sidebar.slider("图像大小", 64, 2048, config["data"]["image_size"], step=64)
    batch_size   = st.sidebar.slider("批次大小", 1, 64, config["training"]["batch_size"])
    learning_rate= st.sidebar.slider("学习率", 0.0001, 0.1, config["training"]["learning_rate"], step=0.0001)
    num_epochs   = st.sidebar.slider("训练轮数", 1, 200, config["training"]["num_epochs"])
    loss_func    = st.sidebar.selectbox("损失函数", ["MSE", "L1", "SSIM", "Mixed"], index=0)

    model_name   = st.sidebar.selectbox("EfficientNet版本", ["efficientnetv2_s", "efficientnet_b3"], index=0)
    pretrained   = st.sidebar.checkbox("使用预训练", value=False)

    # Decoder可调
    decoder_layers    = st.sidebar.slider("Decoder层数", 1, 5, config["autoencoder"]["decoder"]["num_layers"])
    decoder_kernelsz  = st.sidebar.selectbox("卷积核大小", [3, 4, 5], index=1)  # 默认4
    decoder_activation= st.sidebar.selectbox("激活函数", ["ReLU", "LeakyReLU", "ELU"], index=0)
    use_skip          = st.sidebar.checkbox("使用skip-connection", value=False)

    # 3) 当点击 "开始训练" 时
    if st.sidebar.button("开始训练"):
        # 更新配置
        config["data"]["dataset_path"] = dataset_path
        config["data"]["image_size"] = image_size
        config["training"]["batch_size"] = batch_size
        config["training"]["learning_rate"] = learning_rate
        config["training"]["num_epochs"] = num_epochs
        config["training"]["loss_function"] = loss_func
        config["model"]["name"] = model_name
        config["model"]["pretrained"] = pretrained

        config["autoencoder"]["decoder"]["num_layers"] = decoder_layers
        config["autoencoder"]["decoder"]["kernel_size"] = decoder_kernelsz
        config["autoencoder"]["decoder"]["activation"] = decoder_activation
        config["autoencoder"]["decoder"]["use_skip_connection"] = use_skip

        # 保存新的config
        save_config(config, config_path)

        # 调用训练
        from utils.train_pipeline import train_autoencoder
        train_autoencoder(config_path)
        st.success("训练完成, 模型已保存!")

    # 4) 推理部分
    st.subheader("2. 推理: 上传图片进行缺陷检测")
    threshold = st.sidebar.slider("缺陷阈值(越小越敏感)", 0.0, 0.1, 0.02, step=0.001)
    uploaded_file = st.file_uploader("上传图片", type=["jpg","png","jpeg"])

    if uploaded_file:
        # 加载最新config
        config = load_config(config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 Autoencoder
        from models.m_autoencoder import build_autoencoder
        auto_model = build_autoencoder(
            model_name=config["model"]["name"],
            pretrained=config["model"]["pretrained"],
            latent_channels=config["autoencoder"]["latent_channels"],
            decoder_config=config["autoencoder"]["decoder"]
        ).to(device)

        # 加载最优权重
        best_model_path = os.path.join(config["model"]["save_path"], "autoencoder_best.pth")
        if not os.path.exists(best_model_path):
            st.error("未找到 autoencoder_best.pth, 请先训练!")
            return
        auto_model.load_state_dict(torch.load(best_model_path, map_location=device))
        auto_model.eval()

        # 图像预处理
        image = Image.open(uploaded_file).convert("RGB")
        tfms = transforms.Compose([
            transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        img_tensor = tfms(image).unsqueeze(0).to(device)

        # 计算重建误差
        with torch.no_grad():
            recon = auto_model(img_tensor)
        mse = torch.mean((img_tensor - recon)**2).item()

        st.image(uploaded_file, caption=f"上传图片 (原尺寸: {image.size})", use_container_width=True)
        st.write(f"重建误差: {mse:.6f}")

        if mse > threshold:
            st.error("检测到缺陷 ❌")
        else:
            st.success("物品完好 ✅")

if __name__ == "__main__":
    run_visualization()
