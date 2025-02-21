import os
import yaml
import torch
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from utils.diff_map import compute_diff_map
from utils.gpu_info import get_gpu_info
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from datasets.mask_unet import generate_masks
from datasets.mask_unet import remove_bg

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def save_config(config, config_path="configs/config.yaml"):
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

def run_visualization():
    st.title("缺陷检测系统 - 自编码器")

    # 读取原有config
    config_path = "configs/config.yaml"
    config_last_path = "configs/config_last.yaml"
    config = load_config(config_path)

    st.sidebar.header("训练参数设置")
    # 在侧栏中让用户调参
    dataset_path = st.sidebar.text_input("数据集路径", value=config["data"]["dataset_path"])
    image_size   = st.sidebar.slider("图像大小", 64, 1024, config["data"]["image_size"], step=64)
    batch_size   = st.sidebar.slider("批次大小", 8, 64, config["training"]["batch_size"])
    learning_rate= st.sidebar.slider("学习率", 0.0001, 0.1, config["training"]["learning_rate"], step=0.0001, format="%.4f")
    num_epochs   = st.sidebar.slider("训练轮数", 1, 500, config["training"]["num_epochs"])
    loss_func    = st.sidebar.selectbox("损失函数", ["MSE", "L1", "SSIM", "MIXED"], index=0)
    model_name   = st.sidebar.selectbox("Train Model", ["efficientnetv2_s", "efficientnet_b3"], index=0)
    pretrained   = st.sidebar.checkbox("使用预训练", value=False)

    # Decoder可调
    st.sidebar.header("Decoder设置")
    decoder_layers    = st.sidebar.slider("Decoder层数", 1, 5, config["autoencoder"]["decoder"]["num_layers"])
    decoder_kernelsz  = st.sidebar.selectbox("卷积核大小", [3, 4, 5], index=1)  # 默认4
    decoder_activation= st.sidebar.selectbox("激活函数", ["ReLU", "LeakyReLU", "ELU"], index=0)
    use_skip          = st.sidebar.checkbox("使用skip-connection", value=False)

    # Mask分割可调
    st.sidebar.header("Mask分割设置")
    blur_ksize = st.sidebar.slider("模糊核大小", 1, 15, config["mask"]["blur_ksize"])
    thresh_method = st.sidebar.selectbox("二值化方法", ["binary", "binary_inv"], index=0)
    thresh_val = st.sidebar.slider("阈值", 0, 255, config["mask"]["thresh_val"])
    min_contour_area = st.sidebar.slider("最小轮廓面积", 1, 1000, config["mask"]["min_contour_area"])

    use_masked_data = st.sidebar.checkbox("使用分割后的数据？", value=False)
    columns = st.sidebar.columns(2)
    with columns[0]:
        segment_button = st.button("分割数据")
    with columns[1]:
        start_train_button = st.button("开始训练")
    

    # If user clicks 分割数据
    if segment_button:
        # Update the config fields
        mask_path = os.path.join(dataset_path, "seg_mask")
        config["mask"]["mask_path"] = mask_path
        config["mask"]["blur_ksize"] = blur_ksize
        config["mask"]["thresh_method"] = thresh_method
        config["mask"]["thresh_val"] = thresh_val
        config["mask"]["min_contour_area"] = min_contour_area
        config["data"]["dataset_path"] = dataset_path  # if needed

        save_config(config, config_last_path)
        generate_masks(
            dataset_path=dataset_path,
            mask_path=mask_path,
            blur_ksize=blur_ksize,
            thresh_method=thresh_method,
            thresh_val=thresh_val,
            min_contour_area=min_contour_area
        )

        st.success("分割完成！已生成mask文件")

    # 当点击 "开始训练" 时
    if start_train_button:
        # 更新配置
        config["data"]["dataset_path"] = dataset_path
        config["data"]["image_size"] = image_size
        config["data"]["use_masked"]   = use_masked_data
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
        save_config(config, config_last_path)

        # -- 在页面上添加一个提示, 以及进度条和文字占位
        st.write("正在训练，请稍候...")
        progress_bar = st.progress(0)
        status_text_line1 = st.empty()
        status_text_line2 = st.empty()
        chart_placeholder = st.empty()

        # We define columns once, up front for Augmentation samples:
        st.subheader("On-the-fly Augmentation Samples (Every 3 Batches)")
        columns = st.columns(4)
        # Make placeholders for each column
        place_holders = [col.empty() for col in columns]

        # 所以先读出配置中的 num_epochs
        total_epochs = config["training"]["num_epochs"]
        # 创建一个数组来保存 [None, None, None, ...] 大小为 total_epochs
        loss_array = [None] * total_epochs

        # 若 session_state 中不存在 loss_history，就初始化一个空list
        if "loss_history" not in st.session_state:
            st.session_state.loss_history = []

        # 定义回调函数, 用于更新页面
        def st_callback(epoch, total_epochs, avg_loss, elapsed, remain, get_gpu_info, aug_images=None):
            """
            epoch: 当前轮数(1-based)
            total_epochs: 总轮数
            avg_loss: 当前epoch平均loss
            elapsed: 已用时间(s)
            remain: 预计剩余时间(s)
            """
            # 计算进度0~1之间
            current_progress = epoch / total_epochs
            progress_bar.progress(current_progress)

            # 显示信息: 如 Epoch X, 已用时间, 预计剩余时间, 当前loss
            status_text_line1.write(
                f"**Epoch {epoch}/{total_epochs}** | "
                f"Loss={avg_loss:.6f} | "
                f"已用时: {elapsed:.1f}s | "
                f"预计剩余: {remain:.1f}s"
            )
            status_text_line2.write(f"{get_gpu_info}")

            # If we received a list of images to show
            if aug_images is not None:
                for i, ph in enumerate(place_holders):
                    # Update the same placeholder each time
                    ph.image(aug_images[i], caption=f"Aug Image {i+1}", use_container_width=True)

            # 更新 loss_history
            loss_array[epoch - 1] = avg_loss

            # 使用 Matplotlib 绘图, 固定X轴为1~total_epochs
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.set_title("RT Training Loss Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_xlim([1, total_epochs])
            ax.set_ylim([0, 0.4]) 
            ax.grid(True)
            ax.grid(which='major', linestyle='--', linewidth='0.1', color='gray')
            ax.grid(which='minor', linestyle=':', linewidth='0.1', color='gray')

            x_vals = np.arange(1, total_epochs + 1)
            y_vals = [v if v is not None else np.nan for v in loss_array] # 注意: 有些值还没训练到, 会是 None, 可以先替换成 np.nan

            ax.plot(
                x_vals, 
                y_vals,
                marker='.',         # 也可改成 '.', ',' 等更小的点
                color='blue',
                linewidth=1,        # 线条更细
                markersize=3        # 点更小
            )
            chart_placeholder.pyplot(fig)
            plt.close(fig)

        # 调用训练
        from utils.train_pipeline import train_autoencoder
        train_autoencoder(config_last_path, progress_callback=st_callback)
        st.success("训练完成, 模型已保存!")

    # 4) 推理部分
    st.subheader("推理: 上传图片进行缺陷检测")

    uploaded_file = st.file_uploader("请选择要上传的图片", type=["jpg", "png", "jpeg"])
    threshold = st.sidebar.slider("缺陷阈值(越小越敏感)", 0.0, 0.2, 0.001, step=0.001, format="%.3f")

    if uploaded_file:
        # 加载最新config
        config = load_config(config_last_path)
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
        # remove background
        loaded_img = Image.open(uploaded_file).convert("RGB")
        image = remove_bg(loaded_img)
        tfms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        img_tensor = tfms(image).unsqueeze(0).to(device)

        # 计算重建误差
        with torch.no_grad():
            recon = auto_model(img_tensor)
        mse = torch.mean((img_tensor - recon)**2).item()

        # 计算差异图
        diff_img = compute_diff_map(img_tensor, recon)       

        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((512, 512)), caption="原图", use_container_width=True)
        with col2:
            st.image(diff_img, caption="差异图", use_container_width=True)

        st.write(f"重建误差: {mse:.6f}")
        if mse > threshold:
            st.error("检测到缺陷 ❌")
        else:
            st.success("物品完好 ✅")

if __name__ == "__main__":
    run_visualization()
