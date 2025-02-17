import os
import yaml
import torch
import streamlit as st

with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_visualization():
    st.title("工业缺陷检测系统 - 可视化界面")
    st.sidebar.header("Training 参数设置")

    # 配置参数选择
    dataset_path = st.sidebar.text_input("数据集路径", config['data']['dataset_path'])
    image_size = st.sidebar.slider("图像大小", 64, 1024, config['data']['image_size'][0])
    batch_size = st.sidebar.slider("批次大小", 1, 64, config['training']['batch_size'])
    learning_rate = st.sidebar.slider("学习率", 0.001, 1.0, config['training']['learning_rate'], step=0.001, format="%.3f")
    num_epochs = st.sidebar.slider("训练轮数", 1, 100, config['training']['num_epochs'])
    model_name = st.sidebar.selectbox("选择模型", ["efficientnetv2_s", "YoloV8"])

    # 训练模型（动态调用训练脚本）
    if st.sidebar.button("开始训练"):
        from utils.train_pipeline import train_and_save_config  # 只在需要时导入，避免循环依赖
        new_config = {
            "data": {
                "dataset_path": dataset_path,
                "image_size": [image_size, image_size]
            },
            "training": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            },
            "model": {
                "name": model_name,
                "save_path": config['model']['save_path']
            }
        }
        train_and_save_config(new_config)  # 训练并保存 config
        st.success("训练完成，模型与参数已保存！")

    # 选择加载已有模型或重新训练
    st.subheader("Inference")
    load_model = st.sidebar.checkbox("加载已有模型", value=False)

    if load_model:
        model_path = st.sidebar.text_input("模型路径", os.path.join(config['model']['save_path'], model_name + ".onnx"))
        if os.path.exists(model_path):
            st.success(f"已加载模型: {model_path}")
        else:
            st.error("模型路径无效，请检查！")

    # 上传图片进行推理
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        from utils.inference_pipeline import run_inference  # 只在需要时导入
        result = run_inference(uploaded_file, model_name, image_size)
        st.image(uploaded_file, caption=f"预测类别: {result}", use_column_width=True)
    
    st.subheader("Model Evaluation")