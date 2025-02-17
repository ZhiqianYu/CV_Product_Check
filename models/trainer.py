import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import streamlit as st
import subprocess

def get_amd_gpu_usage():
    """
    获取 AMD GPU 使用率和显存占用
    """
    try:
        # 运行 `rocm-smi` 获取 GPU 负载和显存
        result = subprocess.run(["rocm-smi", "-d", "0" ,"-P", "-t", "-f", "-g", "--showmemuse"], capture_output=True, text=True)
        output = result.stdout.split("\n")

        # 解析 GPU 使用率和显存
        gpu_info = {
            "温度": "N/A",
            "频率": "N/A",
            "风扇": "N/A",
            "功耗": "N/A",
            "显存占用": "N/A",
            "显存活动": "N/A"
        }

        for line in output:
            if "Temperature (Sensor junction)" in line:
                gpu_info["温度"] = line.split(":")[-1].strip()
            elif "sclk clock level" in line:
                gpu_info["频率"] = line.split("(")[-1].split(")")[0].strip()
            elif "fan speed" in line:
                gpu_info["风扇"] = line.split(":")[-1].strip() 
            elif "Average Graphics Package Power" in line:
                gpu_info["功耗"] = line.split(":")[-1].strip()
            elif "GPU Memory Allocated (VRAM%)" in line:
                gpu_info["显存占用"] = line.split(":")[-1].strip()
            elif "GPU Memory Read/Write Activity" in line:
                gpu_info["显存活动"] = line.split(":")[-1].strip()

        return f"AMD GPU: 温度 {gpu_info['温度']} C | {gpu_info['频率']} | {gpu_info['风扇']} | 功耗 {gpu_info['功耗']} W | 显存占用 {gpu_info['显存占用']} % | 显存活动 {gpu_info['显存活动']} %"
    
    except Exception as e:
        return f"无法获取 AMD GPU 信息: {e}"

def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    """
    训练 EfficientNet 模型，并在 Streamlit 界面显示实时日志 & GPU 监控
    """
    model.to(device)
    best_acc = 0.0
    best_model_wts = model.state_dict()

    # Streamlit UI
    st.subheader("训练过程")
    progress_bar = st.progress(0)  # 进度条
    epoch_text = st.empty()  # 训练信息
    gpu_text = st.empty()  # GPU 使用率

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            print(images.shape, labels)
            images, labels = images.to(device), labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)
            print(outputs)
            loss = criterion(outputs, labels)

            # 确保 loss 是正常数值
            if loss.item() == 0 or torch.isnan(loss):
                print("❌ Loss 出现问题，跳过此 batch")
                break
                #continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        elapsed_time = time.time() - start_time

        # 监控 GPU 使用情况
        gpu_info = get_amd_gpu_usage()

        # 更新 Streamlit UI
        progress_bar.progress((epoch + 1) / num_epochs)
        epoch_text.text(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Time: {elapsed_time:.2f}s")
        gpu_text.text(gpu_info)

        # 测试模型
        test_acc = evaluate_model(model, test_loader, device)
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()  # 更新最佳模型

    print(f"✅ 训练完成！最佳测试精度: {best_acc:.4f}")

    # 保存最佳模型
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "models/files/best_model.pth")
    print(f"✅ 最佳模型已保存至 models/files/best_model.pth")

    return model

def evaluate_model(model, test_loader, device):
    """
    评估模型
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"✅ 测试集准确率: {accuracy:.4f}")
    return accuracy
