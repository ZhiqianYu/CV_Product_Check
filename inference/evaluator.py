# =========================
# 4️⃣ 模型评估模块（models/evaluator.py）
# =========================
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
import streamlit as st
from visualization.app import num_images

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    class_names = test_loader.dataset.classes # 获取类别名称
    images, labels = next(iter(test_loader))
    images, labels = images[:num_images].to(device), labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    st.subheader("测试图片预测结果")
    for i in range(num_images):
        img = transforms.ToPILImage()(images[i].cpu())
        st.image(img, caption=f"真实类别: {class_names[labels[i]]} | 预测类别: {class_names[preds[i]]}", use_column_width=True)
    
    st.subheader("分类报告")
    all_preds, all_labels = [], []

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))  # 使用真实类别
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))