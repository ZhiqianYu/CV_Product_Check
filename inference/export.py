# =========================
# 5️⃣ 模型导出模块（inference/export.py）
# =========================
import torch

def export_model(model, device, file_name="efficientnetv2_mvtec.onnx"):
    model.to(device)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(model, dummy_input, file_name, export_params=True, opset_version=11, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])
    print(f"ONNX Model saved as {file_name}")