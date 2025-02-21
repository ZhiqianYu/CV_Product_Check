import os
import io
import cv2
import numpy as np
from PIL import Image
from rembg import remove

def generate_masks(
    dataset_path: str,
    mask_path: str,
    blur_ksize: int = 5,
    thresh_method: str = "binary",
    thresh_val: int = 127,
    min_contour_area: int = 100
):

    os.makedirs(mask_path, exist_ok=True)
    # 先获取绝对路径，便于比较
    abs_dataset_path = os.path.abspath(dataset_path)
    abs_mask_path = os.path.abspath(mask_path)

    for root, dirs, files in os.walk(abs_dataset_path):
        # 跳过已生成的 mask_path 及其子目录，防止递归进入
        abs_root = os.path.abspath(root)
        if abs_root.startswith(abs_mask_path):
            # 如果当前root就是(或在)mask目录里，则跳过
            continue

        for file_name in files:
            # 仅处理图像文件
            if not(file_name.lower().endswith(".jpg")
                   or file_name.lower().endswith(".png")
                   or file_name.lower().endswith(".jpeg")):
                continue

            # 原图完整路径
            input_path = os.path.join(root, file_name)
            if not os.path.exists(input_path):
                continue

            # 根据 dataset_path 计算相对路径
            relative_path = os.path.relpath(input_path, start=dataset_path)

            # 拼出掩码输出路径
            output_path = os.path.join(mask_path, relative_path)
            # 确保输出子文件夹存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # ===== 以下为具体的阈值化与形态学处理流程 =====
            input_img = Image.open(input_path)
            output = remove(input_img)
            
            # 将透明部分填充为白色
            output = output.convert("RGBA")
            background = Image.new("RGBA", output.size, (255, 255, 255, 255))
            output = Image.alpha_composite(background, output).convert("RGB")
            
            output.save(output_path)
    print(f"[generate_masks] 处理完成，掩码输出至: {mask_path}")

def remove_bg(image: Image.Image) -> Image.Image:
    # 将输入图像转换为 PNG 格式的字节流，以保留透明通道信息
    with io.BytesIO() as input_buffer:
        image.save(input_buffer, format="PNG")
        input_bytes = input_buffer.getvalue()
    output_bytes = remove(input_bytes)
    output_img = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
    background = Image.new("RGBA", output_img.size, (255, 255, 255, 255))
    output_img = Image.alpha_composite(background, output_img).convert("RGB")
    return output_img
