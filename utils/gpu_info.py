import subprocess

def get_gpu_info():
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

        return f"AMD GPU: 温度: {gpu_info['温度']} C | Freq: {gpu_info['频率']} | Fans: {gpu_info['风扇']} | Power: {gpu_info['功耗']} W | Mem: {gpu_info['显存占用']} %"
    
    except Exception as e:
        return f"无法获取 AMD GPU 信息: {e}"