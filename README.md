# CV_Product_Check
Use AI based CV method to check product quality during the producing process.

industrial_defect_detection/
│── datasets/           # 数据集处理（加载、预处理）
│   ├── mvtec.py       # MVTec AD 数据处理
│   ├── dagm.py        # DAGM 2007 数据处理
│   ├── augment.py     # 数据增强
│── models/            # 训练 & 评估
│   ├── efficientnet.py # EfficientNet 训练
│   ├── yolo.py        # YOLO 目标检测
│   ├── trainer.py     # 训练框架（支持不同模型）
│── inference/         # 推理（ONNX / C++）
│   ├── infer.py       # Python 推理
│   ├── onnx_infer.py  # ONNX 推理
│   ├── cpp_infer.cpp  # C++ 端推理（可选）
│── visualization/     # 结果可视化
│   ├── streamlit_app.py # Streamlit 界面
│   ├── plot_metrics.py  # 绘制损失 & 精度曲线
│── configs/           # 训练 & 超参数配置
│   ├── config.yaml    # 训练超参数
│── utils/             # 工具函数
│   ├── logger.py      # 日志管理
│   ├── metrics.py     # 计算评估指标
│── main.py            # 入口（训练 & 推理）
│── requirements.txt   # 依赖库
│── README.md          # 项目说明
