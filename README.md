# CableVision AI - 钢芯铝绞线缺陷检测系统

基于 YOLOv8 的钢芯铝绞线(ACSR)缺陷智能检测系统，适用于电力线路质量检测。

## 项目结构

```
├── demos/              # 演示脚本
│   ├── api_server.py   # API服务
│   └── realtime_web_demo.py  # 实时检测Web演示
├── hardware/           # 硬件设计
│   └── device_final.html     # 3D设备设计
├── web_app/            # Web应用
│   ├── app.py          # Flask应用
│   └── templates/      # 页面模板
└── yolo_training/      # 模型训练
    ├── train.py        # 训练脚本
    ├── detect.py       # 推理脚本
    └── models/         # 模型配置
```

## 模型性能

- 模型: YOLOv8n
- mAP50: 84.2%
- 数据集: [CableInspect-AD](https://mila-iqia.github.io/cableinspect-ad/) (已转换为YOLO格式)

## 快速开始

### 安装依赖
```bash
pip install ultralytics flask
```

### 启动Web应用
```bash
cd web_app
python app.py
```

### 训练模型
```bash
cd yolo_training
yolo train model=yolov8n.pt data=cableinspect_yolo/dataset.yaml epochs=100
```

## 功能特性

- 单图检测
- 批量检测
- 实时视频检测
- 3D设备可视化
- 检测报告生成

## License

MIT
