# CableVision AI - YOLOv8 电缆缺陷检测

## 模型说明

本项目基于 **YOLOv8n** 进行电缆缺陷检测，当前最佳模型 mAP50 = 84.2%。

## 数据集说明

本项目使用 **CableInspect-AD** 数据集，已转换为 YOLO 格式。

**原始数据集来源**: [CableInspect-AD (Mila)](https://mila-iqia.github.io/cableinspect-ad/)

**数据集准备**:
1. 从官方下载原始 CableInspect-AD 数据集
2. 使用转换脚本将其转换为 YOLO 格式
3. 放置到 `yolo_training/cableinspect_yolo/` 目录

**YOLO格式结构**:
```
cableinspect_yolo/
├── dataset.yaml          # 数据集配置
├── train/
│   ├── images/           # 训练图片
│   └── labels/           # YOLO格式标注 (txt)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

> 注意: 数据集约 24GB，未包含在 Git 仓库中。

## 项目结构

```
yolo_training/
├── cableinspect_yolo/    # 数据集目录 (需自行准备)
├── train.py              # YOLOv8 训练脚本
├── detect.py             # 推理脚本
├── models/               # 模型配置
├── requirements.txt      # 依赖包
└── README.md
```

## 快速开始

### 1. 安装依赖
```bash
pip install ultralytics
```

### 2. 训练模型
```bash
yolo train model=yolov8n.pt data=cableinspect_yolo/dataset.yaml epochs=100 imgsz=640
```

### 3. 验证模型
```bash
yolo val model=runs/detect/train/weights/best.pt data=cableinspect_yolo/dataset.yaml
```

### 4. 推理检测
```bash
yolo predict model=runs/detect/train/weights/best.pt source=test_image.jpg
```

## 模型导出
```bash
yolo export model=best.pt format=onnx
```
