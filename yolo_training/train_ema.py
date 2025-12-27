"""
YOLOv8-EMA 优化模型训练脚本
基于原最佳模型(mAP=84.2%)进行微调
添加EMA注意力机制和P2小目标检测层
"""
import sys
import os

# 添加模型路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f, SPPF, Concat, Detect
import torch
import torch.nn as nn

# 注册EMA模块到ultralytics
from models.ema_attention import EMA, SimAM, CoordAtt

# 注册自定义模块
import ultralytics.nn.modules as modules
modules.EMA = EMA
modules.SimAM = SimAM
modules.CoordAtt = CoordAtt

# 同时注册到tasks
from ultralytics.nn import tasks
tasks.EMA = EMA
tasks.SimAM = SimAM
tasks.CoordAtt = CoordAtt


def train_ema_model():
    """训练EMA优化模型"""
    
    # 数据集配置
    data_yaml = "cableinspect_yolo/dataset.yaml"
    
    # 方案1: 基于预训练YOLOv8n + EMA架构从头训练
    print("=" * 60)
    print("方案1: YOLOv8n-EMA-P2 全新训练")
    print("=" * 60)
    
    try:
        # 尝试加载自定义EMA模型
        model = YOLO("models/yolov8n-ema-p2.yaml")
        
        # 训练配置 - 优化参数
        results = model.train(
            data=data_yaml,
            epochs=100,
            imgsz=640,
            batch=8,  # RTX 3050 4GB显存
            device=0,
            workers=0,
            project="../runs/train",
            name="cable_ema_p2",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=5,
            warmup_momentum=0.8,
            cos_lr=True,  # 余弦退火学习率
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,  # 旋转增强
            translate=0.1,
            scale=0.5,
            shear=2.0,
            flipud=0.5,  # 上下翻转
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,  # MixUp增强
            copy_paste=0.1,  # Copy-Paste增强
            # 其他
            patience=50,
            save_period=10,
            val=True,
            plots=True,
            verbose=True,
        )
        
        print("\n训练完成!")
        print(f"最佳模型保存在: runs/train/cable_ema_p2/weights/best.pt")
        return results
        
    except Exception as e:
        print(f"EMA模型训练失败: {e}")
        print("\n尝试方案2: 基于原模型微调...")
        return train_finetune()


def train_finetune():
    """方案2: 基于原最佳模型微调"""
    print("=" * 60)
    print("方案2: 基于原模型(mAP=84.2%)继续微调")
    print("=" * 60)
    
    # 加载原最佳模型
    model = YOLO("runs/cableinspect_real/weights/best.pt")
    
    # 微调训练 - 使用更小学习率
    results = model.train(
        data="cableinspect_yolo/dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="../runs/train",
        name="cable_finetune",
        exist_ok=True,
        optimizer="AdamW",
        lr0=0.0001,  # 更小学习率用于微调
        lrf=0.001,
        cos_lr=True,
        # 增强数据增强
        degrees=15.0,
        translate=0.15,
        scale=0.6,
        shear=3.0,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        patience=30,
        save_period=5,
    )
    
    return results


if __name__ == "__main__":
    print("YOLOv8-EMA 电缆缺陷检测优化训练")
    print("=" * 60)
    print("原模型已备份至: backups/backup_20251226_best_mAP84.2/")
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: 未检测到GPU，将使用CPU训练")
    
    print()
    train_ema_model()
