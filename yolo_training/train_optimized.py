"""
基于原最佳模型(mAP=84.2%)的优化训练脚本
使用更强的数据增强和优化的超参数
"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("基于原模型优化训练 (mAP=84.2% -> 目标90%+)")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载原最佳模型
    model = YOLO("runs/cableinspect_real/weights/best.pt")
    
    # 优化训练配置
    results = model.train(
        data="yolo_training/cableinspect_yolo/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project="../runs/train",
        name="cable_optimized",
        exist_ok=True,
        # 优化器配置
        optimizer="AdamW",
        lr0=0.0005,      # 微调用较小学习率
        lrf=0.01,
        cos_lr=True,     # 余弦退火
        warmup_epochs=5,
        # 增强数据增强 - 提升泛化能力
        degrees=15.0,    # 旋转
        translate=0.15,
        scale=0.6,
        shear=3.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,       # MixUp增强
        copy_paste=0.2,  # Copy-Paste增强
        # 正则化
        weight_decay=0.001,
        dropout=0.1,     # Dropout防过拟合
        # 其他
        patience=50,
        save_period=10,
        val=True,
        plots=True,
    )
    
    print("\n训练完成!")
    print(f"结果保存在: runs/train/cable_optimized/")

if __name__ == "__main__":
    main()
