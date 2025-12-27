"""
CableVision AI - CBAM增强版YOLO训练脚本
支持自定义注意力机制模块
"""
import os
import sys
import argparse
from pathlib import Path

# 添加models目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'models'))

from ultralytics import YOLO
from models.cbam import CBAM, SE, ECA


def register_custom_modules():
    """注册自定义模块到ultralytics"""
    import ultralytics.nn.modules as modules
    import ultralytics.nn.tasks as tasks
    from models import cbam
    
    # 注册到modules
    modules.CBAM = cbam.CBAM
    modules.SE = cbam.SE
    modules.ECA = cbam.ECA
    
    # 注册到tasks的全局命名空间
    tasks.CBAM = cbam.CBAM
    tasks.SE = cbam.SE
    tasks.ECA = cbam.ECA
    
    print("✓ 自定义注意力模块已注册: CBAM, SE, ECA")


def train_cbam(args):
    """使用CBAM增强模型训练"""
    register_custom_modules()
    
    # 加载模型配置
    if args.model.endswith('.yaml'):
        model = YOLO(args.model)
        print(f"从配置文件创建模型: {args.model}")
    else:
        model = YOLO(args.model)
        print(f"加载预训练模型: {args.model}")
    
    # 训练参数 - 针对小目标优化
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        # 数据增强 - 增强小目标检测
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,      # 增加旋转
        'translate': 0.2,     # 增加平移
        'scale': 0.9,         # 增加缩放范围
        'shear': 5.0,         # 添加剪切
        'flipud': 0.5,        # 添加上下翻转
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,         # 添加mixup
        'copy_paste': 0.1,    # 添加copy-paste
    }
    
    print("\n" + "=" * 50)
    print("CBAM增强版训练配置")
    print("=" * 50)
    print(f"模型: {args.model}")
    print(f"数据集: {args.data}")
    print(f"轮数: {args.epochs}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"批次大小: {args.batch}")
    print("=" * 50 + "\n")
    
    # 开始训练
    results = model.train(**train_args)
    print(f"\n训练完成! 模型保存在: {args.project}/{args.name}")
    return results


def main():
    parser = argparse.ArgumentParser(description='CableVision AI - CBAM增强训练')
    parser.add_argument('--model', type=str, default='models/yolov8n-cbam.yaml',
                        help='模型配置文件或预训练权重')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                        help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--batch', type=int, default=8, help='批次大小')
    parser.add_argument('--device', type=str, default='0', help='设备')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程')
    parser.add_argument('--project', type=str, default='../runs/train', help='项目目录')
    parser.add_argument('--name', type=str, default='cable_cbam', help='实验名称')
    
    args = parser.parse_args()
    train_cbam(args)


if __name__ == '__main__':
    main()
