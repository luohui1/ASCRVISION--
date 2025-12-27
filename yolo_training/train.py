"""
CableVision AI - YOLO电缆缺陷检测模型训练脚本
"""
import os
import argparse
from pathlib import Path
from ultralytics import YOLO


def train(args):
    """训练YOLO模型"""
    # 加载模型
    if args.resume:
        model = YOLO(args.resume)
        print(f"从检查点恢复训练: {args.resume}")
    else:
        model = YOLO(args.model)
        print(f"加载预训练模型: {args.model}")
    
    # 训练参数
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
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
    }
    
    # 开始训练
    results = model.train(**train_args)
    print(f"\n训练完成! 模型保存在: {args.project}/{args.name}")
    return results


def validate(args):
    """验证模型"""
    model = YOLO(args.weights)
    results = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch)
    print(f"\n验证结果:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    return results


def export_model(args):
    """导出模型"""
    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"模型已导出为 {args.format} 格式")


def main():
    parser = argparse.ArgumentParser(description='CableVision AI - YOLO训练')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--model', type=str, default='yolov8n.pt', help='预训练模型')
    train_parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    train_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    train_parser.add_argument('--device', type=str, default='0', help='设备 (0,1,2... 或 cpu)')
    train_parser.add_argument('--workers', type=int, default=8, help='数据加载线程数')
    train_parser.add_argument('--project', type=str, default='runs/train', help='项目目录')
    train_parser.add_argument('--name', type=str, default='cable_defect', help='实验名称')
    train_parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点')
    
    # 验证命令
    val_parser = subparsers.add_parser('val', help='验证模型')
    val_parser.add_argument('--weights', type=str, required=True, help='模型权重')
    val_parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置')
    val_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    val_parser.add_argument('--batch', type=int, default=16, help='批次大小')
    
    # 导出命令
    export_parser = subparsers.add_parser('export', help='导出模型')
    export_parser.add_argument('--weights', type=str, required=True, help='模型权重')
    export_parser.add_argument('--format', type=str, default='onnx', help='导出格式')
    export_parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    export_parser.add_argument('--half', action='store_true', help='FP16量化')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train(args)
    elif args.command == 'val':
        validate(args)
    elif args.command == 'export':
        export_model(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
