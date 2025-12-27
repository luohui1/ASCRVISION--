"""
CableVision AI - 公开数据集下载与转换工具
"""
import os
import json
import shutil
import urllib.request
from pathlib import Path

# 系统10类缺陷映射
SYSTEM_CLASSES = {
    'scratch': 0, 'bubble': 1, 'crack': 2, 'dent': 3, 'discolor': 4,
    'print_miss': 5, 'eccentric': 6, 'impurity': 7, 'wrinkle': 8, 'peel': 9
}

# 公开数据集类别映射到系统类别
DATASET_MAPPINGS = {
    'mpcd': {
        'broken_strand': 'crack',
        'damage': 'scratch',
        'corrosion': 'discolor',
    },
    'cplid': {
        'defect': 'crack',
        'flashover': 'discolor',
        'damage': 'scratch',
    }
}

DATASET_URLS = {
    'mpcd': 'https://github.com/phd-benel/powerline-mtyolo',
    'roboflow_cable': 'https://universe.roboflow.com/search?q=cable+defect',
}


def download_file(url, dest):
    """下载文件"""
    print(f"下载: {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"保存到: {dest}")


def convert_labels(src_dir, dst_dir, mapping, src_format='yolo'):
    """转换标注格式到系统类别"""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for label_file in Path(src_dir).glob('*.txt'):
        new_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    old_class = parts[0]
                    if old_class in mapping:
                        new_class = SYSTEM_CLASSES.get(mapping[old_class], -1)
                        if new_class >= 0:
                            new_lines.append(f"{new_class} {' '.join(parts[1:])}\n")
        
        if new_lines:
            with open(dst_dir / label_file.name, 'w') as f:
                f.writelines(new_lines)


def setup_dataset_structure(base_dir='dataset'):
    """创建数据集目录结构"""
    base = Path(base_dir)
    for split in ['train', 'val', 'test']:
        (base / 'images' / split).mkdir(parents=True, exist_ok=True)
        (base / 'labels' / split).mkdir(parents=True, exist_ok=True)
    print(f"数据集目录已创建: {base}")


def print_dataset_info():
    """打印可用数据集信息"""
    print("\n=== 可用公开数据集 ===\n")
    print("1. MPCD (Merged Public Power Cable Dataset)")
    print("   - 1,871张图片，电缆分割和断股检测")
    print(f"   - 链接: {DATASET_URLS['mpcd']}")
    print("   - 下载: git clone https://github.com/phd-benel/powerline-mtyolo")
    print()
    print("2. Roboflow 电缆缺陷数据集")
    print(f"   - 搜索: {DATASET_URLS['roboflow_cable']}")
    print("   - 支持直接导出YOLO格式")
    print()
    print("3. 合成数据集（推荐快速测试）")
    print("   - 运行: python generate_dataset.py --train 1000 --val 200")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='下载和转换公开数据集')
    parser.add_argument('--dataset', type=str, choices=['mpcd', 'roboflow', 'info'], 
                        default='info', help='数据集名称')
    parser.add_argument('--output', type=str, default='dataset', help='输出目录')
    args = parser.parse_args()
    
    if args.dataset == 'info':
        print_dataset_info()
    else:
        setup_dataset_structure(args.output)
        print(f"\n请手动下载 {args.dataset} 数据集，然后将图片放入 {args.output}/images/")
        print(f"标注文件放入 {args.output}/labels/")
        print("\n下载链接:")
        print(f"  {DATASET_URLS.get(args.dataset, '未知')}")


if __name__ == '__main__':
    main()
