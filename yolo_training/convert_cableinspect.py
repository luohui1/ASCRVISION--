"""
将CableInspect-AD数据集从COCO格式转换为YOLO格式
"""
import json
import os
import shutil
from pathlib import Path
import random

# 路径配置
DATASET_ROOT = Path(r"E:\工作\个人项目开发\电缆AI\新建文件夹\yolo_training\CableInspect-AD\CableInspect-AD")
OUTPUT_ROOT = Path(r"E:\工作\个人项目开发\电缆AI\新建文件夹\yolo_training\cableinspect_yolo")

# 类别映射 (COCO类别名 -> YOLO类别ID)
CATEGORIES = {
    'broken strand': 0,
    'welded strand': 1,
    'bent strand': 2,
    'long scratch': 3,
    'crushed': 4,
    'spaced strand': 5,
    'deposit': 6
}

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """将COCO格式bbox [x,y,w,h] 转换为YOLO格式 [cx,cy,w,h] (归一化)"""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    return [cx, cy, nw, nh]

def process_cable(cable_name, json_file):
    """处理单个电缆的数据"""
    print(f"\n处理 {cable_name}...")
    
    with open(DATASET_ROOT / json_file, 'r') as f:
        data = json.load(f)
    
    # 建立类别ID映射
    coco_cat_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 建立图片ID到信息的映射
    img_info = {img['id']: img for img in data['images']}
    
    # 建立图片ID到标注的映射
    img_annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    results = []
    
    for img in data['images']:
        img_id = img['id']
        file_name = img['file_name']
        width = img['width']
        height = img['height']
        
        # 文件名已包含完整相对路径，如 Cable_1/images/01/xxx.png
        img_path = DATASET_ROOT / file_name
        
        if not img_path.exists():
            continue
        
        # 获取标注
        annotations = img_annotations.get(img_id, [])
        
        # 转换为YOLO格式
        yolo_labels = []
        for ann in annotations:
            cat_name = coco_cat_to_name[ann['category_id']]
            if cat_name in CATEGORIES:
                class_id = CATEGORIES[cat_name]
                bbox = coco_to_yolo_bbox(ann['bbox'], width, height)
                yolo_labels.append(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}")
        
        results.append({
            'img_path': img_path,
            'labels': yolo_labels,
            'has_defect': len(yolo_labels) > 0
        })
    
    print(f"  找到 {len(results)} 张图片, {sum(1 for r in results if r['has_defect'])} 张有缺陷")
    return results

def main():
    print("=" * 50)
    print("CableInspect-AD -> YOLO 格式转换")
    print("=" * 50)
    
    # 创建输出目录
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 处理所有电缆
    all_data = []
    all_data.extend(process_cable('Cable_1', 'cable_1.json'))
    all_data.extend(process_cable('Cable_2', 'cable_2.json'))
    all_data.extend(process_cable('Cable_3', 'cable_3.json'))
    
    print(f"\n总计: {len(all_data)} 张图片")
    
    # 打乱并分割数据集 (70% train, 20% val, 10% test)
    random.seed(42)
    random.shuffle(all_data)
    
    n_total = len(all_data)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.2)
    
    splits = {
        'train': all_data[:n_train],
        'val': all_data[n_train:n_train+n_val],
        'test': all_data[n_train+n_val:]
    }
    
    # 复制文件
    for split_name, split_data in splits.items():
        print(f"\n处理 {split_name} 集 ({len(split_data)} 张)...")
        for i, item in enumerate(split_data):
            # 复制图片
            new_img_name = f"cable_{split_name}_{i:05d}.jpg"
            dst_img = OUTPUT_ROOT / 'images' / split_name / new_img_name
            shutil.copy2(item['img_path'], dst_img)
            
            # 写入标签
            new_label_name = f"cable_{split_name}_{i:05d}.txt"
            dst_label = OUTPUT_ROOT / 'labels' / split_name / new_label_name
            with open(dst_label, 'w') as f:
                f.write('\n'.join(item['labels']))
            
            if (i + 1) % 500 == 0:
                print(f"  已处理 {i+1}/{len(split_data)}")
    
    # 创建dataset.yaml
    yaml_content = f"""# CableInspect-AD Dataset (YOLO格式)
path: {OUTPUT_ROOT}
train: images/train
val: images/val
test: images/test

nc: 7
names:
  0: broken_strand
  1: welded_strand
  2: bent_strand
  3: long_scratch
  4: crushed
  5: spaced_strand
  6: deposit
"""
    
    with open(OUTPUT_ROOT / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "=" * 50)
    print("转换完成!")
    print(f"训练集: {len(splits['train'])} 张")
    print(f"验证集: {len(splits['val'])} 张")
    print(f"测试集: {len(splits['test'])} 张")
    print(f"输出目录: {OUTPUT_ROOT}")
    print("=" * 50)

if __name__ == '__main__':
    main()
