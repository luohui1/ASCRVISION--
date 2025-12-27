"""
CableVision AI - 电缆缺陷数据集生成工具
生成合成训练数据，适配系统10类缺陷检测
"""
import os
import cv2
import numpy as np
import random
import json
from pathlib import Path
from datetime import datetime

# 系统定义的10类缺陷（与detect.py和dataset.yaml一致）
DEFECT_CLASSES = {
    0: 'scratch',      # 表面划伤
    1: 'bubble',       # 绝缘气泡
    2: 'crack',        # 护套裂纹
    3: 'dent',         # 凹陷变形
    4: 'discolor',     # 颜色异常
    5: 'print_miss',   # 印字缺失
    6: 'eccentric',    # 偏心
    7: 'impurity',     # 杂质
    8: 'wrinkle',      # 褶皱
    9: 'peel',         # 剥离
}

class CableImageGenerator:
    """电缆图像生成器"""
    
    def __init__(self, img_size=(640, 640)):
        self.img_size = img_size
        self.cable_colors = [
            (40, 40, 40),    # 黑色
            (200, 200, 200), # 白色
            (50, 50, 180),   # 红色
            (180, 100, 50),  # 蓝色
            (50, 150, 50),   # 绿色
            (50, 180, 180),  # 黄色
        ]
    
    def generate_cable_background(self):
        """生成电缆背景图像"""
        img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # 随机背景色（工业环境）
        bg_color = random.choice([
            (30, 30, 35),   # 深灰
            (45, 42, 40),   # 棕灰
            (25, 25, 30),   # 深蓝灰
        ])
        img[:] = bg_color
        
        # 添加噪声
        noise = np.random.randint(-15, 15, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def draw_cable(self, img, y_center=None, height=None):
        """绘制电缆主体"""
        h, w = img.shape[:2]
        y_center = y_center or h // 2
        height = height or random.randint(80, 150)
        
        cable_color = random.choice(self.cable_colors)
        y1, y2 = y_center - height // 2, y_center + height // 2
        
        # 绘制电缆主体（带渐变效果）
        for y in range(y1, y2):
            ratio = abs(y - y_center) / (height / 2)
            shade = 1 - ratio * 0.3
            color = tuple(int(c * shade) for c in cable_color)
            cv2.line(img, (0, y), (w, y), color, 1)
        
        # 添加高光
        highlight_y = y_center - height // 4
        cv2.line(img, (0, highlight_y), (w, highlight_y), 
                 tuple(min(255, int(c * 1.3)) for c in cable_color), 2)
        
        return img, (0, y1, w, y2)
    
    def add_print_text(self, img, cable_bbox):
        """添加电缆印字"""
        x1, y1, x2, y2 = cable_bbox
        y_center = (y1 + y2) // 2
        
        texts = [
            "YJV-3x150 10kV 2024",
            "GB/T 12706.2",
            "CABLE VISION AI",
            "BV-2.5mm2 450/750V",
        ]
        text = random.choice(texts)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = random.uniform(0.4, 0.6)
        thickness = 1
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x = random.randint(50, max(51, img.shape[1] - text_size[0] - 50))
        y = y_center + random.randint(-10, 10)
        
        # 印字颜色（白色或浅色）
        text_color = (200, 200, 200) if random.random() > 0.5 else (180, 180, 150)
        cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)
        
        return img, (x, y - text_size[1], x + text_size[0], y)


class DefectGenerator:
    """缺陷生成器"""
    
    @staticmethod
    def add_scratch(img, cable_bbox):
        """添加划伤缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        # 随机划伤位置和大小
        sx = random.randint(x1 + 50, x2 - 100)
        sy = random.randint(y1 + 10, y2 - 10)
        length = random.randint(30, 120)
        
        # 划伤颜色（比电缆深或浅）
        color = random.choice([(60, 60, 70), (150, 140, 130)])
        thickness = random.randint(1, 3)
        
        # 绘制划伤（可能有角度）
        angle = random.uniform(-0.2, 0.2)
        ex = sx + length
        ey = int(sy + length * angle)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)
        
        # 计算边界框
        bbox = (min(sx, ex) - 5, min(sy, ey) - 5, 
                max(sx, ex) + 5, max(sy, ey) + 5)
        return img, bbox, 0  # class_id = 0
    
    @staticmethod
    def add_bubble(img, cable_bbox):
        """添加气泡缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        cx = random.randint(x1 + 50, x2 - 50)
        cy = random.randint(y1 + 15, y2 - 15)
        radius = random.randint(5, 20)
        
        # 气泡效果（半透明圆形）
        overlay = img.copy()
        color = (180, 180, 170)
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        
        # 高光
        cv2.circle(img, (cx - radius//3, cy - radius//3), radius//4, (220, 220, 210), -1)
        
        bbox = (cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3)
        return img, bbox, 1  # class_id = 1
    
    @staticmethod
    def add_crack(img, cable_bbox):
        """添加裂纹缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        sx = random.randint(x1 + 50, x2 - 80)
        sy = random.randint(y1 + 5, y2 - 5)
        
        # 绘制不规则裂纹
        points = [(sx, sy)]
        for _ in range(random.randint(3, 6)):
            dx = random.randint(5, 20)
            dy = random.randint(-15, 15)
            points.append((points[-1][0] + dx, points[-1][1] + dy))
        
        color = (30, 30, 35)
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], color, random.randint(1, 2))
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs) - 5, min(ys) - 5, max(xs) + 5, max(ys) + 5)
        return img, bbox, 2  # class_id = 2
    
    @staticmethod
    def add_dent(img, cable_bbox):
        """添加凹陷缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        cx = random.randint(x1 + 50, x2 - 50)
        cy = (y1 + y2) // 2
        w = random.randint(20, 50)
        h = random.randint(10, 25)
        
        # 绘制椭圆形凹陷（阴影效果）
        color = (50, 50, 55)
        cv2.ellipse(img, (cx, cy), (w, h), 0, 0, 360, color, -1)
        
        bbox = (cx - w - 3, cy - h - 3, cx + w + 3, cy + h + 3)
        return img, bbox, 3  # class_id = 3
    
    @staticmethod
    def add_discolor(img, cable_bbox):
        """添加颜色异常"""
        x1, y1, x2, y2 = cable_bbox
        
        rx = random.randint(x1 + 30, x2 - 80)
        ry = random.randint(y1 + 5, y2 - 30)
        rw = random.randint(30, 80)
        rh = random.randint(20, min(40, y2 - ry - 5))
        
        # 异常颜色区域
        overlay = img.copy()
        color = random.choice([(100, 80, 60), (80, 100, 80), (60, 60, 100)])
        cv2.rectangle(overlay, (rx, ry), (rx + rw, ry + rh), color, -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        
        bbox = (rx, ry, rx + rw, ry + rh)
        return img, bbox, 4  # class_id = 4
    
    @staticmethod
    def add_print_miss(img, cable_bbox, text_bbox=None):
        """添加印字缺失"""
        x1, y1, x2, y2 = cable_bbox
        
        if text_bbox:
            tx1, ty1, tx2, ty2 = text_bbox
            # 遮盖部分印字
            mx = random.randint(tx1, max(tx1+1, tx2 - 30))
            mw = random.randint(20, min(50, tx2 - mx))
        else:
            mx = random.randint(x1 + 100, x2 - 100)
            mw = random.randint(30, 60)
            ty1 = (y1 + y2) // 2 - 10
            ty2 = (y1 + y2) // 2 + 10
        
        # 用电缆颜色遮盖
        region = img[ty1:ty2, mx:mx+mw]
        if region.size > 0:
            avg_color = tuple(map(int, region.mean(axis=(0,1))))
            cv2.rectangle(img, (mx, ty1), (mx + mw, ty2), avg_color, -1)
        
        bbox = (mx, ty1, mx + mw, ty2)
        return img, bbox, 5  # class_id = 5
    
    @staticmethod
    def add_eccentric(img, cable_bbox):
        """添加偏心缺陷（通过视觉效果模拟）"""
        x1, y1, x2, y2 = cable_bbox
        h = y2 - y1
        
        # 在一侧添加不均匀厚度效果
        side = random.choice(['top', 'bottom'])
        sx = random.randint(x1 + 50, x2 - 100)
        sw = random.randint(60, 120)
        
        if side == 'top':
            sy, sh = y1, random.randint(5, 15)
            color = (70, 70, 75)
        else:
            sy, sh = y2 - random.randint(5, 15), random.randint(5, 15)
            color = (35, 35, 40)
        
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), color, -1)
        
        bbox = (sx, sy, sx + sw, sy + sh)
        return img, bbox, 6  # class_id = 6
    
    @staticmethod
    def add_impurity(img, cable_bbox):
        """添加杂质缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        # 随机小点/斑点
        num_spots = random.randint(2, 5)
        spots = []
        for _ in range(num_spots):
            cx = random.randint(x1 + 30, x2 - 30)
            cy = random.randint(y1 + 10, y2 - 10)
            r = random.randint(2, 6)
            color = random.choice([(20, 20, 25), (100, 90, 80)])
            cv2.circle(img, (cx, cy), r, color, -1)
            spots.append((cx, cy, r))
        
        # 计算包围所有杂质的边界框
        xs = [s[0] for s in spots]
        ys = [s[1] for s in spots]
        rs = [s[2] for s in spots]
        bbox = (min(xs) - max(rs) - 3, min(ys) - max(rs) - 3,
                max(xs) + max(rs) + 3, max(ys) + max(rs) + 3)
        return img, bbox, 7  # class_id = 7
    
    @staticmethod
    def add_wrinkle(img, cable_bbox):
        """添加褶皱缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        sx = random.randint(x1 + 50, x2 - 80)
        sy = random.randint(y1 + 10, y2 - 10)
        
        # 绘制波浪线模拟褶皱
        points = []
        for i in range(random.randint(4, 8)):
            x = sx + i * 10
            y = sy + int(5 * np.sin(i * 1.5))
            points.append((x, y))
        
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], (60, 60, 65), 2)
        
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = (min(xs) - 5, min(ys) - 8, max(xs) + 5, max(ys) + 8)
        return img, bbox, 8  # class_id = 8
    
    @staticmethod
    def add_peel(img, cable_bbox):
        """添加剥离缺陷"""
        x1, y1, x2, y2 = cable_bbox
        
        px = random.randint(x1 + 50, x2 - 80)
        py = random.choice([y1, y2 - 15])
        pw = random.randint(30, 70)
        ph = random.randint(10, 20)
        
        # 剥离效果（露出内层）
        inner_color = (180, 140, 100)  # 铜色
        cv2.rectangle(img, (px, py), (px + pw, py + ph), inner_color, -1)
        
        # 边缘效果
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (50, 50, 55), 1)
        
        bbox = (px - 2, py - 2, px + pw + 2, py + ph + 2)
        return img, bbox, 9  # class_id = 9


class DatasetGenerator:
    """数据集生成器主类"""
    
    def __init__(self, output_dir='dataset', img_size=(640, 640)):
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.cable_gen = CableImageGenerator(img_size)
        self.defect_methods = [
            DefectGenerator.add_scratch,
            DefectGenerator.add_bubble,
            DefectGenerator.add_crack,
            DefectGenerator.add_dent,
            DefectGenerator.add_discolor,
            DefectGenerator.add_print_miss,
            DefectGenerator.add_eccentric,
            DefectGenerator.add_impurity,
            DefectGenerator.add_wrinkle,
            DefectGenerator.add_peel,
        ]
    
    def setup_dirs(self):
        """创建数据集目录结构"""
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    def bbox_to_yolo(self, bbox, img_w, img_h):
        """转换为YOLO格式 (x_center, y_center, width, height) 归一化"""
        x1, y1, x2, y2 = bbox
        x1, x2 = max(0, x1), min(img_w, x2)
        y1, y2 = max(0, y1), min(img_h, y2)
        
        x_center = (x1 + x2) / 2 / img_w
        y_center = (y1 + y2) / 2 / img_h
        width = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        
        return x_center, y_center, width, height
    
    def generate_sample(self, include_defects=True, max_defects=3):
        """生成单个样本"""
        img = self.cable_gen.generate_cable_background()
        img, cable_bbox = self.cable_gen.draw_cable(img)
        img, text_bbox = self.cable_gen.add_print_text(img, cable_bbox)
        
        annotations = []
        
        if include_defects:
            num_defects = random.randint(1, max_defects)
            selected = random.sample(range(10), min(num_defects, 10))
            
            for class_id in selected:
                try:
                    if class_id == 5:  # print_miss需要text_bbox
                        img, bbox, cid = self.defect_methods[class_id](img, cable_bbox, text_bbox)
                    else:
                        img, bbox, cid = self.defect_methods[class_id](img, cable_bbox)
                    
                    yolo_bbox = self.bbox_to_yolo(bbox, self.img_size[0], self.img_size[1])
                    if all(0 <= v <= 1 for v in yolo_bbox) and yolo_bbox[2] > 0.01 and yolo_bbox[3] > 0.01:
                        annotations.append((cid, *yolo_bbox))
                except:
                    continue
        
        return img, annotations
    
    def generate_dataset(self, num_train=800, num_val=150, num_test=50):
        """生成完整数据集"""
        self.setup_dirs()
        splits = {'train': num_train, 'val': num_val, 'test': num_test}
        
        for split, num in splits.items():
            print(f"生成 {split} 集: {num} 张图片...")
            for i in range(num):
                # 80%有缺陷，20%正常
                has_defect = random.random() < 0.8
                img, annotations = self.generate_sample(include_defects=has_defect)
                
                # 保存图片
                img_name = f"cable_{split}_{i:05d}.jpg"
                img_path = self.output_dir / 'images' / split / img_name
                cv2.imwrite(str(img_path), img)
                
                # 保存标注
                label_name = f"cable_{split}_{i:05d}.txt"
                label_path = self.output_dir / 'labels' / split / label_name
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
                
                if (i + 1) % 100 == 0:
                    print(f"  已生成 {i + 1}/{num}")
        
        print(f"\n数据集生成完成! 保存在: {self.output_dir}")
        print(f"  训练集: {num_train} 张")
        print(f"  验证集: {num_val} 张")
        print(f"  测试集: {num_test} 张")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='生成电缆缺陷检测数据集')
    parser.add_argument('--output', type=str, default='dataset', help='输出目录')
    parser.add_argument('--train', type=int, default=800, help='训练集数量')
    parser.add_argument('--val', type=int, default=150, help='验证集数量')
    parser.add_argument('--test', type=int, default=50, help='测试集数量')
    parser.add_argument('--size', type=int, default=640, help='图像尺寸')
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.output, (args.size, args.size))
    generator.generate_dataset(args.train, args.val, args.test)


if __name__ == '__main__':
    main()
