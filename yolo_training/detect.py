"""
CableVision AI - YOLO电缆缺陷检测推理脚本
"""
import os
import cv2
import json
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# 缺陷类别信息
DEFECT_INFO = {
    0: {'name': '表面划伤', 'name_en': 'scratch', 'color': (71, 99, 255), 'severity': 'high'},
    1: {'name': '绝缘气泡', 'name_en': 'bubble', 'color': (0, 165, 255), 'severity': 'medium'},
    2: {'name': '护套裂纹', 'name_en': 'crack', 'color': (0, 0, 255), 'severity': 'high'},
    3: {'name': '凹陷变形', 'name_en': 'dent', 'color': (255, 165, 0), 'severity': 'medium'},
    4: {'name': '颜色异常', 'name_en': 'discolor', 'color': (255, 255, 0), 'severity': 'low'},
    5: {'name': '印字缺失', 'name_en': 'print_miss', 'color': (255, 0, 255), 'severity': 'medium'},
    6: {'name': '偏心', 'name_en': 'eccentric', 'color': (0, 255, 255), 'severity': 'high'},
    7: {'name': '杂质', 'name_en': 'impurity', 'color': (128, 0, 128), 'severity': 'medium'},
    8: {'name': '褶皱', 'name_en': 'wrinkle', 'color': (0, 128, 128), 'severity': 'low'},
    9: {'name': '剥离', 'name_en': 'peel', 'color': (128, 128, 0), 'severity': 'high'},
}


class CableDefectDetector:
    """电缆缺陷检测器"""
    
    def __init__(self, model_path, conf_threshold=0.5, iou_threshold=0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
    
    def detect(self, image_path):
        """检测单张图片"""
        results = self.model(
            image_path,
            conf=self.conf_threshold,
            iou=self.iou_threshold
        )[0]
        return self._parse_results(results)
    
    def detect_batch(self, image_paths):
        """批量检测"""
        all_results = []
        for path in image_paths:
            result = self.detect(path)
            result['image_path'] = str(path)
            all_results.append(result)
        return all_results
    
    def detect_video(self, video_path, output_path=None, show=False):
        """视频检测"""
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        defect_frames = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            annotated = results.plot()
            
            if len(results.boxes) > 0:
                defect_frames.append({
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'defects': self._parse_results(results)['defects']
                })
            
            if writer:
                writer.write(annotated)
            if show:
                cv2.imshow('CableVision AI', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        return {'total_frames': frame_count, 'defect_frames': defect_frames}
    
    def _parse_results(self, results):
        """解析检测结果"""
        defects = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            defect_info = DEFECT_INFO.get(cls_id, {})
            defects.append({
                'class_id': cls_id,
                'class_name': defect_info.get('name', f'class_{cls_id}'),
                'class_name_en': defect_info.get('name_en', f'class_{cls_id}'),
                'confidence': round(conf, 4),
                'severity': defect_info.get('severity', 'unknown'),
                'bbox': [round(x, 2) for x in xyxy]
            })
        
        has_defect = len(defects) > 0
        has_critical = any(d['severity'] == 'high' for d in defects)
        
        return {
            'has_defect': has_defect,
            'is_qualified': not has_critical,
            'defect_count': len(defects),
            'defects': defects,
            'timestamp': datetime.now().isoformat()
        }
    
    def visualize(self, image_path, output_path=None, show=False):
        """可视化检测结果"""
        results = self.model(image_path, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        annotated = results.plot()
        
        if output_path:
            cv2.imwrite(output_path, annotated)
        if show:
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated


def main():
    parser = argparse.ArgumentParser(description='CableVision AI - 缺陷检测')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--source', type=str, required=True, help='输入源(图片/视频/目录)')
    parser.add_argument('--output', type=str, default='results', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU阈值')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--save-json', action='store_true', help='保存JSON结果')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    detector = CableDefectDetector(args.model, args.conf, args.iou)
    source = Path(args.source)
    
    if source.is_file():
        if source.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
            output_video = str(Path(args.output) / f'detected_{source.name}')
            results = detector.detect_video(str(source), output_video, args.show)
        else:
            results = detector.detect(str(source))
            output_img = str(Path(args.output) / f'detected_{source.name}')
            detector.visualize(str(source), output_img, args.show)
    elif source.is_dir():
        image_files = list(source.glob('*.jpg')) + list(source.glob('*.png'))
        results = detector.detect_batch(image_files)
        for img_path in image_files:
            output_img = str(Path(args.output) / f'detected_{img_path.name}')
            detector.visualize(str(img_path), output_img)
    
    if args.save_json:
        json_path = Path(args.output) / 'results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {json_path}")
    
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
