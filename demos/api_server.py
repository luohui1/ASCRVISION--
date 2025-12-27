"""
CableVision AI - Flask后端API服务
集成YOLO模型提供检测接口
"""
import os
import sys
import json
import base64
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO

# 添加yolo_training到路径
sys.path.insert(0, str(Path(__file__).parent / 'yolo_training'))
from services.report_ai import AIReportGenerator

app = Flask(__name__)
CORS(app)

# 配置
MODEL_PATH = r"E:\工作\个人项目开发\电缆AI\runs\train\cable_defect\weights\best.pt"
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# 加载模型
print("加载模型...")
model = YOLO(MODEL_PATH)
print("模型加载完成!")

# 缺陷类别信息
DEFECT_INFO = {
    0: {'name': '表面划伤', 'name_en': 'scratch', 'color': '#ff4757', 'severity': 'high'},
    1: {'name': '绝缘气泡', 'name_en': 'bubble', 'color': '#ffa502', 'severity': 'medium'},
    2: {'name': '护套裂纹', 'name_en': 'crack', 'color': '#ff6348', 'severity': 'high'},
    3: {'name': '凹陷变形', 'name_en': 'dent', 'color': '#ffd93d', 'severity': 'medium'},
    4: {'name': '颜色异常', 'name_en': 'discolor', 'color': '#4dabf7', 'severity': 'low'},
    5: {'name': '印字缺失', 'name_en': 'print_miss', 'color': '#a55eea', 'severity': 'medium'},
    6: {'name': '偏心', 'name_en': 'eccentric', 'color': '#00d4ff', 'severity': 'high'},
    7: {'name': '杂质', 'name_en': 'impurity', 'color': '#8854d0', 'severity': 'medium'},
    8: {'name': '褶皱', 'name_en': 'wrinkle', 'color': '#20bf6b', 'severity': 'low'},
    9: {'name': '剥离', 'name_en': 'peel', 'color': '#eb3b5a', 'severity': 'high'},
}

# 统计数据
stats = {
    'total_detections': 0,
    'pass_count': 0,
    'fail_count': 0,
    'defect_counts': {i: 0 for i in range(10)},
    'history': []
}


def detect_image(image_path, conf=0.5):
    """检测单张图片"""
    results = model(image_path, conf=conf)[0]
    defects = []
    
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_score = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        
        info = DEFECT_INFO.get(cls_id, {})
        defects.append({
            'class_id': cls_id,
            'class_name': info.get('name', f'类别{cls_id}'),
            'class_name_en': info.get('name_en', f'class_{cls_id}'),
            'confidence': round(conf_score, 4),
            'severity': info.get('severity', 'unknown'),
            'color': info.get('color', '#888'),
            'bbox': [round(x, 2) for x in xyxy]
        })
    
    has_defect = len(defects) > 0
    has_critical = any(d['severity'] == 'high' for d in defects)
    
    return {
        'has_defect': has_defect,
        'is_qualified': not has_critical,
        'defect_count': len(defects),
        'defects': defects
    }


@app.route('/api/detect', methods=['POST'])
def api_detect():
    """图片检测接口"""
    try:
        if 'image' not in request.files:
            # 尝试从base64获取
            data = request.get_json()
            if data and 'image' in data:
                img_data = base64.b64decode(data['image'].split(',')[-1])
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img_id = str(uuid.uuid4())[:8]
                img_path = UPLOAD_DIR / f"{img_id}.jpg"
                cv2.imwrite(str(img_path), img)
            else:
                return jsonify({'error': '未提供图片'}), 400
        else:
            file = request.files['image']
            img_id = str(uuid.uuid4())[:8]
            img_path = UPLOAD_DIR / f"{img_id}_{file.filename}"
            file.save(str(img_path))
        
        conf = float(request.form.get('conf', 0.5))
        result = detect_image(str(img_path), conf)
        
        # 生成标注图片
        results = model(str(img_path), conf=conf)[0]
        annotated = results.plot()
        result_path = RESULTS_DIR / f"result_{img_id}.jpg"
        cv2.imwrite(str(result_path), annotated)
        
        # 更新统计
        stats['total_detections'] += 1
        if result['is_qualified']:
            stats['pass_count'] += 1
        else:
            stats['fail_count'] += 1
        for d in result['defects']:
            stats['defect_counts'][d['class_id']] += 1
        
        # 记录历史
        record = {
            'id': img_id,
            'timestamp': datetime.now().isoformat(),
            'result': result,
            'image_path': str(result_path)
        }
        stats['history'].insert(0, record)
        if len(stats['history']) > 100:
            stats['history'] = stats['history'][:100]
        
        result['id'] = img_id
        result['result_image'] = f"/api/result/{img_id}"
        result['timestamp'] = record['timestamp']
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/result/<img_id>')
def get_result_image(img_id):
    """获取检测结果图片"""
    result_path = RESULTS_DIR / f"result_{img_id}.jpg"
    if result_path.exists():
        return send_file(str(result_path), mimetype='image/jpeg')
    return jsonify({'error': '图片不存在'}), 404


@app.route('/api/stats')
def get_stats():
    """获取统计数据"""
    pass_rate = 0
    if stats['total_detections'] > 0:
        pass_rate = round(stats['pass_count'] / stats['total_detections'] * 100, 1)
    
    defect_distribution = []
    for cls_id, count in stats['defect_counts'].items():
        if count > 0:
            info = DEFECT_INFO[cls_id]
            defect_distribution.append({
                'class_id': cls_id,
                'name': info['name'],
                'count': count,
                'color': info['color']
            })
    defect_distribution.sort(key=lambda x: x['count'], reverse=True)
    
    return jsonify({
        'total_detections': stats['total_detections'],
        'pass_count': stats['pass_count'],
        'fail_count': stats['fail_count'],
        'pass_rate': pass_rate,
        'defect_distribution': defect_distribution
    })


@app.route('/api/history')
def get_history():
    """获取检测历史"""
    limit = int(request.args.get('limit', 20))
    return jsonify(stats['history'][:limit])


@app.route('/api/defect_types')
def get_defect_types():
    """获取缺陷类型列表"""
    return jsonify(DEFECT_INFO)


@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


# ========== 智能报告生成接口 ==========
# 初始化报告生成器
report_generator = AIReportGenerator(api_type='qwen')

@app.route('/api/report/generate', methods=['POST'])
def generate_report():
    """生成智能质检报告"""
    try:
        data = request.get_json()
        detection_result = data.get('detection_result')
        sample_info = data.get('sample_info', {})
        
        if not detection_result:
            return jsonify({'error': '缺少检测结果数据'}), 400
        
        report = report_generator.generate_report(detection_result, sample_info)
        return jsonify(report)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/report/export/<format_type>', methods=['POST'])
def export_report(format_type):
    """导出报告 (json/markdown)"""
    try:
        data = request.get_json()
        report = data.get('report')
        
        if not report:
            return jsonify({'error': '缺少报告数据'}), 400
        
        report_id = report.get('report_id', 'report')
        
        if format_type == 'json':
            filepath = RESULTS_DIR / f"{report_id}.json"
            report_generator.export_to_json(report, str(filepath))
            return send_file(str(filepath), as_attachment=True)
        
        elif format_type == 'markdown':
            filepath = RESULTS_DIR / f"{report_id}.md"
            report_generator.export_to_markdown(report, str(filepath))
            return send_file(str(filepath), as_attachment=True)
        
        else:
            return jsonify({'error': '不支持的格式'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/detect_with_report', methods=['POST'])
def detect_with_report():
    """检测并生成报告（一站式接口）"""
    try:
        # 先执行检测
        if 'image' not in request.files:
            return jsonify({'error': '未提供图片'}), 400
        
        file = request.files['image']
        img_id = str(uuid.uuid4())[:8]
        img_path = UPLOAD_DIR / f"{img_id}_{file.filename}"
        file.save(str(img_path))
        
        conf = float(request.form.get('conf', 0.5))
        detection_result = detect_image(str(img_path), conf)
        
        # 生成标注图片
        results = model(str(img_path), conf=conf)[0]
        annotated = results.plot()
        result_path = RESULTS_DIR / f"result_{img_id}.jpg"
        cv2.imwrite(str(result_path), annotated)
        
        # 生成报告
        sample_info = {
            'image_id': img_id,
            'filename': file.filename,
            'confidence_threshold': conf
        }
        report = report_generator.generate_report(detection_result, sample_info)
        
        # 更新统计
        stats['total_detections'] += 1
        if detection_result['is_qualified']:
            stats['pass_count'] += 1
        else:
            stats['fail_count'] += 1
        
        return jsonify({
            'detection': detection_result,
            'report': report,
            'result_image': f"/api/result/{img_id}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("启动CableVision AI后端服务...")
    print("API地址: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
