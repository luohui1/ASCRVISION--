"""
钢芯铝绞线智能质检系统 - Web应用
"""
from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
from ultralytics import YOLO
import os
import cv2
import numpy as np
from datetime import datetime
import base64
import json
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['REPORT_FOLDER'] = 'reports'

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORT_FOLDER'], exist_ok=True)

# 加载模型
MODEL_PATH = '../runs/cableinspect_real/weights/best.pt'
model = YOLO(MODEL_PATH)

# 缺陷类别名称（中文）
CLASS_NAMES = {
    0: '断股',
    1: '焊接股', 
    2: '弯曲股',
    3: '划痕',
    4: '压伤',
    5: '散股',
    6: '异物'
}

# 更新模型的类别名称为中文
model.model.names = CLASS_NAMES

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/detect')
def detect_page():
    return render_template('detect.html')

@app.route('/batch')
def batch_page():
    return render_template('batch.html')

@app.route('/3d')
def view_3d():
    return render_template('3d_view.html')

@app.route('/detect', methods=['POST'])
def detect():
    # 支持两种字段名: 'image' 和 'file'
    file = request.files.get('image') or request.files.get('file')
    if not file:
        return jsonify({'error': '没有上传图片'}), 400
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    # 保存上传的图片
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{timestamp}_{file.filename}'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 运行检测
    results = model(filepath, conf=0.25)
    result = results[0]
    
    # 解析检测结果
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        detections.append({
            'class_id': cls_id,
            'class_name': CLASS_NAMES.get(cls_id, f'类别{cls_id}'),
            'confidence': round(conf * 100, 1),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    # 保存标注后的图片
    result_filename = f'result_{filename}'
    result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    result_img = result.plot()
    cv2.imwrite(result_path, result_img)
    
    # 生成质检结论
    if len(detections) == 0:
        conclusion = '合格'
        conclusion_class = 'pass'
    else:
        conclusion = '不合格'
        conclusion_class = 'fail'
    
    return jsonify({
        'success': True,
        'detections': detections,
        'defect_count': len(detections),
        'conclusion': conclusion,
        'conclusion_class': conclusion_class,
        'result_image': f'/results/{result_filename}',
        'timestamp': timestamp
    })

@app.route('/results/<filename>')
def get_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def get_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/report/generate', methods=['POST'])
def generate_report():
    """生成检测报告"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '无检测数据'}), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'report_{timestamp}.txt'
    filepath = os.path.join(app.config['REPORT_FOLDER'], filename)
    
    # 生成报告内容
    defects = data.get('detections', [])
    conclusion = '合格' if len(defects) == 0 else '不合格'
    
    report_content = f"""
================================================================================
                    钢芯铝绞线智能质检报告
                        ACSR VISION
================================================================================

检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
报告编号: RPT-{timestamp}

--------------------------------------------------------------------------------
                           检测结果
--------------------------------------------------------------------------------

质检结论: {conclusion}
缺陷数量: {len(defects)}

"""
    if defects:
        report_content += "缺陷详情:\n"
        for i, d in enumerate(defects, 1):
            report_content += f"  {i}. {d.get('class_name', '未知')} - 置信度: {d.get('confidence', 0)}%\n"
    else:
        report_content += "未检测到缺陷，产品质量合格。\n"
    
    report_content += f"""
--------------------------------------------------------------------------------
                           备注说明
--------------------------------------------------------------------------------

本报告由ACSR VISION智能质检系统自动生成。
检测模型: YOLOv8 + CBAM注意力机制
适用标准: GB/T 1179-2017 钢芯铝绞线

================================================================================
"""
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return jsonify({'success': True, 'filename': filename})

@app.route('/report/download/<filename>')
def download_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename, as_attachment=True)

@app.route('/batch/detect', methods=['POST'])
def batch_detect():
    """批量检测"""
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': '没有上传图片'}), 400
    
    results = []
    for file in files:
        if file.filename == '':
            continue
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = f'{timestamp}_{file.filename}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 运行检测
        det_results = model(filepath, conf=0.25)
        result = det_results[0]
        
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append({
                'class_id': cls_id,
                'class_name': CLASS_NAMES.get(cls_id, f'类别{cls_id}'),
                'confidence': round(conf * 100, 1)
            })
        
        result_filename = f'result_{filename}'
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        result_img = result.plot()
        cv2.imwrite(result_path, result_img)
        
        results.append({
            'filename': file.filename,
            'result_image': f'/results/{result_filename}',
            'detections': detections,
            'conclusion': '合格' if len(detections) == 0 else '不合格'
        })
    
    return jsonify({'success': True, 'results': results, 'total': len(results)})

@app.route('/report/pdf', methods=['POST'])
def generate_pdf_report():
    """生成PDF报告"""
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '无检测数据'}), 400
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'report_{timestamp}.html'
    filepath = os.path.join(app.config['REPORT_FOLDER'], filename)
    
    detections = data.get('detections', [])
    result_image = data.get('result_image', '')
    conclusion = '合格' if len(detections) == 0 else '不合格'
    
    html = f'''<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>质检报告</title>
<style>
body{{font-family:SimHei,sans-serif;padding:40px;max-width:800px;margin:0 auto}}
.header{{text-align:center;border-bottom:2px solid #d4af37;padding-bottom:20px;margin-bottom:30px}}
.logo{{font-size:24px;font-weight:bold;color:#d4af37}}
.title{{font-size:20px;margin-top:10px}}
.info-row{{display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #eee}}
.section{{margin:30px 0}}
.section-title{{font-size:16px;font-weight:bold;color:#333;border-left:4px solid #d4af37;padding-left:10px;margin-bottom:15px}}
.result-box{{padding:20px;border-radius:8px;text-align:center;font-size:24px;font-weight:bold}}
.pass{{background:#d4edda;color:#155724}}
.fail{{background:#f8d7da;color:#721c24}}
.defect-table{{width:100%;border-collapse:collapse;margin-top:15px}}
.defect-table th,.defect-table td{{border:1px solid #ddd;padding:10px;text-align:left}}
.defect-table th{{background:#f5f5f5}}
.image-box{{text-align:center;margin:20px 0}}
.image-box img{{max-width:100%;border:1px solid #ddd}}
.footer{{text-align:center;margin-top:40px;padding-top:20px;border-top:1px solid #ddd;color:#666;font-size:12px}}
@media print{{body{{padding:20px}}}}
</style></head><body>
<div class="header"><div class="logo">ACSR VISION</div><div class="title">钢芯铝绞线智能质检报告</div></div>
<div class="section"><div class="section-title">基本信息</div>
<div class="info-row"><span>报告编号</span><span>RPT-{timestamp}</span></div>
<div class="info-row"><span>检测时间</span><span>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</span></div>
<div class="info-row"><span>执行标准</span><span>GB/T 1179-2017</span></div></div>
<div class="section"><div class="section-title">检测结果</div>
<div class="result-box {'pass' if len(detections)==0 else 'fail'}">{conclusion}</div></div>'''
    
    if result_image:
        html += f'<div class="section"><div class="section-title">检测图像</div><div class="image-box"><img src="{result_image}" alt="检测结果"></div></div>'
    
    if detections:
        html += '<div class="section"><div class="section-title">缺陷详情</div><table class="defect-table"><tr><th>序号</th><th>缺陷类型</th><th>置信度</th></tr>'
        for i, d in enumerate(detections, 1):
            html += f'<tr><td>{i}</td><td>{d.get("class_name","未知")}</td><td>{d.get("confidence",0)}%</td></tr>'
        html += '</table></div>'
    
    html += '<div class="footer"><p>本报告由ACSR VISION智能质检系统自动生成</p><p>检测模型: YOLOv8 + CBAM注意力机制</p></div></body></html>'
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return jsonify({'success': True, 'filename': filename, 'url': f'/report/view/{filename}'})

@app.route('/report/view/<filename>')
def view_report(filename):
    return send_from_directory(app.config['REPORT_FOLDER'], filename)

if __name__ == '__main__':
    print("=" * 50)
    print("钢芯铝绞线智能质检系统")
    print("模型: ", MODEL_PATH)
    print("访问: http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host='0.0.0.0', port=5000)
