"""
电缆缺陷AI实时检测演示 - Web版本
"""
from flask import Flask, Response, render_template_string
import cv2
import os
import glob
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

app = Flask(__name__)

# 配置
MODEL_PATH = "runs/cableinspect_real/weights/best.pt"
IMAGE_DIR = "yolo_training/cableinspect_yolo/images/test"

CLASS_NAMES_CN = {
    0: '断股', 1: '焊接股', 2: '弯曲股', 
    3: '划痕', 4: '压伤', 5: '散股', 6: '异物'
}

COLORS = {
    0: (0, 0, 255), 1: (0, 165, 255), 2: (0, 255, 255),
    3: (255, 0, 255), 4: (255, 0, 0), 5: (0, 255, 0), 6: (255, 255, 0),
}

model = None
images = []

# 加载中文字体
def get_font(size=20):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
    except:
        return ImageFont.load_default()

def put_chinese_text(img, text, pos, color, size=20):
    """在图片上绘制中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = get_font(size)
    # BGR转RGB颜色
    rgb_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=font, fill=rgb_color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def load_model():
    global model, images
    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    print(f"加载 {len(images)} 张图片")

def generate_frames():
    idx = 0
    while True:
        frame = cv2.imread(images[idx])
        if frame is None:
            idx = (idx + 1) % len(images)
            continue
        
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)))
        
        results = model(frame, conf=0.5, verbose=False)
        result = results[0]
        
        annotated = frame.copy()
        total_defects = 0
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            color = COLORS.get(cls, (255, 255, 255))
            label = f"{CLASS_NAMES_CN.get(cls)} {conf:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            annotated = put_chinese_text(annotated, label, (x1, y1-25), color, 18)
            total_defects += 1
        
        info = f"[{idx+1}/{len(images)}]"
        annotated = put_chinese_text(annotated, info, (10, 10), (0, 255, 0), 24)
        
        status = "合格" if total_defects == 0 else f"缺陷 ({total_defects})"
        color = (0, 255, 0) if total_defects == 0 else (0, 0, 255)
        annotated = put_chinese_text(annotated, status, (annotated.shape[1]-150, 20), color, 36)
        
        ret, buffer = cv2.imencode('.jpg', annotated)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        idx = (idx + 1) % len(images)
        time.sleep(0.5)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>电缆缺陷AI检测演示</title>
    <style>
        body { margin: 0; background: #1a1a2e; color: white; font-family: Arial; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; text-align: center; }
        .header h1 { margin: 0; font-size: 28px; }
        .container { display: flex; justify-content: center; padding: 20px; }
        img { max-width: 100%; border: 3px solid #667eea; border-radius: 10px; }
        .info { text-align: center; padding: 10px; color: #aaa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ACSR电缆缺陷AI实时检测系统</h1>
    </div>
    <div class="container">
        <img src="/video_feed" alt="检测画面">
    </div>
    <div class="info">实时AI检测演示 | 模型: YOLOv8 | 7类缺陷识别</div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_model()
    print("启动Web服务: http://localhost:5005")
    app.run(host='0.0.0.0', port=5005, threaded=True)
