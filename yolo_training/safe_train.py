"""
安全训练脚本 - 训练前自动备份，训练后自动备份
"""
from ultralytics import YOLO
import os
import shutil
from datetime import datetime

def backup_weights(note=""):
    """备份当前模型权重"""
    backup_dir = "../backups"
    weights_dir = "../runs/cableinspect_real/weights"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    best_pt = os.path.join(weights_dir, "best.pt")
    if not os.path.exists(best_pt):
        print("没有找到模型，跳过备份")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{timestamp}_{note}" if note else f"backup_{timestamp}"
    backup_path = os.path.join(backup_dir, backup_name)
    os.makedirs(backup_path, exist_ok=True)
    
    shutil.copy2(best_pt, os.path.join(backup_path, "best.pt"))
    last_pt = os.path.join(weights_dir, "last.pt")
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, os.path.join(backup_path, "last.pt"))
    
    print(f"[备份完成] {backup_path}")

def safe_train(epochs=70, resume=True):
    """安全训练 - 自动备份"""
    print("=" * 50)
    print("安全训练模式")
    print("=" * 50)
    
    # 训练前备份
    print("\n[1/3] 训练前备份...")
    backup_weights("before_train")
    
    # 开始训练
    print("\n[2/3] 开始训练...")
    model = YOLO('../runs/cableinspect_real/weights/last.pt')
    model.train(resume=resume, epochs=epochs, workers=0)
    
    # 训练后备份
    print("\n[3/3] 训练后备份...")
    backup_weights(f"epoch{epochs}")
    
    print("\n训练完成！")

if __name__ == "__main__":
    safe_train(epochs=70, resume=True)
