"""
模型备份系统 - 自动备份训练好的模型
使用方法:
    python backup.py                    # 备份当前最佳模型
    python backup.py --list             # 列出所有备份
    python backup.py --restore v1       # 恢复指定版本
"""
import os
import shutil
import argparse
from datetime import datetime

BACKUP_DIR = "../backups"
WEIGHTS_DIR = "../runs/cableinspect_real/weights"

def ensure_backup_dir():
    os.makedirs(BACKUP_DIR, exist_ok=True)

def backup_model(note=""):
    ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    best_pt = os.path.join(WEIGHTS_DIR, "best.pt")
    last_pt = os.path.join(WEIGHTS_DIR, "last.pt")
    
    if not os.path.exists(best_pt):
        print("错误: 没有找到模型文件")
        return False
    
    backup_name = f"backup_{timestamp}"
    if note:
        backup_name += f"_{note}"
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    os.makedirs(backup_path, exist_ok=True)
    
    shutil.copy2(best_pt, os.path.join(backup_path, "best.pt"))
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, os.path.join(backup_path, "last.pt"))
    
    # 保存备份信息
    info_file = os.path.join(backup_path, "info.txt")
    with open(info_file, "w", encoding="utf-8") as f:
        f.write(f"备份时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"备注: {note}\n")
        f.write(f"best.pt 大小: {os.path.getsize(os.path.join(backup_path, 'best.pt')) / 1024 / 1024:.2f} MB\n")
    
    print(f"备份成功: {backup_path}")
    return True

def list_backups():
    ensure_backup_dir()
    backups = []
    for name in os.listdir(BACKUP_DIR):
        path = os.path.join(BACKUP_DIR, name)
        if os.path.isdir(path):
            info_file = os.path.join(path, "info.txt")
            info = ""
            if os.path.exists(info_file):
                with open(info_file, "r", encoding="utf-8") as f:
                    info = f.read()
            backups.append((name, info))
    
    if not backups:
        print("没有找到备份")
        return
    
    print("=" * 50)
    print("可用备份列表:")
    print("=" * 50)
    for name, info in sorted(backups, reverse=True):
        print(f"\n[{name}]")
        print(info)

def restore_backup(backup_name):
    backup_path = os.path.join(BACKUP_DIR, backup_name)
    if not os.path.exists(backup_path):
        # 尝试模糊匹配
        for name in os.listdir(BACKUP_DIR):
            if backup_name in name:
                backup_path = os.path.join(BACKUP_DIR, name)
                break
    
    if not os.path.exists(backup_path):
        print(f"错误: 找不到备份 {backup_name}")
        return False
    
    best_pt = os.path.join(backup_path, "best.pt")
    if not os.path.exists(best_pt):
        print("错误: 备份文件损坏")
        return False
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    shutil.copy2(best_pt, os.path.join(WEIGHTS_DIR, "best.pt"))
    
    last_pt = os.path.join(backup_path, "last.pt")
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, os.path.join(WEIGHTS_DIR, "last.pt"))
    
    print(f"恢复成功: {backup_path} -> {WEIGHTS_DIR}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型备份系统")
    parser.add_argument("--list", action="store_true", help="列出所有备份")
    parser.add_argument("--restore", type=str, help="恢复指定备份")
    parser.add_argument("--note", type=str, default="", help="备份备注")
    args = parser.parse_args()
    
    if args.list:
        list_backups()
    elif args.restore:
        restore_backup(args.restore)
    else:
        backup_model(args.note)
