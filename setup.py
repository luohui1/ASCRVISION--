#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ACSR Vision - 一键环境检查与安装脚本
============================================
功能:
  1. 检查Python版本
  2. 检查并安装pip
  3. 检查并安装PyTorch (自动选择CPU/GPU版本)
  4. 检查并安装所有依赖包
  5. 检查GPU/CUDA环境
  6. 检查模型文件
  7. 检查数据集
  8. 提供快速启动指南

运行方式: 
  python setup.py           # 完整检查和安装
  python setup.py --check   # 仅检查不安装
  python setup.py --install # 强制重新安装所有依赖
"""

import subprocess
import sys
import os
import platform
import argparse
import urllib.request
import shutil
import time

# ============================================
# 配置区域
# ============================================

# Python版本要求
MIN_PYTHON_VERSION = (3, 8)
MAX_PYTHON_VERSION = (3, 12)

# 必需的依赖包
REQUIRED_PACKAGES = [
    "ultralytics>=8.0.0",
    "opencv-python>=4.8.0",
    "flask>=2.0.0",
    "flask-cors>=4.0.0",
    "pillow>=9.0.0",
    "numpy>=1.24.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.7.0",
    "pandas>=2.0.0",
    "seaborn>=0.12.0",
    "requests>=2.28.0",
]

# 可选依赖（用于高级功能）
OPTIONAL_PACKAGES = [
    ("onnx", "ONNX模型导出"),
    ("onnxruntime", "ONNX推理加速"),
    ("tensorboard", "训练可视化"),
]

# PyTorch安装源 (使用清华镜像加速)
PYTORCH_URLS = {
    "cuda121": "torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121",
    "cuda118": "torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu118",
    "cpu": "torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple",
}

# 国内pip镜像
PIP_MIRROR = "https://pypi.tuna.tsinghua.edu.cn/simple"

# ============================================
# 工具函数
# ============================================

class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{Colors.RESET}")

def print_status(name, status, detail=""):
    if status:
        icon = f"{Colors.GREEN}[OK]{Colors.RESET}"
    else:
        icon = f"{Colors.RED}[X]{Colors.RESET}"
    msg = f"  {icon} {name}"
    if detail:
        msg += f" - {detail}"
    print(msg)

def print_info(text):
    print(f"  {Colors.BLUE}[INFO]{Colors.RESET} {text}")

def print_warn(text):
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {text}")

def run_command(cmd, show_output=False):
    """运行命令并返回结果"""
    try:
        if show_output:
            result = subprocess.run(cmd, shell=True, check=True)
        else:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return True, result.stdout if hasattr(result, 'stdout') else ""
    except subprocess.CalledProcessError as e:
        return False, str(e)

def get_script_dir():
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))

# ============================================
# 检查函数
# ============================================

def check_python():
    """检查Python版本"""
    print_header("1. 检查 Python 环境")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if MIN_PYTHON_VERSION <= (version.major, version.minor) <= MAX_PYTHON_VERSION:
        print_status("Python版本", True, f"{version_str} (推荐 3.9-3.11)")
        print_status("Python路径", True, sys.executable)
        return True
    else:
        print_status("Python版本", False, f"{version_str}")
        print_warn(f"需要 Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} - {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}")
        print_info("下载地址: https://www.python.org/downloads/")
        return False

def check_pip():
    """检查pip"""
    print_header("2. 检查 pip 包管理器")
    try:
        import pip
        version = pip.__version__
        print_status("pip已安装", True, f"v{version}")
        return True
    except ImportError:
        print_status("pip已安装", False)
        print_info("正在安装pip...")
        success, _ = run_command(f"{sys.executable} -m ensurepip --default-pip")
        if success:
            print_status("pip安装", True)
            return True
        else:
            print_status("pip安装", False)
            print_info("请手动安装: python -m ensurepip --default-pip")
            return False

def check_and_install_pytorch(auto_install=False):
    """检查并安装PyTorch"""
    print_header("3. 检查 PyTorch")
    
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        
        print_status("PyTorch已安装", True, f"v{version}")
        
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_status("CUDA可用", True, f"v{cuda_version}")
            print_status("GPU设备", True, gpu_name)
            print_status("GPU显存", True, f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print_status("CUDA可用", False, "将使用CPU运行")
            print_warn("GPU加速不可用，推理速度会较慢")
        return True
        
    except ImportError:
        print_status("PyTorch已安装", False)
        
        if not auto_install:
            choice = input("\n是否安装PyTorch? (y/n): ").strip().lower()
            if choice != 'y':
                print_info("跳过PyTorch安装")
                return False
        
        # 检测CUDA版本
        print_info("检测系统CUDA版本...")
        cuda_version = detect_cuda_version()
        
        if cuda_version:
            print_info(f"检测到CUDA {cuda_version}")
            if cuda_version >= 12.1:
                pytorch_cmd = PYTORCH_URLS["cuda121"]
            elif cuda_version >= 11.8:
                pytorch_cmd = PYTORCH_URLS["cuda118"]
            else:
                pytorch_cmd = PYTORCH_URLS["cpu"]
                print_warn(f"CUDA {cuda_version} 版本较旧，使用CPU版本")
        else:
            print_info("未检测到CUDA，安装CPU版本")
            pytorch_cmd = PYTORCH_URLS["cpu"]
        
        print_info(f"正在安装PyTorch...")
        print_info(f"命令: pip install {pytorch_cmd}")
        
        success, _ = run_command(f"{sys.executable} -m pip install {pytorch_cmd}", show_output=True)
        
        if success:
            print_status("PyTorch安装", True)
            return True
        else:
            print_status("PyTorch安装", False)
            print_info("请手动安装: pip install torch torchvision")
            return False

def detect_cuda_version():
    """检测系统CUDA版本"""
    # 尝试nvcc
    try:
        result = subprocess.run("nvcc --version", shell=True, capture_output=True, 
                               encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            import re
            match = re.search(r"release (\d+\.\d+)", result.stdout)
            if match:
                return float(match.group(1))
    except:
        pass
    
    # 尝试nvidia-smi
    try:
        result = subprocess.run("nvidia-smi", shell=True, capture_output=True,
                               encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            import re
            match = re.search(r"CUDA Version: (\d+\.\d+)", result.stdout)
            if match:
                return float(match.group(1))
    except:
        pass
    
    # 尝试从环境变量检测
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        import re
        match = re.search(r"v(\d+\.\d+)", cuda_path)
        if match:
            return float(match.group(1))
    
    return None

def check_package(package_name):
    """检查单个包是否已安装"""
    name = package_name.split(">=")[0].split("==")[0].replace("-", "_")
    try:
        __import__(name)
        return True
    except ImportError:
        return False

def get_package_version(package_name):
    """获取包版本"""
    name = package_name.split(">=")[0].split("==")[0]
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", name],
            capture_output=True, text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except:
        pass
    return None

def install_package(package, show_progress=True):
    """安装单个包"""
    if show_progress:
        print(f"    安装 {package}...", end=" ", flush=True)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if show_progress:
            print(f"{Colors.GREEN}成功{Colors.RESET}")
        return True
    except subprocess.CalledProcessError:
        if show_progress:
            print(f"{Colors.RED}失败{Colors.RESET}")
        return False

def check_and_install_packages(auto_install=False):
    """检查并安装依赖包"""
    print_header("4. 检查依赖包")
    
    missing = []
    installed = []
    
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">=")[0].split("==")[0]
        if check_package(name):
            version = get_package_version(name)
            print_status(name, True, f"v{version}" if version else "已安装")
            installed.append(name)
        else:
            print_status(name, False, "未安装")
            missing.append(pkg)
    
    if missing:
        print(f"\n  发现 {len(missing)} 个缺失的依赖包")
        
        if not auto_install:
            choice = input("  是否自动安装? (y/n): ").strip().lower()
            if choice != 'y':
                print_info("跳过安装")
                print_info(f"手动安装: pip install {' '.join(missing)}")
                return False
        
        print("\n  正在安装依赖包...")
        failed = []
        for pkg in missing:
            if not install_package(pkg):
                failed.append(pkg)
        
        if failed:
            print_warn(f"以下包安装失败: {', '.join(failed)}")
            return False
        else:
            print_info("所有依赖包安装成功")
            return True
    else:
        print_info("所有依赖包已安装")
        return True

def check_optional_packages():
    """检查可选依赖包"""
    print_header("5. 检查可选依赖包")
    
    missing = []
    for pkg, desc in OPTIONAL_PACKAGES:
        if check_package(pkg):
            version = get_package_version(pkg)
            print_status(f"{pkg} ({desc})", True, f"v{version}" if version else "已安装")
        else:
            print_status(f"{pkg} ({desc})", False, "未安装")
            missing.append((pkg, desc))
    
    if missing:
        print_info("可选包不影响基本功能，可按需安装")

def check_model():
    """检查模型文件"""
    print_header("6. 检查模型文件")
    
    script_dir = get_script_dir()
    model_paths = [
        ("best.pt (mAP84.2%)", os.path.join(script_dir, "backups", "backup_20251226_best_mAP84.2", "best.pt")),
        ("best.pt (训练输出)", os.path.join(script_dir, "runs", "cableinspect_real", "weights", "best.pt")),
        ("yolo11n.pt (预训练)", os.path.join(script_dir, "yolo11n.pt")),
    ]
    
    found_model = None
    for name, path in model_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print_status(name, True, f"{size_mb:.1f} MB")
            if found_model is None:
                found_model = path
        else:
            print_status(name, False, "未找到")
    
    if found_model:
        print_info(f"推荐模型: {found_model}")
        return True
    else:
        print_warn("未找到训练好的模型文件")
        print_info("可从GitHub Release下载或重新训练")
        return False

def check_dataset():
    """检查数据集"""
    print_header("7. 检查数据集")
    
    script_dir = get_script_dir()
    dataset_path = os.path.join(script_dir, "yolo_training", "cableinspect_yolo")
    
    if os.path.exists(dataset_path):
        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        if os.path.exists(yaml_path):
            print_status("数据集配置", True, "dataset.yaml")
        
        total_images = 0
        for split in ["train", "val", "test"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                images = os.path.join(split_path, "images")
                if os.path.exists(images):
                    count = len([f for f in os.listdir(images) if f.endswith(('.jpg', '.png', '.jpeg'))])
                    total_images += count
                    print_status(f"{split}集", True, f"{count} 张图片")
                else:
                    print_status(f"{split}集", False, "images目录不存在")
            else:
                print_status(f"{split}集", False, "目录不存在")
        
        if total_images > 0:
            print_info(f"数据集共 {total_images} 张图片")
        return True
    else:
        print_status("数据集目录", False, "未找到")
        print_warn("数据集需要单独下载 (约24GB)")
        print_info("数据集来源: CableInspect-AD (Mila/Hydro-Quebec)")
        print_info("许可证: CC BY-NC-SA 4.0 (仅限非商业用途)")
        return False

def show_quick_start():
    """显示快速启动指南"""
    print_header("快速启动指南")
    script_dir = get_script_dir()
    
    print(f"""
  {Colors.BOLD}1. 启动主系统 (推荐){Colors.RESET}
     python setup.py --run app
     或: python web_app/app.py
     然后访问: http://localhost:5000
     
  {Colors.BOLD}2. 启动Web演示{Colors.RESET}
     python setup.py --run web
     或: python demos/realtime_web_demo.py
     
  {Colors.BOLD}3. 启动API服务{Colors.RESET}
     python setup.py --run api
     或: python demos/api_server.py
     
  {Colors.BOLD}4. 实时检测演示{Colors.RESET}
     python setup.py --run demo
     或: python realtime_demo.py
     
  {Colors.BOLD}5. 训练模型 (需要数据集){Colors.RESET}
     cd yolo_training
     python train.py
     
  {Colors.BOLD}6. 查看3D设备设计{Colors.RESET}
     用浏览器打开: hardware/device_final.html
""")

def run_service(service_type):
    """启动指定服务"""
    script_dir = get_script_dir()
    os.chdir(script_dir)
    
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("       ACSR Vision - 启动服务")
    print(f"{'=' * 60}{Colors.RESET}")
    
    # 检查基本依赖
    print_info("检查环境...")
    try:
        import torch
        import ultralytics
        print_status("PyTorch", True, torch.__version__)
        print_status("Ultralytics", True, ultralytics.__version__)
    except ImportError as e:
        print_status("依赖检查", False, str(e))
        print_warn("请先运行 python setup.py 安装依赖")
        input("\n按回车键退出...")
        return
    
    # 查找模型
    model_paths = [
        os.path.join(script_dir, "backups", "backup_20251226_best_mAP84.2", "best.pt"),
        os.path.join(script_dir, "runs", "cableinspect_real", "weights", "best.pt"),
        os.path.join(script_dir, "yolo11n.pt"),
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path:
        print_status("模型文件", True, os.path.basename(model_path))
    else:
        print_status("模型文件", False, "未找到")
        print_warn("部分功能可能不可用")
    
    # 启动服务
    if service_type == "app":
        start_main_app(script_dir, model_path)
    elif service_type == "web":
        start_web_demo(script_dir, model_path)
    elif service_type == "api":
        start_api_server(script_dir, model_path)
    elif service_type == "demo":
        start_realtime_demo(script_dir, model_path)

def start_main_app(script_dir, model_path):
    """启动主系统"""
    app_path = os.path.join(script_dir, "web_app", "app.py")
    
    if not os.path.exists(app_path):
        print_warn(f"未找到 {app_path}")
        return
    
    print(f"\n{Colors.GREEN}启动 ACSR Vision 主系统...{Colors.RESET}")
    print_info("访问地址: http://localhost:5000")
    print_info("按 Ctrl+C 停止服务\n")
    
    try:
        subprocess.run([sys.executable, app_path], cwd=os.path.join(script_dir, "web_app"))
    except KeyboardInterrupt:
        print("\n服务已停止")

def start_web_demo(script_dir, model_path):
    """启动Web演示"""
    web_demo_path = os.path.join(script_dir, "demos", "realtime_web_demo.py")
    
    if not os.path.exists(web_demo_path):
        print_warn(f"未找到 {web_demo_path}")
        return
    
    print(f"\n{Colors.GREEN}启动 Web 演示服务...{Colors.RESET}")
    print_info("访问地址: http://localhost:5000")
    print_info("按 Ctrl+C 停止服务\n")
    
    try:
        subprocess.run([sys.executable, web_demo_path], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n服务已停止")

def start_api_server(script_dir, model_path):
    """启动API服务"""
    api_path = os.path.join(script_dir, "demos", "api_server.py")
    
    if not os.path.exists(api_path):
        print_warn(f"未找到 {api_path}")
        return
    
    print(f"\n{Colors.GREEN}启动 API 服务...{Colors.RESET}")
    print_info("API地址: http://localhost:5000/api")
    print_info("按 Ctrl+C 停止服务\n")
    
    try:
        subprocess.run([sys.executable, api_path], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n服务已停止")

def start_realtime_demo(script_dir, model_path):
    """启动实时检测演示"""
    demo_path = os.path.join(script_dir, "realtime_demo.py")
    
    if not os.path.exists(demo_path):
        print_warn(f"未找到 {demo_path}")
        return
    
    print(f"\n{Colors.GREEN}启动实时检测演示...{Colors.RESET}")
    print_info("按 Q 退出演示\n")
    
    try:
        subprocess.run([sys.executable, demo_path], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n演示已停止")

def create_venv():
    """创建虚拟环境并安装依赖"""
    script_dir = get_script_dir()
    venv_path = os.path.join(script_dir, "venv")
    
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("       ACSR Vision - 创建虚拟环境")
    print(f"{'=' * 60}{Colors.RESET}")
    
    # 检查是否已存在
    if os.path.exists(venv_path):
        print_warn(f"虚拟环境已存在: {venv_path}")
        choice = input("是否删除并重新创建? (y/n): ").strip().lower()
        if choice == 'y':
            print_info("删除旧虚拟环境...")
            shutil.rmtree(venv_path)
        else:
            print_info("跳过创建")
            show_venv_usage(venv_path)
            return
    
    # 创建虚拟环境
    print_info(f"创建虚拟环境: {venv_path}")
    success, _ = run_command(f"{sys.executable} -m venv \"{venv_path}\"", show_output=True)
    
    if not success:
        print_status("创建虚拟环境", False)
        return
    
    print_status("创建虚拟环境", True)
    
    # 获取虚拟环境中的pip路径
    if platform.system() == "Windows":
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # 升级pip
    print_info("升级pip...")
    run_command(f"\"{python_path}\" -m pip install --upgrade pip -q -i {PIP_MIRROR}")
    
    # 检测CUDA并安装PyTorch
    print_info("检测CUDA版本...")
    cuda_version = detect_cuda_version()
    
    if cuda_version:
        print_info(f"检测到CUDA {cuda_version}")
        if cuda_version >= 12.1:
            pytorch_cmd = PYTORCH_URLS["cuda121"]
        elif cuda_version >= 11.8:
            pytorch_cmd = PYTORCH_URLS["cuda118"]
        else:
            pytorch_cmd = PYTORCH_URLS["cpu"]
    else:
        print_info("未检测到CUDA，安装CPU版本")
        pytorch_cmd = PYTORCH_URLS["cpu"]
    
    print_info("安装PyTorch (可能需要几分钟)...")
    success, _ = run_command(f"\"{pip_path}\" install {pytorch_cmd}", show_output=True)
    if success:
        print_status("PyTorch安装", True)
    else:
        print_status("PyTorch安装", False)
        print_warn("PyTorch安装失败，尝试使用国内镜像...")
        # 尝试使用国内镜像安装CPU版本
        success, _ = run_command(f"\"{pip_path}\" install torch torchvision -i {PIP_MIRROR}", show_output=True)
        if success:
            print_status("PyTorch安装(镜像)", True)
    
    # 安装其他依赖
    print_info("安装其他依赖包 (使用清华镜像)...")
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split(">=")[0].split("==")[0]
        print(f"    安装 {name}...", end=" ", flush=True)
        success, _ = run_command(f"\"{pip_path}\" install \"{pkg}\" -q -i {PIP_MIRROR}")
        if success:
            print(f"{Colors.GREEN}成功{Colors.RESET}")
        else:
            print(f"{Colors.RED}失败{Colors.RESET}")
    
    print(f"\n{Colors.GREEN}虚拟环境创建完成!{Colors.RESET}")
    show_venv_usage(venv_path)

def show_venv_usage(venv_path):
    """显示虚拟环境使用说明"""
    script_dir = get_script_dir()
    
    if platform.system() == "Windows":
        activate_cmd = f".\\venv\\Scripts\\Activate.ps1"
        python_cmd = f".\\venv\\Scripts\\python.exe"
    else:
        activate_cmd = f"source venv/bin/activate"
        python_cmd = f"./venv/bin/python"
    
    print(f"""
{Colors.BOLD}虚拟环境使用说明:{Colors.RESET}

  {Colors.BOLD}1. 激活虚拟环境:{Colors.RESET}
     {activate_cmd}
     
  {Colors.BOLD}2. 直接使用虚拟环境Python:{Colors.RESET}
     & "{venv_path}\\Scripts\\python.exe" setup.py --run web
     
  {Colors.BOLD}3. 退出虚拟环境:{Colors.RESET}
     deactivate

{Colors.BOLD}快捷启动 (无需激活):{Colors.RESET}
     & "{python_cmd}" setup.py --run web
     & "{python_cmd}" setup.py --run api
     & "{python_cmd}" setup.py --run demo
""")

def show_summary(results):
    """显示检查总结"""
    print_header("环境检查总结")
    
    items = [
        ("Python环境", results.get("python", False)),
        ("pip包管理器", results.get("pip", False)),
        ("PyTorch", results.get("pytorch", False)),
        ("依赖包", results.get("packages", False)),
        ("模型文件", results.get("model", False)),
        ("数据集", results.get("dataset", False)),
    ]
    
    for name, status in items:
        print_status(name, status)
    
    all_ok = all([results.get("python"), results.get("pip"), 
                  results.get("pytorch"), results.get("packages")])
    
    if all_ok:
        print(f"\n  {Colors.GREEN}{Colors.BOLD}环境就绪，可以运行系统!{Colors.RESET}")
        return True
    else:
        print(f"\n  {Colors.YELLOW}请解决上述问题后重新运行此脚本{Colors.RESET}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ACSR Vision 环境检查与安装工具")
    parser.add_argument("--check", action="store_true", help="仅检查不安装")
    parser.add_argument("--install", action="store_true", help="强制安装所有依赖")
    parser.add_argument("--venv", action="store_true", help="创建虚拟环境")
    parser.add_argument("--run", choices=["web", "api", "demo", "app"], help="启动服务: app(主系统)/web(演示)/api/demo")
    args = parser.parse_args()
    
    # 如果指定了--venv，创建虚拟环境
    if args.venv:
        create_venv()
        return
    
    # 如果指定了--run，直接启动服务
    if args.run:
        run_service(args.run)
        return
    
    auto_install = args.install
    check_only = args.check
    
    # 标题
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("       ACSR Vision - 环境检查与安装工具")
    print("       钢芯铝绞线智能缺陷检测系统")
    print(f"{'=' * 60}{Colors.RESET}")
    print(f"\n  系统: {platform.system()} {platform.release()}")
    print(f"  目录: {get_script_dir()}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. 检查Python
    results["python"] = check_python()
    if not results["python"]:
        print_warn("Python版本不满足要求，请先安装正确版本")
        input("\n按回车键退出...")
        return
    
    # 2. 检查pip
    results["pip"] = check_pip()
    
    # 3. 检查PyTorch
    if check_only:
        try:
            import torch
            results["pytorch"] = True
            print_header("3. 检查 PyTorch")
            print_status("PyTorch已安装", True, torch.__version__)
        except ImportError:
            results["pytorch"] = False
            print_header("3. 检查 PyTorch")
            print_status("PyTorch已安装", False)
    else:
        results["pytorch"] = check_and_install_pytorch(auto_install)
    
    # 4. 检查依赖包
    if check_only:
        print_header("4. 检查依赖包")
        missing = []
        for pkg in REQUIRED_PACKAGES:
            name = pkg.split(">=")[0].split("==")[0]
            if check_package(name):
                version = get_package_version(name)
                print_status(name, True, f"v{version}" if version else "已安装")
            else:
                print_status(name, False, "未安装")
                missing.append(pkg)
        results["packages"] = len(missing) == 0
        if missing:
            print_info(f"缺失 {len(missing)} 个包: pip install {' '.join(missing)}")
    else:
        results["packages"] = check_and_install_packages(auto_install)
    
    # 5. 检查可选包
    check_optional_packages()
    
    # 6. 检查模型
    results["model"] = check_model()
    
    # 7. 检查数据集
    results["dataset"] = check_dataset()
    
    # 总结
    ready = show_summary(results)
    
    if ready:
        show_quick_start()
    
    input("\n按回车键退出...")

if __name__ == "__main__":
    main()
