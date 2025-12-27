@echo off
echo ========================================
echo 钢芯铝绞线智能质检系统
echo ========================================
echo.
cd /d "%~dp0"
echo 正在启动服务器...
echo 请在浏览器访问: http://localhost:5000
echo.
python app.py
pause
