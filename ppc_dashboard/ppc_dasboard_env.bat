@echo off
chcp 65001 >nul
title SmartPet AI - 智能环境配置向导

:: 1. 自动触发 UAC 管理员提权检查
>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"
if '%errorlevel%' NEQ '0' (
    echo [INFO] 正在请求管理员权限，请在弹出的窗口中点击“是”...
    goto UACPrompt
) else ( goto gotAdmin )

:UACPrompt
    echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
    echo UAC.ShellExecute "cmd.exe", "/c ""%~s0""", "", "runas", 1 >> "%temp%\getadmin.vbs"
    "%temp%\getadmin.vbs"
    del "%temp%\getadmin.vbs"
    exit /B

:gotAdmin
:: 2. 提权成功后，切回真实的专案工作目录 (绕过 System32 默认路径)
cd /d "%~dp0"

echo ==========================================
echo [INFO] 正在加载智能环境安装脚本并解除安全限制...
echo ==========================================

:: 3. 核心机制：呼叫 PowerShell 提取自身底部嵌的代码并绕过执行策略 (Bypass) 运行
powershell -NoProfile -ExecutionPolicy Bypass -Command "$script = Get-Content '%~f0' -Raw; $code = $script -replace '(?s)^.*<#POWERSHELL_START#>\r?\n', ''; Invoke-Expression $code"
exit /b

<#POWERSHELL_START#>
# ====================================================================
# 👇 以下为真正的 PowerShell 智能分流与安装逻辑 👇
# ====================================================================

$CurrentDir = Get-Location
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host " 🛠️ SmartPet AI 智能环境配置向导" -ForegroundColor Cyan
Write-Host " 工作目录: $CurrentDir" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

# 硬件嗅探 (Hardware Sniffer)
$cpu = Get-CimInstance Win32_Processor
$cpuName = $cpu.Name
$cpuThreads = $cpu.NumberOfLogicalProcessors
Write-Host ">>> 💻 侦测到处理器: $cpuName ($cpuThreads 线程)" -ForegroundColor Cyan

# 判定是否为老旧架构 (3代及以下，或核心数 <= 4)
$isOldCPU = ($cpuName -match "i[357]-[23]\d{3}") -or ($cpuThreads -le 4)

# 下载并静默安装 Python 3.11
$pythonInstaller = "python-3.11.9-amd64.exe"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host ">>> 正在下载 Python 3.11.9..."
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/$pythonInstaller" -OutFile $pythonInstaller
    Write-Host ">>> 正在静默安装 Python (请耐心稍候，约需1-2分钟)..."
    Start-Process -FilePath ".\$pythonInstaller" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" -Wait
    Remove-Item $pythonInstaller
    Write-Host "✅ Python 3.11 安装完成！" -ForegroundColor Green
} else {
    Write-Host "✅ 系统已检测到 Python，跳过安装。" -ForegroundColor Green
}

# 下载并静默安装 Node.js
$nodeVersion = "22.14.0"
$nodeInstaller = "node-v$nodeVersion-x64.msi"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host ">>> 正在下载 Node.js v$nodeVersion..."
    Invoke-WebRequest -Uri "https://nodejs.org/dist/v$nodeVersion/$nodeInstaller" -OutFile $nodeInstaller
    Write-Host ">>> 正在静默安装 Node.js..."
    Start-Process -FilePath "msiexec.exe" -ArgumentList "/i $nodeInstaller /quiet /norestart" -Wait
    Remove-Item $nodeInstaller
    Write-Host "✅ Node.js 安装完成！" -ForegroundColor Green
} else {
    Write-Host "✅ 系统已检测到 Node.js，跳过安装。" -ForegroundColor Green
}

# 刷新全局环境变量
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host ">>> 配置 NPM 淘宝镜像并升级..."
npm config set registry https://registry.npmmirror.com/ >$null 2>&1
npm install -g npm@10.9.4 >$null 2>&1

Write-Host ">>> 正在升级 pip..."
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >$null 2>&1

Write-Host ">>> 安装基础框架 (OpenCV, Flask 等)..."
python -m pip install opencv-python numpy==1.26.4 flask flask-cors psutil waitress pyyaml -i https://pypi.tuna.tsinghua.edu.cn/simple

if ($isOldCPU) {
    Write-Host "⚠️ [硬件降级模式] 侦测到早期架构 CPU，开始安装【高兼容性降级包】(跳过 AVX2 指令集)..." -ForegroundColor Yellow
    
    Write-Host " -> 安装 PyTorch 1.13.1 (兼容版)"
    python -m pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
    
    Write-Host " -> 安装 OpenVINO 2023.3.0 与 ONNXRuntime 1.14.1 (兼容版)"
    python -m pip install openvino==2023.3.0 nncf onnxruntime==1.14.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    
    Write-Host " -> 安装 Ultralytics (忽略 torch 依赖，防止覆盖旧版)"
    python -m pip install ultralytics --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple
} else {
    Write-Host "🚀 [性能满血模式] 侦测到现代架构 CPU，开始安装【最新性能包】..." -ForegroundColor Green
    
    if (Test-Path "requirements.txt") {
        Write-Host " -> 发现 requirements.txt，优先按照清单安装依赖..."
        python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
    } else {
        Write-Host " -> 安装最新版 PyTorch CUDA 加速版..."
        python -m pip install torch==2.5.1+cu121 torchvision --extra-index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
    }
    
    Write-Host " -> 安装最新版 OpenVINO, NNCF 与 ONNXRuntime..."
    python -m pip install openvino nncf onnxruntime ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
}

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🎉 基础环境全自动配置完毕！您可以直接启动专案了。" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Cyan

Write-Host "按任意键退出..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null