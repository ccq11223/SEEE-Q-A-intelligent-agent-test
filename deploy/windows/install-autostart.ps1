# 在 Windows 服务器上运行一次：注册开机自启、放行防火墙、立即启动服务。
# 用法（管理员 PowerShell）：
#   powershell -ExecutionPolicy Bypass -File deploy\windows\install-autostart.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root

$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Host "找不到 python。请先安装 Python 3.11+，安装时勾选 Add python.exe to PATH。" -ForegroundColor Red
    exit 1
}
Write-Host "使用 Python：$python"

if (-not (Test-Path (Join-Path $root ".env"))) {
    Write-Host "找不到 .env，请先把本机的 .env 复制到 $root 下。" -ForegroundColor Red
    exit 1
}

# 放行 8000 端口（重复执行不会报错）
New-NetFirewallRule `
    -DisplayName "SEEQ Tutor 8000" `
    -Direction Inbound `
    -Protocol TCP `
    -LocalPort 8000 `
    -Action Allow `
    -ErrorAction SilentlyContinue | Out-Null

# 用绝对路径生成启动脚本：计划任务以 SYSTEM 运行时，用户级 PATH 里的 python 可能找不到
$generated = Join-Path $root "data\start-server.generated.cmd"
$lines = @(
    "@echo off",
    "cd /d `"$root`"",
    "set PYTHONIOENCODING=utf-8",
    "set PYTHONUTF8=1",
    "`"$python`" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> `"$root\data\server.log`" 2>&1"
)
Set-Content -Path $generated -Value $lines -Encoding ASCII

$action = New-ScheduledTaskAction -Execute $generated -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)

Register-ScheduledTask `
    -TaskName "SEEQ-Tutor" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null

Start-ScheduledTask -TaskName "SEEQ-Tutor"
Start-Sleep -Seconds 6

Write-Host ""
Write-Host "已注册开机自启任务：SEEQ-Tutor" -ForegroundColor Green
Write-Host "本机访问： http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "外网访问： http://<服务器公网IP>:8000" -ForegroundColor Green
Write-Host ""
Write-Host "如果外网打不开，检查云控制台的安全组是否放行了 TCP 8000。" -ForegroundColor Yellow
Write-Host "日志： $root\data\server.log"
