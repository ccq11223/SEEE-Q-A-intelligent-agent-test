# 双击「启动公网服务.cmd」会运行这个脚本。
# 它做三件事：启动本地服务 → 开一条 Cloudflare 临时隧道 → 把公网链接复制到剪贴板。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    $OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

$host.UI.RawUI.WindowTitle = "电路理论答疑 · 公网服务"

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

function Stop-Recorded {
    $pidFile = Join-Path $root "data\service.pids"
    if (-not (Test-Path $pidFile)) { return }
    Get-Content $pidFile -ErrorAction SilentlyContinue | ForEach-Object {
        $parts = $_ -split "=", 2
        if ($parts.Count -ne 2) { return }
        $procId = 0
        if (-not [int]::TryParse($parts[1], [ref]$procId)) { return }
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        if ($null -eq $proc) { return }
        $name = $proc.ProcessName.ToLower()
        if ($name -like "*python*" -or $name -like "*cloudflared*") {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

function Test-App {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/health" -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Path (Join-Path $root ".env"))) {
    Write-Host "找不到 .env，请先把 .env.example 复制成 .env 并填好密钥。" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

# 清理上一次可能残留的进程
Stop-Recorded

$python = $null
foreach ($candidate in @(
    (Join-Path $root ".venv\Scripts\python.exe"),
    "D:\Python\python.exe"
)) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
    $found = Get-Command python -ErrorAction SilentlyContinue
    if ($found) { $python = $found.Source }
}
if (-not $python) {
    Write-Host "找不到 Python，请先安装 Python 3.11+。" -ForegroundColor Red
    Read-Host "按回车退出"
    exit 1
}

$tools = Join-Path $root "tools"
$cloudflared = Join-Path $tools "cloudflared.exe"
if (-not (Test-Path $cloudflared)) {
    Write-Step "首次运行，正在下载 cloudflared（约 55MB，只需一次）"
    New-Item -ItemType Directory -Force -Path $tools | Out-Null
    Invoke-WebRequest `
        -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" `
        -OutFile $cloudflared `
        -TimeoutSec 300
}

$dataDir = Join-Path $root "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
$appOut = Join-Path $dataDir "app.out.log"
$appErr = Join-Path $dataDir "app.err.log"
$tunnelOut = Join-Path $dataDir "tunnel.out.log"
$tunnelErr = Join-Path $dataDir "tunnel.err.log"
$pidFile = Join-Path $dataDir "service.pids"

$startedApp = $false
$app = $null
$tunnel = $null

try {
    if (Test-App) {
        Write-Step "本地服务已经在运行，直接开公网隧道"
    } else {
        Write-Step "启动本地答疑服务"
        $app = Start-Process -FilePath $python `
            -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000") `
            -WorkingDirectory $root `
            -WindowStyle Hidden `
            -PassThru `
            -RedirectStandardOutput $appOut `
            -RedirectStandardError $appErr
        $startedApp = $true

        $ready = $false
        for ($i = 0; $i -lt 40; $i++) {
            Start-Sleep -Milliseconds 500
            if (Test-App) { $ready = $true; break }
            if ($app.HasExited) { break }
        }
        if (-not $ready) {
            Write-Host "本地服务没有起来，最近的错误信息：" -ForegroundColor Red
            if (Test-Path $appErr) {
                Get-Content $appErr -Tail 20 | ForEach-Object { Write-Host "  $_" }
            }
            Read-Host "按回车退出"
            return
        }
        Write-Host "本地服务已就绪：http://127.0.0.1:8000"
    }

    Write-Step "正在建立公网隧道（第一次可能要等 10 秒左右）"
    Remove-Item $tunnelOut, $tunnelErr -ErrorAction SilentlyContinue
    $tunnel = Start-Process -FilePath $cloudflared `
        -ArgumentList @("tunnel", "--url", "http://127.0.0.1:8000", "--no-autoupdate") `
        -WorkingDirectory $root `
        -WindowStyle Hidden `
        -PassThru `
        -RedirectStandardOutput $tunnelOut `
        -RedirectStandardError $tunnelErr

    $url = $null
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Milliseconds 500
        foreach ($file in @($tunnelErr, $tunnelOut)) {
            if (-not (Test-Path $file)) { continue }
            $match = Select-String -Path $file -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -AllMatches -ErrorAction SilentlyContinue
            if ($match -and $match.Matches.Count -gt 0) {
                $url = $match.Matches[0].Value
                break
            }
        }
        if ($url) { break }
        if ($tunnel.HasExited) { break }
    }

    if (-not $url) {
        Write-Host "隧道没有起来，最近的错误信息：" -ForegroundColor Red
        if (Test-Path $tunnelErr) {
            Get-Content $tunnelErr -Tail 20 | ForEach-Object { Write-Host "  $_" }
        }
        Read-Host "按回车退出"
        return
    }

    $pidLines = @()
    if ($app) { $pidLines += "app=$($app.Id)" }
    if ($tunnel) { $pidLines += "tunnel=$($tunnel.Id)" }
    $pidLines | Set-Content $pidFile -Encoding ASCII

    try { Set-Clipboard -Value $url } catch {}

    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host "  公网链接（已复制到剪贴板）：" -ForegroundColor Green
    Write-Host "  $url" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  直接把这个链接发给同学，手机/电脑都能打开。" -ForegroundColor Green
    Write-Host "  这个窗口不要关；关掉服务就停了，链接也会失效。" -ForegroundColor DarkYellow
    Write-Host "==================================================" -ForegroundColor Green
    Write-Host ""

    Read-Host "按回车停止服务"
} finally {
    Write-Host ""
    Write-Host "正在停止服务..." -ForegroundColor DarkGray
    if ($tunnel -and -not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id -Force -ErrorAction SilentlyContinue
    }
    if ($startedApp -and $app -and -not $app.HasExited) {
        Stop-Process -Id $app.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}
