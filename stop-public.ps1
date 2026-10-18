# 停止「启动公网服务」拉起的本地服务和隧道。

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root "data\service.pids"

try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
} catch {}

if (-not (Test-Path $pidFile)) {
    Write-Host "没有记录到正在运行的服务。" -ForegroundColor Yellow
    exit 0
}

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
        Write-Host "已停止 $($parts[0])（PID $procId）"
    }
}

Remove-Item $pidFile -ErrorAction SilentlyContinue
Write-Host "服务已停止。" -ForegroundColor Green
