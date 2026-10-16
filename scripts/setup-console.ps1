# 把当前 PowerShell 窗口切成全程 UTF-8，消除中文乱码。
#
# 用法（注意开头那个点，表示「在当前会话里生效」，不能省略）：
#     cd 'D:\学习资料\学业指导中心事务\2023级学业指导中心\agent-code'
#     . .\scripts\setup-console.ps1
#
# 为什么需要这么麻烦：
#   Windows 控制台默认用 GBK（代码页 936），而 PowerShell 5.1 把
#   Python 的输出接进管道（| Select-String）时，会用控制台的代码页去解码。
#   两边不一致时，中文就变成乱码怪字。
#   把控制台、PowerShell、Python 三方都设成 UTF-8，问题就不存在了。
#
# 如果你用的是 PowerShell 7（pwsh），它默认就是 UTF-8，不需要跑这个脚本。

$ErrorActionPreference = 'SilentlyContinue'

try { chcp 65001 > $null } catch { Write-Host '无法切换代码页，跳过。' }

$utf8 = New-Object System.Text.UTF8Encoding $false
try {
    [Console]::OutputEncoding = $utf8
    [Console]::InputEncoding = $utf8
} catch {
    Write-Host '无法设置控制台编码（可能不在真实控制台里），跳过。'
}

$OutputEncoding = $utf8          # PowerShell 往管道写东西时用 UTF-8
$env:PYTHONIOENCODING = 'utf-8'  # 让 Python 也输出 UTF-8，覆盖程序的自动判断

Write-Host ''
Write-Host '已切换为 UTF-8：'
Write-Host ('  控制台代码页      : ' + (chcp))
Write-Host ('  [Console]::Output : ' + [Console]::OutputEncoding.WebName)
Write-Host ('  PYTHONIOENCODING  : ' + $env:PYTHONIOENCODING)
Write-Host ''
Write-Host '验证一下，下面这行应该显示「中文测试 OK」：'
python -c "import sys; sys.stdout.buffer.write('中文测试 OK\n'.encode('utf-8'))"
Write-Host ''
Write-Host '显示正常就可以继续跑 pipeline 命令了。'
