# 诊断并修好 PowerShell 里的中文乱码。
#
# 用法：
#     cd 'D:\学习资料\学业指导中心事务\2023级学业指导中心\agent-code'
#     . .\scripts\setup-console.ps1            # 诊断 + 修当前窗口
#     . .\scripts\setup-console.ps1 -Persist   # 顺便写进 PowerShell 配置文件
#
# 开头那个点不能省（点表示「在当前会话里生效」）。
#
# 这个脚本会**先记录修改前的原始状态**，再动手修，最后把「改前 / 改后」两份
# 信息写进 encoding-report.txt。排查乱码最需要的就是改前那份值——
# 而剪贴板里的中文本身可能就是乱的，所以报告写文件、不走剪贴板。

[CmdletBinding()]
param(
    [switch]$Persist
)

$ErrorActionPreference = 'SilentlyContinue'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$report = New-Object System.Collections.Generic.List[string]

function Add-Line($text) {
    $report.Add([string]$text)
    Write-Host $text
}

function Get-PythonProbe {
    # 这一次调用很关键：既看 Python 的默认编码，也看我们自己的
    # configure_stdio() 会把它改成什么——乱码往往就出在这两者之间。
    $probePath = Join-Path $env:TEMP 'seeq-pyprobe.py'
@'
import os, sys, locale
# Python 会把「脚本所在目录」放在 sys.path[0]，而探针脚本在 %TEMP%，
# 所以必须手动把项目根目录加进来，否则 import pipeline 会失败、
# 也就测不出 configure_stdio() 到底把编码改成了什么。
sys.path.insert(0, os.getcwd())
print("  Python 版本          :", sys.version.split()[0])
print("  默认 stdout.encoding :", sys.stdout.encoding)
print("  stdout.isatty()      :", sys.stdout.isatty())
print("  locale 首选编码      :", locale.getpreferredencoding(False))
print("  PYTHONIOENCODING     :", os.environ.get("PYTHONIOENCODING"))
print("  PYTHONUTF8           :", os.environ.get("PYTHONUTF8"))
try:
    from pipeline.cli import configure_stdio
    configure_stdio()
    print("  configure_stdio 之后 :", sys.stdout.encoding)
except Exception as exc:
    print("  configure_stdio 调用失败:", exc)
print("  中文样本             : 电路理论 电压 u(t) 阻抗 Z = R + jX")
'@ | Out-File -FilePath $probePath -Encoding utf8
    & python $probePath 2>&1
}

# ================= 1. 修改前的原始状态 =================
Add-Line '################ 修改前（原始状态） ################'
Add-Line ('PowerShell 版本    : ' + $PSVersionTable.PSVersion.ToString())
Add-Line ('PSEdition          : ' + $PSVersionTable.PSEdition)
Add-Line ('控制台代码页       : ' + (chcp))
Add-Line ('[Console]::Output  : ' + [Console]::OutputEncoding.WebName)
Add-Line ('$OutputEncoding    : ' + $OutputEncoding.WebName)
Add-Line ('终端标识 WT_SESSION: ' + $(if ($env:WT_SESSION) { '有（Windows Terminal）' } else { '无' }))
Add-Line ('终端标识 TERM_PROG : ' + $(if ($env:TERM_PROGRAM) { $env:TERM_PROGRAM } else { '无' }))
Add-Line 'Python：'
Push-Location $projectRoot
foreach ($line in (Get-PythonProbe)) { Add-Line $line }
Pop-Location

# ================= 2. 应用修复 =================
chcp 65001 > $null
$utf8 = New-Object System.Text.UTF8Encoding $false
try {
    [Console]::OutputEncoding = $utf8
    [Console]::InputEncoding = $utf8
} catch {
    Add-Line '（无法设置控制台编码，可能不在真实控制台里）'
}
$OutputEncoding = $utf8
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# ================= 3. 修改后的状态 =================
Add-Line ''
Add-Line '################ 修改后 ################'
Add-Line ('控制台代码页       : ' + (chcp))
Add-Line ('[Console]::Output  : ' + [Console]::OutputEncoding.WebName)
Add-Line ('$OutputEncoding    : ' + $OutputEncoding.WebName)
Add-Line ('PYTHONIOENCODING   : ' + $env:PYTHONIOENCODING)
Add-Line ('PYTHONUTF8         : ' + $env:PYTHONUTF8)
Add-Line 'Python：'
Push-Location $projectRoot
foreach ($line in (Get-PythonProbe)) { Add-Line $line }

Add-Line ''
Add-Line '下面这几行由 pipeline 自己打印，中文应该正常显示：'
& python -m pipeline manifest 2>&1 | Select-Object -First 3 | ForEach-Object { Add-Line ('  ' + $_) }
Pop-Location

# ================= 4. 写报告（不走剪贴板） =================
$reportPath = Join-Path $projectRoot 'encoding-report.txt'
$report | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host ''
Write-Host ('诊断报告已写入：' + $reportPath)
Write-Host '如果上面「中文样本」那行正常，说明当前窗口已修好；否则请把这个 txt 发我。'

# ================= 5. 可选：写进配置文件，以后自动生效 =================
if ($Persist) {
    $profileHasBlock = (Test-Path $PROFILE) -and ((Get-Content -LiteralPath $PROFILE -Raw) -match 'seeq-utf8')
    if ($profileHasBlock) {
        Write-Host ''
        Write-Host '配置文件里已经有 UTF-8 块，跳过。'
    } else {
        $block = @'

# >>> seeq-utf8 >>> 由 scripts/setup-console.ps1 -Persist 写入，删掉这一段即可撤销
chcp 65001 > $null
try {
    [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
    [Console]::InputEncoding = New-Object System.Text.UTF8Encoding $false
} catch { }
$OutputEncoding = New-Object System.Text.UTF8Encoding $false
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
# <<< seeq-utf8 <<<
'@
        $profileDir = Split-Path -Parent $PROFILE
        if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Path $profileDir -Force | Out-Null }
        if (-not (Test-Path $PROFILE)) { New-Item -ItemType File -Path $PROFILE -Force | Out-Null }
        Add-Content -LiteralPath $PROFILE -Value $block -Encoding UTF8
        Write-Host ''
        Write-Host ('已写入配置文件：' + $PROFILE)
        Write-Host '以后新开的 PowerShell 窗口会自动设成 UTF-8。撤销方式：删掉 seeq-utf8 那一段。'
    }
}
