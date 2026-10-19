# 部署到 Windows 云服务器（有桌面、不用 Docker）

如果你的服务器登进去是一个 Windows 桌面（开始菜单、任务栏、此电脑），
按这份做。全程不需要 Docker。

---

## 一、确认系统

在服务器上按 `Win + R`，输入 `winver`，看到 `Windows Server 2019/2022` 就对了。

---

## 二、安装 Python

1. 在服务器里打开浏览器，访问 <https://www.python.org/downloads/windows/>；
2. 下载 **Python 3.12** 的 Windows installer（64-bit）；
3. 安装时务必勾选 **Add python.exe to PATH**；
4. 装完打开 PowerShell，输入：

```powershell
python --version
```

显示 `Python 3.12.x` 即可。

> 服务器下载慢的话，可以在你电脑上下好安装包，通过远程桌面的磁盘映射复制过去。

---

## 三、把项目复制到服务器

推荐用远程桌面的磁盘映射：

1. 在你电脑上打开「远程桌面连接」，点「显示选项」→「本地资源」→「详细信息」→ 勾选你的 C 盘或 D 盘；
2. 重新连接服务器；
3. 在服务器的「此电脑」里会多出一个你电脑的磁盘，把整个 `agent-code` 文件夹复制到服务器，例如 `C:\seeq`。

复制完成后，服务器上应该有：

```
C:\seeq\app
C:\seeq\pipeline
C:\seeq\config
C:\seeq\data\work
C:\seeq\.env
C:\seeq\requirements-app.txt
```

> `.env` 一定要复制过去，里面有数据库和模型密钥。

---

## 四、安装依赖

在服务器上打开 PowerShell（开始菜单搜 PowerShell，右键「以管理员身份运行」）：

```powershell
cd C:\seeq
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-app.txt
```

等它装完，最后出现 `Successfully installed ...` 即可。

---

## 五、先手动试跑

```powershell
cd C:\seeq
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

然后在服务器上打开浏览器访问 <http://127.0.0.1:8000>，能看到答题页面就说明服务正常。

确认没问题后，在 PowerShell 里按 `Ctrl+C` 停掉。

---

## 六、设置开机自启 + 放行防火墙

在服务器上，用管理员 PowerShell 执行：

```powershell
cd C:\seeq
powershell -ExecutionPolicy Bypass -File deploy\windows\install-autostart.ps1
```

它会做三件事：

1. 放行 Windows 防火墙的 8000 端口；
2. 注册一个开机自启任务 `SEEQ-Tutor`，进程挂了会自动重启；
3. 立刻启动服务。

---

## 七、在云控制台放行端口

登录云厂商控制台 → 找到这台服务器 → 「安全组 / 防火墙」→ 添加入站规则：

| 协议 | 端口 | 来源 |
| --- | --- | --- |
| TCP | 8000 | 0.0.0.0/0 |

然后在手机或别的电脑上打开：

```
http://服务器公网IP:8000
```

这个链接就是固定的，你的电脑关机也不影响。

---

## 八、日常维护

查看服务是否在跑：

```powershell
Get-ScheduledTask -TaskName "SEEQ-Tutor"
```

看日志：

```powershell
Get-Content C:\seeq\data\server.log -Tail 100
```

重启服务：

```powershell
Stop-ScheduledTask -TaskName "SEEQ-Tutor"
Start-ScheduledTask -TaskName "SEEQ-Tutor"
```

更新代码后，替换 `app`、`pipeline`、`config`、`data\work`，再重启任务即可。

---

## 九、常见问题

**`python` 不是内部或外部命令**
安装 Python 时没勾 PATH。重新运行安装包，选 Modify，勾上 Add to PATH。

**外网打不开，本机 127.0.0.1 能开**
九成是云控制台的安全组没放行 8000，或者 Windows 防火墙没放行。
重新运行 `install-autostart.ps1`，并检查安全组。

**服务启动后立刻退出**
看 `C:\seeq\data\server.log` 的最后几十行，通常是 `.env` 没放对位置或密钥不对。

**想用 HTTPS**
Windows 上最简单的是装 Caddy：
<https://caddyserver.com/download>，然后把域名反代到 `127.0.0.1:8000`。
没有域名的话，先用 `http://IP:8000` 也能正常答疑，只是「复制诊断报告」按钮不可用。
