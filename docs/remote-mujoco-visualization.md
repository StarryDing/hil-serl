# 远程可视化 MuJoCo（TurboVNC + VirtualGL）

从 Mac 远程连接 Ubuntu 桌面服务器，在独立虚拟桌面中运行 MuJoCo 交互式 viewer，不影响服务器本地用户的图形会话。

## 环境信息

| 项目 | 详情 |
|---|---|
| 服务器 | Ubuntu 24.04 LTS (Starry-Psibot) |
| GPU | NVIDIA GeForce RTX 5090 x2 |
| 驱动 | 590.48.01 |
| 客户端 | macOS |
| 远程桌面 | Xfce（轻量，VNC 兼容性好） |

## 一、安装（服务器端，一次性操作）

### 1.1 安装 VirtualGL

```bash
wget https://github.com/VirtualGL/virtualgl/releases/download/3.1.4/virtualgl_3.1.4_amd64.deb
sudo dpkg -i virtualgl_3.1.4_amd64.deb
sudo apt-get install -f -y
```

### 1.2 安装 TurboVNC

```bash
wget https://github.com/TurboVNC/turbovnc/releases/download/3.3/turbovnc_3.3_amd64.deb
sudo dpkg -i turbovnc_3.3_amd64.deb
sudo apt-get install -f -y
```

### 1.3 安装 Xfce 桌面环境

比 GNOME 更轻量、对 VNC 兼容性更好，不会受 compositing 影响。

```bash
sudo apt-get install -y xfce4 xfce4-terminal
```

### 1.4 加入 GPU 设备组

```bash
sudo usermod -aG video,render $USER
```

需要**重新登录 SSH**（或 `newgrp video`）使新组生效，可通过 `groups` 命令验证。

### 1.5 配置 TurboVNC 启动脚本

```bash
mkdir -p ~/.vnc
cat > ~/.vnc/xstartup.turbovnc << 'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XDG_SESSION_TYPE=x11
exec startxfce4
EOF
chmod +x ~/.vnc/xstartup.turbovnc
```

### 1.6 设置 VNC 密码

```bash
/opt/TurboVNC/bin/vncpasswd
```

## 二、安装（Mac 客户端，一次性操作）

```bash
brew install --cask turbovnc-viewer
```

也可以使用 macOS 自带的"屏幕共享"（`vnc://localhost:端口`）。

## 三、日常使用

### 3.1 服务器端：启动 VNC 会话

```bash
/opt/TurboVNC/bin/vncserver -geometry 1920x1080 -xstartup ~/.vnc/xstartup.turbovnc
```

记住输出中的显示号（如 `:2`），对应端口 = 5900 + 显示号（`:2` → `5902`）。

### 3.2 Mac 端：建立 SSH 隧道

```bash
# 显示号 :2 → 端口 5902
ssh -L 5902:localhost:5902 <用户名>@<服务器IP或别名>
```

保持此终端不要关闭。

### 3.3 Mac 端：连接 VNC

打开 TurboVNC Viewer，连接地址：

```
localhost:5902
```

输入 VNC 密码即可进入 Xfce 桌面。

### 3.4 在 VNC 桌面中运行 MuJoCo

在 Xfce 桌面中打开终端（右键桌面 → Terminal），使用 `vglrun` 启动程序：

```bash
conda activate serl

# EGL 后端（推荐，无需额外配置）
VGL_DISPLAY=egl vglrun python your_script.py

# 示例
VGL_DISPLAY=egl vglrun python /home/starry/Projects/serl/franka_sim/franka_sim/test/test_gym_env_human.py
```

### 3.5 验证 GPU 渲染

```bash
VGL_DISPLAY=egl vglrun glxinfo | grep "OpenGL renderer"
# 应输出: NVIDIA GeForce RTX 5090/PCIe/SSE2
```

### 3.6 管理 VNC 会话

```bash
# 查看运行中的会话
/opt/TurboVNC/bin/vncserver -list

# 关闭指定会话
/opt/TurboVNC/bin/vncserver -kill :2

# 修改 VNC 密码
/opt/TurboVNC/bin/vncpasswd
```

## 四、备用方案：GLX 标准模式

如果 EGL 模式下 MuJoCo 渲染异常（画面闪烁、崩溃等），可切换到 GLX 模式。GLX 兼容性更强，但需要**一次性重启显示管理器**（会短暂中断本地用户的桌面会话，约 30 秒）。

### 4.1 配置 VirtualGL（需与本地用户协调）

```bash
# 让本地用户保存工作
sudo systemctl stop gdm3

sudo /opt/VirtualGL/bin/vglserver_config
# 选 1) Configure server for use with VirtualGL (GLX + EGL back ends)
# Restrict 3D X server access to vglusers group?           → Yes
# Restrict framebuffer device access to vglusers group?     → Yes
# Disable XTEST extension?                                  → Yes

# 将自己加入 vglusers 组
sudo usermod -aG vglusers $USER

# 重启显示管理器
sudo systemctl start gdm3
```

本地用户此时可重新登录。此操作只需执行一次。

### 4.2 GLX 模式的日常使用

启动 VNC 会话的命令不变，运行程序时不再需要 `VGL_DISPLAY=egl`：

```bash
# VNC 桌面中
conda activate serl
vglrun python your_script.py
```

## 五、架构说明

```
Mac (TurboVNC Viewer)
  │
  │  VNC 协议（通过 SSH 隧道加密）
  │
  ▼
TurboVNC Server (:2)          本地用户桌面 (:0)
  │  独立虚拟 X Display          │  物理显示器
  │  Xfce 桌面                   │  GNOME 桌面
  │                               │
  └───────── 共享 GPU ──────────┘
         NVIDIA RTX 5090
```

- 两个桌面会话完全隔离，互不可见、互不干扰
- 仅共享 GPU 硬件资源（MuJoCo 渲染占用极小，约 100-200MB 显存）

## 六、常见问题

### VNC 会话启动后立刻退出

检查日志 `~/.vnc/Starry-Psibot:*.log`。常见原因：

- **GNOME 被启动而非 Xfce**：确保使用 `-xstartup ~/.vnc/xstartup.turbovnc`，不要加 `-wm` 或 `-vgl` 参数
- **GPU 权限不足**：日志中出现 `Permission denied` → 执行 `sudo usermod -aG video,render $USER` 后重新登录

### SSH 隧道 "Connection refused"

端口号与显示号不匹配。显示号 `:N` 对应端口 `5900 + N`。

### `vglrun` 报错

尝试指定 EGL：`VGL_DISPLAY=egl vglrun ...`。若仍然失败，切换到 GLX 标准模式（第四节）。
