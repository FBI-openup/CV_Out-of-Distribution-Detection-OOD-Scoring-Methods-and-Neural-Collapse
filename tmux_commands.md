# Tmux 命令速查表

## 常用命令（必备）

### 基础操作
```bash
# 创建新会话
tmux new -s session_name

# 列出所有会话
tmux ls

# 连接到已有会话
tmux attach -t session_name

# 删除会话
tmux kill-session -t session_name

# 分离会话（在会话内按）
Ctrl+b 然后 d
```

### 典型使用流程
```bash
# 1. 创建训练会话
tmux new -s ood_training

# 2. 激活环境并进入目录
source ~/Aster-WorkSpace/CV/OOD_venv/bin/activate
cd ~/Aster-WorkSpace/CV/OOD-Analysis

# 3. 转换notebook为Python脚本（首次需要）
jupyter nbconvert --to python Resnet_OOD.ipynb

# 4. 运行训练（输出到日志文件）
python Resnet_OOD.py 2>&1 | tee training.log

# 5. 分离会话（训练继续运行）
# 按 Ctrl+b，然后按 d

# 6. 断开SSH（训练不会中断）
exit

# 7. 稍后重新连接查看进度
ssh user@server
tmux attach -t ood_training
```

---

## 快捷键速查

**前缀键**: 所有快捷键前先按 `Ctrl+b`

### 会话管理
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b` `d` | 分离会话 |
| `Ctrl+b` `$` | 重命名会话 |

### 窗口管理
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b` `c` | 创建新窗口 |
| `Ctrl+b` `n` | 下一个窗口 |
| `Ctrl+b` `p` | 上一个窗口 |
| `Ctrl+b` `0-9` | 切换到指定窗口 |
| `Ctrl+b` `&` | 关闭当前窗口 |

### 窗格分屏
| 快捷键 | 功能 |
|--------|------|
| `Ctrl+b` `"` | 水平分屏 |
| `Ctrl+b` `%` | 垂直分屏 |
| `Ctrl+b` `方向键` | 切换窗格 |
| `Ctrl+b` `x` | 关闭当前窗格 |

---

## 实用场景

### 监控训练进度
```bash
# 在tmux会话中分屏
# 1. 按 Ctrl+b 然后 " (水平分屏)
# 2. 上半部分运行训练
# 3. 下半部分监控GPU
watch -n 1 nvidia-smi

# 切换窗格: Ctrl+b 然后方向键
```

### 多窗口工作
```bash
# Ctrl+b c  - 创建新窗口
# 窗口0: 运行训练
# 窗口1: 监控系统
# 窗口2: 查看日志

# Ctrl+b 0/1/2 - 快速切换窗口
```

---

## 补充说明

### 什么是tmux
Tmux (Terminal Multiplexer) 是终端多路复用器，主要作用：
- 保持SSH会话持久化
- 断开连接后程序继续运行
- 多窗口/分屏管理

### 概念层级
```
Session (会话)
  └── Window (窗口)
       └── Pane (窗格)
```

### 常见问题

**Q: 关闭SSH窗口后训练还在运行吗？**  
A: 如果按了 `Ctrl+b d` 分离会话，训练会继续运行。

**Q: 如何确认会话还存在？**  
A: 运行 `tmux ls` 查看所有活动会话。

**Q: 训练完成后会话会自动消失吗？**  
A: 不会，需要手动 `tmux kill-session -t session_name` 删除。

**Q: 忘记会话名称怎么办？**  
A: `tmux ls` 列出所有会话，`tmux attach` 连接最近的会话。

### 注意事项

**推荐做法**:
- 使用有意义的会话名称
- 定期检查运行中的会话（`tmux ls`）
- 训练结束后清理会话

**避免**:
- 嵌套使用tmux
- 未分离会话就关闭SSH
- 在tmux内运行大型交互式程序

### 与代码的关系

**重要**: tmux是纯命令行工具，**代码不需要任何修改**。

- 不需要import任何库
- 不需要安装Python包
- 只在终端操作即可

---

## 完整示例

```bash
# === 第1天：启动训练 ===
ssh user@remote-server
tmux new -s ood_training
source ~/OOD_venv/bin/activate
cd ~/OOD-Analysis
python train.py
# Ctrl+b d
exit

# === 第2天：检查进度 ===
ssh user@remote-server
tmux attach -t ood_training
# 查看输出
# Ctrl+b d
exit

# === 第3天：训练完成 ===
ssh user@remote-server
tmux attach -t ood_training
# 确认完成
exit  # 退出tmux
tmux kill-session -t ood_training
```

---

## 参考

更多信息: `man tmux` 或 [tmux官方文档](https://github.com/tmux/tmux/wiki)
