# 工业相机 + YOLO 装甲板实时识别（WSL2 / Ubuntu）

用海康机器人 MVS 工业相机实时取流，做曝光 / 增益 / 图像格式 / 内存异常四组实验，
再把画面送进课程提供的 YOLO 权重做实时装甲板识别。

**处理链路：**

```
工业相机 ──USB3/GigE──> WSL2 Ubuntu ──MVS SDK──> 原始帧(BayerRG8 / BGR8)
                                                   │
                                     Bayer→BGR 转换（SDK 或 OpenCV）
                                                   │
                                          YOLO26n-pose 推理
                                                   │
                        画框 + 类别 + 置信度 + 关键点 + 中心点 + FPS
                                                   │
                                          窗口显示 / 录制 mp4
```

## 1. 实验环境【待填：以你机器实际输出为准】

| 项目 | 内容 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS（WSL2，内核 `6.18.33.2-microsoft-standard-WSL2`） |
| 相机型号 | **MV-CA016-10UC**（海康机器人 CA 系列，1.6 MP） |
| 相机序列号 | **L28498156** |
| 相机固件版本 | 【待填，`python src/lab.py check` 会打印】 |
| 相机接口 | **USB3.0 Vision**（Windows `usbipd` 挂进 WSL） |
| 传感器 / 快门 | Sony IMX273，1/2.9"，3.45 μm，**全局快门** |
| 相机分辨率 / ROI | **1440×1080**（全幅） |
| 镜头 | 8 mm F2.8（6 MP 定焦） |
| MVS 版本 | 5.1.1（Linux x86_64，`/opt/MVS`） |
| OpenCV | 【待填，`python src/lab.py env` 会打印】 |
| Python / PyTorch / ultralytics | 【待填】 |
| GPU | NVIDIA GeForce RTX 5060 Laptop（8 GB） |
| 模型权重 | `weights/best.pt`（YOLO26n-pose，6 类 `B1 B2 B3 R1 R2 R3`，4 关键点） |
| 光源 | 【待填：实验室 LED 面光源 / 顶灯，色温、距离】 |

## 2. 目录结构

```
.
├── src/
│   ├── mvs_camera.py     # 相机封装：枚举/打开/参数/取流/转换/释放
│   ├── render.py         # 画框、类别、置信度、关键点、HUD
│   ├── app.py            # 实时识别主程序（相机 / 视频 / 合成画面）
│   └── lab.py            # 六组实验的命令行入口
├── tools/probe_env.sh    # 环境自检脚本
├── docs/实验记录.md       # 实验记录（提交物）
├── weights/              # best.pt（不入库）
├── data/                 # 测试视频（不入库）
├── runs/                 # 实验产物：图片、CSV、日志、录像（不入库）
└── assets/               # README 用的截图（入库）
```

## 3. 运行方法

```bash
# 0) 环境自检（不接相机也能跑）
bash tools/probe_env.sh
python src/lab.py env

# 1) 连上相机后第一件事：确认型号/序列号/固件，并抓一张图
python src/lab.py check

# 2) 实时识别（显示 + 录像 + 统计）
python src/app.py --source mvs --exposure 1000 --gain 0 \
    --pixel-format BayerRG8 --record runs/demo.mp4 --stats runs/demo.json

# 3) 没有相机时用视频文件跑同一套链路（调试用）
python src/app.py --source video --video data/armor_raw.mp4 \
    --no-display --record runs/dryrun.mp4 --max-frames 300
```

`src/app.py` 常用参数：

| 参数 | 说明 |
| --- | --- |
| `--source mvs/video/test` | 真实相机 / 视频文件 / 合成画面 |
| `--exposure` `--gain` | 曝光（μs）、增益（dB） |
| `--pixel-format` | 相机原生输出格式：`BayerRG8` / `BGR8` / `Mono8` |
| `--convert sdk/cv2` | Bayer→BGR 用 SDK 还是 OpenCV |
| `--imgsz` `--conf` `--device` `--half` | 推理参数 |
| `--record` `--record-fps` | 结果视频 |
| `--display/--no-display` | 是否弹窗（无 WSLg 时用 `--no-display`） |

## 4. 四组相机实验

| 实验 | 命令 | 产物 |
| --- | --- | --- |
| 曝光 | `python src/lab.py exposure --values 500,1000,3000 --tag still` | `runs/exp1_exposure/still/` |
| 曝光（运动） | `python src/lab.py exposure --values 500,1000,3000 --tag moving` | `runs/exp1_exposure/moving/` |
| 增益 | `python src/lab.py gain --values 0,6,12 --exposure 1000` | `runs/exp2_gain/` |
| 格式 | `python src/lab.py format --frames 100` | `runs/exp3_format/` |
| 内存/异常 | `python src/lab.py memory --case all` | `runs/exp4_memory/` |
| 参数对识别 | `python src/lab.py recognize --matrix exposure:500,1000,3000` | `runs/exp6_recognition/` |

每个实验目录里都有 `result.csv`（原始数据）和 `summary.md`（可直接抄进实验记录的表格）。

## 5. 实验结果【待填】

### 5.1 曝光时间

【贴 `runs/exp1_exposure/*/summary.md` 的表格 + 两张对比图】

### 5.2 增益

【贴 `runs/exp2_gain/summary.md` 的表格】

### 5.3 图像格式与性能

【贴 `runs/exp3_format/summary.md` 的表格】

### 5.4 内存与异常处理

【贴 `runs/exp4_memory/summary.md` + 一段日志摘录】

### 5.5 实时识别

| 指标 | 数值 |
| --- | --- |
| 处理帧率 | 【待填】 |
| 平均推理耗时 | 【待填】 ms |
| 平均取图耗时 | 【待填】 ms |
| 平均转换耗时 | 【待填】 ms |
| 检出率（有目标的帧占比） | 【待填】 |

![实时识别截图](assets/demo_frame.jpg)

## 6. 关键实现说明

1. **取图必须成对**：`MV_CC_GetImageBuffer()` 借一帧，`MV_CC_FreeImageBuffer()` 还一帧。
   本项目把这对调用放在 `try/finally` 里（`mvs_camera.Camera.grab`），异常也不会漏还。
   漏还的后果见 `runs/exp4_memory/log_leak.txt`：缓冲池耗尽后取流超时，归还后立刻恢复。
2. **只保留最新一帧**：`app.py` 用后台线程取图、队列长度 1，推理慢时丢掉旧帧，
   保证画面和检测框始终对应"此刻"，不会越跑越延迟。
3. **两种格式转换路径**：`--convert sdk` 用 `MV_CC_ConvertPixelTypeEx`（SIMD），
   `--convert cv2` 用 `cv2.cvtColor`，用于对比转换耗时（实验三）。
4. **曝光/增益前先关自动**：`ExposureAuto=Off`、`GainAuto=Off`，否则手动值不生效。
5. **结果视频转 H.264**：OpenCV 只能稳定写 `mp4v`，最后用 ffmpeg 转 `yuv420p + faststart`，
   否则浏览器/微信可能打不开。

## 7. 已知问题与处理

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 枚举不到相机（GigE） | WSL 默认 NAT，收不到相机广播 | `.wslconfig` 设 `networkingMode=mirrored`，重启 WSL |
| 枚举不到相机（USB） | 设备没挂进 WSL | `usbipd attach --wsl`，再 `lsusb` 确认 |
| `MV_CC_OpenDevice` 失败 | MVS 客户端或另一个进程占着相机 | 关掉 MVS，`pkill -f src/` |
| 画面偏绿/偏紫 | Bayer 排列不是 RGGB | 试 `BayerGR8/BayerGB8/BayerBG8` |
| 帧率上不去 | 曝光太长 / 带宽不够 / 转换慢 | 看 `runs/exp3_format/result.csv` 定位 |

## 8. 提交物

- [ ] 源码（本仓库）
- [ ] 实验记录 `docs/实验记录.md`
- [ ] 真实相机识别演示视频（另附）
- [ ] 异常处理日志 `runs/exp4_memory/log_*.txt`
