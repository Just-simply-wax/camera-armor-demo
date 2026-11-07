#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lab.py —— 课堂实验脚本（曝光 / 增益 / 图像格式 / 内存与异常 / 参数对识别的影响）

一个命令对应一个实验，产物全部落在 runs/ 下（CSV + 图片 + summary.md），
直接把 summary.md 里的表格抄进实验记录即可。

    # 0) 环境自检（不接相机也能跑）
    python src/lab.py env

    # 1) 第一次连上相机：打印型号/序列号/固件版本 + 抓一张图
    python src/lab.py check

    # 2) 曝光实验（静止 / 运动各跑一次）
    python src/lab.py exposure --values 500,1000,3000 --tag still
    python src/lab.py exposure --values 500,1000,3000 --tag moving

    # 3) 增益实验
    python src/lab.py gain --values 0,6,12 --tag still --exposure 1000

    # 4) 图像格式与性能
    python src/lab.py format --frames 100

    # 5) 内存与异常处理
    python src/lab.py memory --case all

    # 6) 相机参数对识别效果的影响
    python src/lab.py recognize --matrix exposure:500,1000,3000 --gain 0
    python src/lab.py recognize --matrix gain:0,6,12 --exposure 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvs_camera import (Camera, CameraError, CameraTimeout,  # noqa: E402
                        error_text, list_devices, sdk_finalize)
from render import draw_detections  # noqa: E402

RUNS = Path("runs")


# --------------------------------------------------------------------------
# 通用小工具
# --------------------------------------------------------------------------
def parse_values(text: str) -> list[float]:
    return [float(x) for x in str(text).replace("，", ",").split(",") if x.strip()]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        print("[warn] 写图失败：%s" % path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def md_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "(无数据)"
    columns = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def frame_metrics(bgr: np.ndarray) -> dict:
    """一张图的客观指标：亮度、过曝比例、清晰度（拉普拉斯方差）、高频能量。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = gray.astype(np.float32)
    high = g - cv2.GaussianBlur(g, (5, 5), 0)
    return {
        "mean_gray": round(float(gray.mean()), 2),
        "p99_gray": round(float(np.percentile(gray, 99)), 1),
        "saturated_ratio": round(float((gray >= 250).mean()), 5),
        "lap_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "hf_energy": round(float(high.std()), 3),
    }


def summarize(frames: list[dict]) -> dict:
    """对同一组参数下的多帧取平均。"""
    keys = [k for k in frames[0] if k != "index"]
    out = {}
    for k in keys:
        values = [f[k] for f in frames if isinstance(f.get(k), (int, float))]
        if values:
            out[k] = round(float(np.mean(values)), 3)
    return out


def rss_mb() -> float:
    """当前进程占用的物理内存（MB）。用来观察有没有泄漏。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except OSError:
        pass
    return -1.0


def configure_or_warn(cam: Camera, **kwargs) -> None:
    """按需设置参数；相机不支持某个节点时只警告，不让整个实验挂掉。"""
    try:
        cam.configure(**{k: v for k, v in kwargs.items() if v is not None})
    except CameraError as exc:
        print("[warn] 参数设置失败（%s），用相机默认值继续" % exc)


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--index", type=int, default=0, help="相机序号（多台时用）")
    p.add_argument("--pixel-format", default="BayerRG8",
                   help="相机原生输出格式：BayerRG8 / BGR8 / Mono8")
    p.add_argument("--fps", type=float, default=30.0, help="采集帧率")


# --------------------------------------------------------------------------
# 0. env —— 环境自检
# --------------------------------------------------------------------------
def cmd_env(args) -> int:
    import cv2 as _cv2
    print("=" * 60)
    print("Python      :", sys.version.split()[0], platform.architecture()[0])
    print("平台        :", platform.platform())
    print("OpenCV      :", _cv2.__version__)
    try:
        import torch
        print("PyTorch     :", torch.__version__,
              "| CUDA 可用:", torch.cuda.is_available(),
              "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    except Exception as exc:  # noqa: BLE001
        print("PyTorch     : 未安装或导入失败 →", exc)
    try:
        import ultralytics
        print("ultralytics :", ultralytics.__version__)
    except Exception as exc:  # noqa: BLE001
        print("ultralytics : 未安装 →", exc)
    print("MVS_ROOT    :", os.environ.get("MVS_ROOT", "/opt/MVS"))
    print("DISPLAY     :", os.environ.get("DISPLAY", "(空，无法弹窗)"))
    print("=" * 60)
    try:
        out = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
        print("[lsusb]\n" + (out.stdout or out.stderr).strip())
    except Exception:
        print("[lsusb] 没有这个命令（apt install usbutils）")
    print("=" * 60)
    try:
        devs = list_devices()
        print("枚举到 %d 台相机" % len(devs))
        for dev in devs:
            print(" ", dev.summary())
    except CameraError as exc:
        print("[error]", exc)
        return 2
    return 0


# --------------------------------------------------------------------------
# 1. check —— 第一次连上相机
# --------------------------------------------------------------------------
def cmd_check(args) -> int:
    devs = list_devices()
    if not devs:
        print("[error] 没有枚举到相机")
        return 1
    print("枚举到 %d 台：" % len(devs))
    for dev in devs:
        print(" ", dev.summary())

    outdir = ensure_dir(RUNS / "check")
    with Camera(index=args.index) as cam:
        info = cam.snapshot_info()
        print("\n[camera info]")
        print(json.dumps(info, ensure_ascii=False, indent=2))

        for key in ("ExposureTime", "Gain", "AcquisitionFrameRate"):
            try:
                cur, lo, hi = cam.get_float(key)
                print("  %-22s 当前=%.2f  范围=[%.2f, %.2f]" % (key, cur, lo, hi))
            except CameraError:
                print("  %-22s 该相机没有这个节点" % key)
        try:
            print("  %-22s %s" % ("PixelFormat",
                                   hex(cam.get_enum("PixelFormat"))))
        except CameraError:
            pass

        configure_or_warn(cam, exposure_us=args.exposure, gain_db=args.gain,
                          pixel_format=args.pixel_format, fps=args.fps)
        cam.start()

        shots = []
        for i in range(max(1, args.frames)):
            t0 = time.perf_counter()
            raw = cam.grab(args.timeout_ms)
            grab_ms = (time.perf_counter() - t0) * 1000
            img = cam.to_bgr(raw)
            shots.append((raw, img))
            print("  帧 %d: %dx%d %s  %.1f KB  grab=%.1fms convert=%.1fms"
                  % (i + 1, raw.width, raw.height, raw.pixel_name,
                     raw.nbytes / 1024.0, grab_ms, cam.last_convert_ms))
        raw, img = shots[-1]
        path = outdir / ("check_%dx%d_%s.png" % (raw.width, raw.height, raw.pixel_name))
        save_png(path, img)
        (outdir / "info.json").write_text(
            json.dumps({"info": info, "snapshot": cam.snapshot_info()},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n[done] 保存一帧到 %s" % path)

    print("[done] 相机已关闭，句柄已销毁")
    return 0


# --------------------------------------------------------------------------
# 2. 曝光 / 增益实验
# --------------------------------------------------------------------------
def _param_experiment(args, kind: str) -> int:
    if kind == "exposure":
        values = parse_values(args.values)
        outdir = ensure_dir(RUNS / "exp1_exposure" / args.tag)
        title = "曝光时间"
    else:
        values = parse_values(args.values)
        outdir = ensure_dir(RUNS / "exp2_gain" / args.tag)
        title = "增益"

    rows: list[dict] = []
    with Camera(index=args.index) as cam:
        configure_or_warn(cam, pixel_format=args.pixel_format, fps=args.fps,
                          exposure_us=args.exposure if kind == "gain" else None,
                          gain_db=args.gain if kind == "exposure" else None)
        cam.start()

        for value in values:
            actual = cam.set_exposure(value) if kind == "exposure" else cam.set_gain(value)
            time.sleep(args.settle)
            for _ in range(args.warmup):          # 丢掉自动曝光/增益收敛过程中的帧
                cam.grab(args.timeout_ms)

            frames = []
            for i in range(args.frames):
                t0 = time.perf_counter()
                raw = cam.grab(args.timeout_ms)
                grab_ms = (time.perf_counter() - t0) * 1000
                img = cam.to_bgr(raw)
                metric = frame_metrics(img)
                metric["index"] = i
                metric["grab_ms"] = round(grab_ms, 2)
                metric["convert_ms"] = round(cam.last_convert_ms, 2)
                frames.append(metric)
                if i < 2:
                    save_png(outdir / ("%s_%04d_f%d.png"
                                       % (kind, int(round(actual)), i)), img)

            row = {"set_value": value, "actual_value": round(actual, 2),
                   "frames": args.frames}
            row.update(summarize(frames))
            rows.append(row)
            print("  %s=%.1f（实际 %.1f）→ 亮度 %.1f，饱和像素 %.3f%%，拉普拉斯方差 %.1f"
                  % (kind, value, actual, row.get("mean_gray", 0),
                     row.get("saturated_ratio", 0) * 100, row.get("lap_var", 0)))

    write_csv(outdir / "result.csv", rows)
    summary = [
        "# %s实验（%s）" % (title, args.tag),
        "",
        "相机：%s" % args.pixel_format,
        "",
        md_table(rows),
        "",
        "> 说明：`mean_gray` 是画面平均灰度（0–255）；`saturated_ratio` 是灰度 ≥250 的像素占比；",
        "> `lap_var` 是拉普拉斯方差，越大表示边缘越锐利（运动模糊会让它变小）；",
        "> `hf_energy` 是高频能量，增益越高噪声越大，它也会变大。",
    ]
    (outdir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("[done] 结果：%s/result.csv" % outdir)
    return 0


def cmd_exposure(args) -> int:
    return _param_experiment(args, "exposure")


def cmd_gain(args) -> int:
    return _param_experiment(args, "gain")


# --------------------------------------------------------------------------
# 3. 图像格式与取流性能
# --------------------------------------------------------------------------
def _set_format(cam: Camera, name: str) -> None:
    """切换相机原生输出格式（必须先停止取流）。"""
    cam.stop()
    cam.set_pixel_format(name)


def cmd_format(args) -> int:
    modes = args.modes or ["bayer_cv2", "bayer_sdk", "bgr8"]
    rows: list[dict] = []
    outdir = ensure_dir(RUNS / "exp3_format")

    with Camera(index=args.index) as cam:
        for mode in modes:
            quality = None
            if mode == "bayer_cv2":
                native, convert = "BayerRG8", "cv2"
            elif mode.startswith("bayer_sdk"):
                native, convert = "BayerRG8", "sdk"
                if ":" in mode:
                    quality = int(mode.split(":", 1)[1])
            elif mode == "bgr8":
                native, convert = "BGR8", "sdk"
            else:
                print("[warn] 不认识的模式 %s，跳过" % mode)
                continue

            try:
                _set_format(cam, native)
            except CameraError as exc:
                print("[warn] 相机不支持 %s：%s" % (native, exc))
                rows.append({"mode": mode, "native": native, "convert": convert,
                             "frames": 0, "fps": "", "grab_ms_mean": "",
                             "convert_ms_mean": "", "convert_ms_p95": "",
                             "cpu_percent": "", "MBps": "", "bytes_per_frame": "",
                             "note": "相机不支持该格式"})
                continue

            if quality is not None:
                try:
                    cam.set_bayer_quality(quality)
                    mode = "bayer_sdk:q%d" % quality
                except CameraError as exc:
                    print("[warn] 无法设置 Bayer 转换质量：%s" % exc)

            cam.start()
            for _ in range(args.warmup):
                cam.grab(args.timeout_ms)

            grab_ms, conv_ms, nbytes = [], [], 0
            cpu0, wall0 = time.process_time(), time.perf_counter()
            ok = 0
            for _ in range(args.frames):
                t0 = time.perf_counter()
                raw = cam.grab(args.timeout_ms)
                t1 = time.perf_counter()
                cam.to_bgr(raw, mode=convert)
                t2 = time.perf_counter()
                grab_ms.append((t1 - t0) * 1000)
                conv_ms.append((t2 - t1) * 1000)
                nbytes += raw.nbytes
                ok += 1
            wall = time.perf_counter() - wall0
            cpu = time.process_time() - cpu0

            row = {
                "mode": mode,
                "native": native,
                "convert": convert,
                "frames": ok,
                "fps": round(ok / wall, 2),
                "grab_ms_mean": round(float(np.mean(grab_ms)), 2),
                "convert_ms_mean": round(float(np.mean(conv_ms)), 2),
                "convert_ms_p95": round(float(np.percentile(conv_ms, 95)), 2),
                "cpu_percent": round(cpu / wall * 100, 1),
                "MBps": round(nbytes / wall / 1e6, 1),
                "bytes_per_frame": int(nbytes / max(1, ok)),
            }
            rows.append(row)
            print("  %-14s fps=%.1f 转换=%.2fms CPU=%.0f%% 带宽=%.1fMB/s"
                  % (mode, row["fps"], row["convert_ms_mean"],
                     row["cpu_percent"], row["MBps"]))
            cam.stop()

    write_csv(outdir / "result.csv", rows)
    summary = ["# 图像格式与取流性能", "",
               md_table(rows), "",
               "> `native` 是相机原生输出格式，`convert` 是程序做的转换；",
               "> `MBps` 是每秒从相机搬过来的原始字节数（Bayer 只有 BGR 的 1/3，所以带宽小）；",
               "> `cpu_percent` = 进程 CPU 时间 / 墙钟时间，多线程转换时会超过 100%。"]
    (outdir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("[done] 结果：%s/result.csv" % outdir)
    return 0


# --------------------------------------------------------------------------
# 4. 内存与异常处理
# --------------------------------------------------------------------------
def _case_normal(cam: Camera, log: list[str]) -> bool:
    log.append("--- 正常取流 200 帧，每帧都 FreeImageBuffer ---")
    start = rss_mb()
    for i in range(200):
        cam.grab(2000)
        if i % 50 == 0:
            log.append("  第 %3d 帧 RSS=%.1f MB" % (i, rss_mb()))
    end = rss_mb()
    log.append("  RSS: %.1f → %.1f MB（差值 %.1f MB）" % (start, end, end - start))
    ok = abs(end - start) < 60
    log.append("  结论：%s" % ("内存平稳，没有泄漏" if ok else "内存明显增长，检查是否漏归还缓冲"))
    return ok


def _case_leak(cam: Camera, log: list[str]) -> bool:
    log.append("--- 故意不归还缓冲：借 12 帧后不再 FreeImageBuffer ---")
    held = []
    try:
        for i in range(12):
            st = cam.grab_unreleased(2000)
            held.append(st)
            info = st.stFrameInfo
            log.append("  借出第 %d 帧 %dx%d（未归还）" % (i + 1, info.nWidth, info.nHeight))
    except CameraError as exc:
        log.append("  借到第 %d 帧就失败了：%s" % (len(held) + 1, exc))

    log.append("  缓冲池被占满后再取图，预期失败：")
    failed = False
    try:
        cam.grab(500)
        log.append("  意外：仍然取到了图（缓冲池比预想的大）")
    except CameraTimeout as exc:
        failed = True
        log.append("  如期超时：%s" % exc)
    except CameraError as exc:
        failed = True
        log.append("  如期报错：%s" % exc)

    log.append("  把借出的缓冲全部归还后，再试一次：")
    for st in held:
        cam.release_frame(st)
    held.clear()
    recovered = True
    try:
        cam.grab(2000)
        log.append("  恢复正常取图 ✅")
    except CameraError as exc:
        recovered = False
        log.append("  仍然取不到图 ❌ %s" % exc)
    return failed and recovered


def _case_timeout(cam: Camera, log: list[str]) -> bool:
    log.append("--- 取流超时：把曝光拉到很长，用很短的 timeout 取图 ---")
    original = cam.get_float("ExposureTime")[0]
    cam.set_bool("AcquisitionFrameRateEnable", False)      # 先放开帧率限制
    try:
        exposed = cam.set_exposure(200000)
    except CameraError:
        exposed = cam.set_exposure(50000)                  # 有些相机上限更小
    log.append("  曝光设为 %.0f us，timeout=50 ms" % exposed)
    timeouts = 0
    for _ in range(3):
        try:
            cam.grab(50)
        except CameraTimeout as exc:
            timeouts += 1
            log.append("  第 %d 次：%s" % (timeouts, exc))
        except CameraError as exc:                         # 掉线/其它错误也算异常路径
            log.append("  第 %d 次拿到其它错误：%s" % (timeouts + 1, exc))
    cam.set_exposure(original)
    cam.set_bool("AcquisitionFrameRateEnable", True)
    time.sleep(0.5)
    recovered = True
    try:
        cam.grab(3000)
        log.append("  恢复曝光 %.0f us 后可以正常取图 ✅" % original)
    except CameraError as exc:
        recovered = False
        log.append("  恢复失败 ❌ %s" % exc)
    return timeouts > 0 and recovered


def _case_badformat(cam: Camera, log: list[str]) -> bool:
    log.append("--- 错误/不支持的图像格式 ---")
    cam.stop()                       # 改格式必须在停止取流时做
    before = cam.get_enum("PixelFormat")
    ret = cam.try_set_enum_str("PixelFormat", "NotExistFormat")
    log.append("  设置 PixelFormat=NotExistFormat → %s（0x%08X）" % (error_text(ret), ret & 0xFFFFFFFF))
    after = cam.get_enum("PixelFormat")
    log.append("  当前格式未被破坏：0x%08X → 0x%08X" % (before, after))

    ret2 = cam.try_set_enum_str("PixelFormat", "BGR8")
    log.append("  设置 PixelFormat=BGR8 → %s（0x%08X）"
               % (error_text(ret2), ret2 & 0xFFFFFFFF))
    if ret2 == 0:
        cam.try_set_enum_str("PixelFormat", "BayerRG8")   # 还原成 Bayer
    cam.start()
    return after == before


def _case_reopen(cam: Camera, log: list[str]) -> bool:
    log.append("--- 反复打开/关闭，检查句柄是否真的释放 ---")
    ok = True
    for i in range(3):
        cam.stop()
        cam.close()
        time.sleep(0.2)
        try:
            cam.open()
            cam.start()
            cam.grab(2000)
            log.append("  第 %d 轮：关闭再打开成功" % (i + 1))
        except CameraError as exc:
            ok = False
            log.append("  第 %d 轮：失败 %s" % (i + 1, exc))
            break
    return ok


def _case_mat(_cam, log: list[str]) -> bool:
    log.append("--- OpenCV Mat / numpy 的内存语义（不需要相机）---")
    a = np.zeros((4, 4), np.uint8)
    b = a                 # 浅拷贝：b 和 a 是同一块内存（类似 cv::Mat 的引用计数赋值）
    c = a.copy()          # 深拷贝：新内存（相当于 cv::Mat::clone()）
    a[0, 0] = 255
    log.append("  a[0,0]=255 后：b[0,0]=%d（看到 255 说明共用内存），c[0,0]=%d（副本不受影响）"
               % (b[0, 0], c[0, 0]))
    log.append("  a.data 指针=0x%X，b.data 指针=0x%X（相同），c.data 指针=0x%X（不同）"
               % (a.ctypes.data, b.ctypes.data, c.ctypes.data))
    log.append("  结论：`b = a` 不复制数据；`a.copy()` 才复制。")
    log.append("  相机取图同理：MV_CC_GetImageBuffer 给的指针在 FreeImageBuffer 之后失效，")
    log.append("  所以必须在归还之前把数据复制出来（本项目的 Camera.grab 已经这么做了）。")
    return b[0, 0] == 255 and c[0, 0] == 0


def cmd_memory(args) -> int:
    cases = ["normal", "leak", "timeout", "badformat", "reopen", "mat"]
    if args.case != "all":
        cases = [args.case]

    outdir = ensure_dir(RUNS / "exp4_memory")
    results = []

    need_camera = [c for c in cases if c != "mat"]
    cam = None
    if need_camera:
        cam = Camera(index=args.index)
        cam.open()
        configure_or_warn(cam, exposure_us=2000, gain_db=0,
                          pixel_format=args.pixel_format, fps=args.fps)
        cam.start()

    try:
        for name in cases:
            log: list[str] = ["# case: %s" % name, "time: %s" % time.strftime("%F %T")]
            fn = {"normal": _case_normal, "leak": _case_leak, "timeout": _case_timeout,
                  "badformat": _case_badformat, "reopen": _case_reopen, "mat": _case_mat}[name]
            print("\n[case] %s" % name)
            try:
                ok = fn(cam, log)
            except Exception as exc:  # noqa: BLE001
                ok = False
                log.append("  异常：%r" % exc)
            finally:
                log.append("  RSS=%.1f MB" % rss_mb())
            for line in log:
                print("  " + line)
            (outdir / ("log_%s.txt" % name)).write_text("\n".join(log) + "\n",
                                                        encoding="utf-8")
            results.append({"case": name, "ok": ok})
            if name == "reopen" and not ok:
                cam = None            # 相机已经关不回去了，后面的 case 不能再跑
    finally:
        if cam is not None:
            cam.close()
        sdk_finalize()

    print("\n[summary]")
    for r in results:
        print("  %-10s %s" % (r["case"], "通过" if r["ok"] else "失败/需人工确认"))
    (outdir / "summary.md").write_text(
        "# 内存与异常处理\n\n" + md_table(results) + "\n", encoding="utf-8")
    print("[done] 日志：%s" % outdir)
    return 0 if all(r["ok"] for r in results) else 1


# --------------------------------------------------------------------------
# 5. 相机参数对识别效果的影响
# --------------------------------------------------------------------------
def cmd_recognize(args) -> int:
    key, _, values_text = args.matrix.partition(":")
    if key not in ("exposure", "gain"):
        print("[error] --matrix 格式：exposure:500,1000,3000 或 gain:0,6,12")
        return 2
    values = parse_values(values_text)
    outdir = ensure_dir(RUNS / "exp6_recognition")

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    half = args.half and str(args.device).lower() not in ("cpu", "-1")

    rows = []
    with Camera(index=args.index) as cam:
        configure_or_warn(cam, pixel_format=args.pixel_format, fps=args.fps,
                          exposure_us=args.exposure, gain_db=args.gain)
        cam.start()

        # 模型预热：第一次推理含 CUDA 初始化，不预热会污染第一组数据
        try:
            _raw = cam.grab(args.timeout_ms)
            _img = cam.to_bgr(_raw)
            for _ in range(3):
                model.predict(_img, conf=args.conf, imgsz=args.imgsz,
                              device=args.device, half=half, verbose=False)
            print("[warmup] 模型预热完成")
        except CameraError as exc:
            print("[warn] 模型预热时取图失败：%s" % exc)

        for value in values:
            if key == "exposure":
                actual = cam.set_exposure(value)
            else:
                actual = cam.set_gain(value)
            time.sleep(args.settle)
            for _ in range(args.warmup):
                cam.grab(args.timeout_ms)

            det_frames, det_count, conf_sum = 0, 0, 0.0
            brightness, satur, infer_ms = [], [], []
            sample = None
            for _ in range(args.frames):
                raw = cam.grab(args.timeout_ms)
                img = cam.to_bgr(raw)
                metric = frame_metrics(img)
                brightness.append(metric["mean_gray"])
                satur.append(metric["saturated_ratio"])

                t0 = time.perf_counter()
                result = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                                       device=args.device, half=half,
                                       verbose=False)[0]
                infer_ms.append((time.perf_counter() - t0) * 1000)
                dets = draw_detections(img, result, names)
                if dets:
                    det_frames += 1
                    det_count += len(dets)
                    conf_sum += sum(d["conf"] for d in dets)
                    sample = img

            row = {
                "param": key,
                "set_value": value,
                "actual_value": round(actual, 2),
                "frames": args.frames,
                "detection_rate": round(det_frames / args.frames, 3),
                "det_per_frame": round(det_count / args.frames, 2),
                "avg_conf": round(conf_sum / max(1, det_count), 3),
                "mean_gray": round(float(np.mean(brightness)), 1),
                "saturated_ratio": round(float(np.mean(satur)), 4),
                "infer_ms": round(float(np.mean(infer_ms)), 2),
            }
            rows.append(row)
            if sample is not None:
                save_png(outdir / ("%s_%04d.jpg" % (key, int(round(actual)))), sample)
            print("  %s=%.1f → 检出率 %.0f%%，每帧目标 %.2f，平均置信度 %.3f，亮度 %.0f"
                  % (key, actual, row["detection_rate"] * 100, row["det_per_frame"],
                     row["avg_conf"], row["mean_gray"]))

    write_csv(outdir / "result.csv", rows)
    (outdir / "summary.md").write_text(
        "# 相机参数对识别效果的影响\n\n" + md_table(rows) + "\n", encoding="utf-8")
    print("[done] 结果：%s/result.csv" % outdir)
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="工业相机课堂实验脚本",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("env", help="环境自检")
    sp.set_defaults(func=cmd_env)

    sp = sub.add_parser("check", help="枚举 + 打开 + 抓图")
    add_common(sp)
    sp.add_argument("--exposure", type=float, default=2000)
    sp.add_argument("--gain", type=float, default=0)
    sp.add_argument("--frames", type=int, default=3)
    sp.add_argument("--timeout-ms", type=int, default=3000)
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("exposure", help="曝光实验")
    add_common(sp)
    sp.add_argument("--values", default="500,1000,3000", help="曝光时间列表，单位 us")
    sp.add_argument("--tag", default="still", help="场景标签：still / moving")
    sp.add_argument("--gain", type=float, default=0.0, help="固定增益 dB")
    sp.add_argument("--frames", type=int, default=20)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--settle", type=float, default=0.5, help="改参数后等待秒数")
    sp.add_argument("--timeout-ms", type=int, default=3000)
    sp.set_defaults(func=cmd_exposure)

    sp = sub.add_parser("gain", help="增益实验")
    add_common(sp)
    sp.add_argument("--values", default="0,6,12", help="增益列表，单位 dB")
    sp.add_argument("--tag", default="still")
    sp.add_argument("--exposure", type=float, default=1000.0, help="固定曝光 us")
    sp.add_argument("--frames", type=int, default=20)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--settle", type=float, default=0.5)
    sp.add_argument("--timeout-ms", type=int, default=3000)
    sp.set_defaults(func=cmd_gain)

    sp = sub.add_parser("format", help="图像格式与取流性能")
    add_common(sp)
    sp.add_argument("--modes", nargs="*",
                    default=["bayer_cv2", "bayer_sdk", "bgr8"],
                    help="bayer_cv2 / bayer_sdk[:质量0-2] / bgr8")
    sp.add_argument("--frames", type=int, default=100)
    sp.add_argument("--warmup", type=int, default=10)
    sp.add_argument("--timeout-ms", type=int, default=3000)
    sp.set_defaults(func=cmd_format)

    sp = sub.add_parser("memory", help="内存与异常处理")
    add_common(sp)
    sp.add_argument("--case", default="all",
                    choices=["all", "normal", "leak", "timeout", "badformat", "reopen", "mat"])
    sp.set_defaults(func=cmd_memory)

    sp = sub.add_parser("recognize", help="参数对识别效果的影响")
    add_common(sp)
    sp.add_argument("--weights", default="weights/best.pt")
    sp.add_argument("--matrix", default="exposure:500,1000,3000",
                    help="exposure:500,1000,3000 或 gain:0,6,12")
    sp.add_argument("--exposure", type=float, default=1000.0, help="固定曝光（做增益实验时用）")
    sp.add_argument("--gain", type=float, default=0.0, help="固定增益（做曝光实验时用）")
    sp.add_argument("--frames", type=int, default=30)
    sp.add_argument("--warmup", type=int, default=5)
    sp.add_argument("--settle", type=float, default=0.5)
    sp.add_argument("--conf", type=float, default=0.25)
    sp.add_argument("--imgsz", type=int, default=640)
    sp.add_argument("--device", default="0")
    sp.add_argument("--half", action="store_true")
    sp.add_argument("--timeout-ms", type=int, default=3000)
    sp.set_defaults(func=cmd_recognize)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except CameraError as exc:
        print("[camera error] %s" % exc)
        return 2
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 130
    finally:
        sdk_finalize()


if __name__ == "__main__":
    sys.exit(main())
