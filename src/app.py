#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py —— 工业相机 + YOLO 装甲板实时识别（主程序）

它把一条完整的链路串起来：

    [相机 / 视频文件]  →  取一帧  →  Bayer→BGR 转换  →  YOLO 推理
                                                            ↓
                                        画框/类别/置信度/关键点/中心点
                                                            ↓
                                              窗口显示 + 录制 + 统计

两个关键设计（验收时会被问）：

1) "取图线程 + 只保留最新一帧"
   如果主循环里"取一帧→推理→显示"串行执行，一旦推理比相机慢，
   队列就会越积越多，画面延迟越来越大——你移动装甲板，框要过几秒才跟上来。
   这里的做法是：后台线程不停地取图，队列长度只有 1，新帧来了就覆盖旧帧。
   主循环每次拿到的永远是"此刻最新的一帧"，所以框能实时跟随目标。

2) 取流和转换分开计时
   grab_ms 是相机取图耗时，convert_ms 是 Bayer→BGR 耗时，infer_ms 是推理耗时。
   三个数字分开记录，才能说清楚"帧率上不去到底卡在哪一步"。

常用命令（在仓库根目录执行）：

    # 真实相机 + 显示 + 录像
    python src/app.py --source mvs --exposure 1000 --gain 0 --record runs/demo.mp4

    # 没有相机时，用视频文件冒充相机（整条链路完全一样）
    python src/app.py --source video --video data/armor_raw.mp4 --no-display \
        --record runs/dryrun.mp4 --max-frames 300
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvs_camera import Camera, CameraError, CameraTimeout, sdk_finalize  # noqa: E402
from render import draw_detections, draw_hud  # noqa: E402


# --------------------------------------------------------------------------
# 一、帧源：真实相机 / 视频文件 / 测试图，接口都是 read() -> (BGR 图, 元信息)
# --------------------------------------------------------------------------
class MvsSource:
    """真实工业相机。"""

    def __init__(self, index: int, convert: str, timeout_ms: int):
        self.cam = Camera(index=index)
        self.cam.open()
        self.convert = convert
        self.timeout_ms = timeout_ms

    def configure(self, **kwargs):
        self.cam.configure(**kwargs)
        return self

    def start(self):
        self.cam.start()
        return self

    def read(self):
        t0 = time.perf_counter()
        raw = self.cam.grab(self.timeout_ms)
        grab_ms = (time.perf_counter() - t0) * 1000
        img = self.cam.to_bgr(raw, mode=self.convert)
        return img, {"kind": "mvs", "ts": raw.host_ts, "frame_num": raw.frame_num,
                     "grab_ms": grab_ms, "convert_ms": self.cam.last_convert_ms,
                     "pixel": raw.pixel_name, "lost": raw.lost_packet}

    def info(self) -> dict:
        return self.cam.snapshot_info()

    def close(self):
        self.cam.close()


class VideoSource:
    """把视频文件当相机用：按视频帧率节拍出图，这样 FPS 统计才有意义。"""

    def __init__(self, path: str, loop: bool = True, realtime: bool = True):
        if not Path(path).exists():
            raise FileNotFoundError("视频不存在：%s" % path)
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError("打不开视频：%s" % path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 < fps < 240 else 30.0
        self.dt = 1.0 / self.fps
        self.loop = loop
        self.realtime = realtime
        self.next_t = time.perf_counter()
        self.count = 0

    def configure(self, **_kwargs):
        return self

    def start(self):
        return self

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            self.next_t = time.perf_counter()
        if not ok:
            raise EOFError("视频播放结束")
        self.count += 1
        if self.realtime:
            self.next_t += self.dt
            delay = self.next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                self.next_t = time.perf_counter()
        return frame, {"kind": "video", "ts": time.perf_counter(),
                       "frame_num": self.count, "grab_ms": 0.0, "convert_ms": 0.0,
                       "pixel": "BGR8", "lost": 0}

    def info(self) -> dict:
        return {"kind": "video", "path": getattr(self, "path", ""),
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": round(self.fps, 2)}

    def close(self):
        self.cap.release()


class TestSource:
    """纯合成画面：不接相机、不读文件也能验证整条链路（画面上没有装甲板）。"""

    def __init__(self, width=1280, height=720, fps=30.0):
        self.w, self.h, self.fps = width, height, fps
        self.count = 0
        self.next_t = time.perf_counter()

    def configure(self, **_kwargs):
        return self

    def start(self):
        return self

    def read(self):
        self.count += 1
        img = np.zeros((self.h, self.w, 3), np.uint8)
        img[:] = (30, 30, 30)
        x = int((self.count * 6) % (self.w - 200))
        cv2.rectangle(img, (x, self.h // 3), (x + 160, self.h // 3 + 110), (0, 200, 255), -1)
        cv2.putText(img, "TEST PATTERN %d" % self.count, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        self.next_t += 1.0 / self.fps
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.perf_counter()
        return img, {"kind": "test", "ts": time.perf_counter(), "frame_num": self.count,
                     "grab_ms": 0.0, "convert_ms": 0.0, "pixel": "-", "lost": 0}

    def info(self) -> dict:
        return {"kind": "test", "width": self.w, "height": self.h, "fps": self.fps}

    def close(self):
        pass


# --------------------------------------------------------------------------
# 二、后台取图：队列只留最新一帧
# --------------------------------------------------------------------------
class LatestFrameGrabber:
    def __init__(self, source, stop_event: threading.Event):
        self.source = source
        self.stop_event = stop_event
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.error = None
        self.captured = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="grabber")

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            try:
                frame, meta = self.source.read()
            except CameraTimeout:
                continue                      # 偶发超时不算错误，继续取
            except Exception as exc:          # noqa: BLE001
                if not self.stop_event.is_set():
                    self.error = exc
                return
            self.captured += 1
            try:
                self.q.put_nowait((frame, meta))
            except queue.Full:
                try:
                    self.q.get_nowait()       # 丢掉旧帧，保证拿到的是最新的
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait((frame, meta))
                except queue.Full:
                    pass

    def get(self, timeout: float = 0.5):
        return self.q.get(timeout=timeout)

    def join(self, timeout: float = 2.0):
        self.thread.join(timeout=timeout)


# --------------------------------------------------------------------------
# 三、录像：OpenCV 只能稳稳地写 mp4v，最后用 ffmpeg 转成通用 H.264
# --------------------------------------------------------------------------
def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _to_h264(src: Path, dst: Path):
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        shutil.move(str(src), str(dst))
        print("[warn] 没找到 ffmpeg，保留 mp4v 编码（部分播放器可能打不开）")
        return
    cmd = [ffmpeg, "-y", "-i", str(src),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "medium", "-crf", "23",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        src.unlink(missing_ok=True)
    except Exception as exc:  # ffmpeg 失败就保留 mp4v 原件，至少能看
        print("[warn] ffmpeg 转码失败（%r），保留 mp4v 原件" % exc)
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))


class Recorder:
    def __init__(self, path: str, fps: float = 30.0):
        self.path = Path(path)
        self.tmp = self.path.with_name("_tmp_" + self.path.name)
        self.fps = fps
        self.writer = None
        self.frames = 0

    def write(self, frame: np.ndarray):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if hasattr(cv2, "VideoWriter_fourcc"):
                code = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                code = cv2.VideoWriter.fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.tmp), code, float(self.fps), (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("VideoWriter 打不开：%s" % self.tmp)
        self.writer.write(frame)
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            _to_h264(self.tmp, self.path)


# --------------------------------------------------------------------------
# 四、命令行
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="工业相机 / 视频 + YOLO 装甲板实时识别",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--weights", default="weights/best.pt", help="模型权重（.pt）")
    p.add_argument("--source", choices=["mvs", "video", "test"], default="mvs",
                   help="mvs=真实相机，video=视频文件，test=合成画面")
    p.add_argument("--video", default="data/armor_raw.mp4", help="--source video 时的视频路径")
    p.add_argument("--camera-index", type=int, default=0, help="多台相机时的序号")

    p.add_argument("--exposure", type=float, default=None, help="曝光时间，微秒")
    p.add_argument("--gain", type=float, default=None, help="增益，dB")
    p.add_argument("--pixel-format", default=None, help="相机原生格式：BayerRG8 / BGR8 / Mono8")
    p.add_argument("--fps", type=float, default=30.0, help="相机采集帧率")
    p.add_argument("--convert", choices=["sdk", "cv2"], default="sdk",
                   help="Bayer→BGR 用谁转（sdk 快 / cv2 用于对比实验）")
    p.add_argument("--timeout-ms", type=int, default=2000, help="取一帧的超时")

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="0", help="0=第一块 GPU，cpu=CPU")
    p.add_argument("--half", action="store_true", help="FP16 推理（仅 CUDA 有效）")
    p.add_argument("--warmup", type=int, default=3,
                   help="正式统计前先空跑 N 次推理（第一次推理含 CUDA 初始化，很慢）")

    p.add_argument("--display", dest="display", action="store_true", default=None)
    p.add_argument("--no-display", dest="display", action="store_false")
    p.add_argument("--record", default="", help="结果视频路径（留空不录）")
    p.add_argument("--record-fps", type=float, default=30.0)
    p.add_argument("--save-dir", default="", help="每隔 N 帧存一张带框的 jpg")
    p.add_argument("--save-every", type=int, default=120)
    p.add_argument("--max-frames", type=int, default=0, help="只处理 N 帧后退出（0=不限）")
    p.add_argument("--stats", default="", help="统计 JSON 输出路径（留空则只打印）")
    p.add_argument("--no-keypoints", action="store_true", help="不画关键点")
    p.add_argument("--window", default="armor", help="窗口标题（英文）")
    return p.parse_args()


def build_source(args):
    if args.source == "video":
        return VideoSource(args.video)
    if args.source == "test":
        return TestSource()
    src = MvsSource(args.camera_index, args.convert, args.timeout_ms)
    src.configure(exposure_us=args.exposure, gain_db=args.gain,
                  pixel_format=args.pixel_format, fps=args.fps)
    return src


def main() -> int:
    args = parse_args()
    if args.display is None:
        args.display = bool(os.environ.get("DISPLAY")) or os.name == "nt"

    if not Path(args.weights).exists():
        print("[error] 权重不存在：%s" % args.weights)
        return 2

    from ultralytics import YOLO

    model = YOLO(args.weights)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    kpt_shape = getattr(getattr(model, "model", None), "kpt_shape", None)
    print("[model] task=%s names=%s kpt_shape=%s" % (model.task, names, kpt_shape))

    stop_event = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_event.set())

    try:
        source = build_source(args)
        source.start()
    except Exception as exc:  # noqa: BLE001
        print("[error] 打开帧源失败：%s" % exc)
        sdk_finalize()
        return 2
    print("[source] %s" % json.dumps(source.info(), ensure_ascii=False))

    grabber = LatestFrameGrabber(source, stop_event).start()
    recorder = Recorder(args.record, args.record_fps) if args.record else None
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    half = args.half and str(args.device).lower() not in ("cpu", "-1")

    def predict(frame):
        return model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                             device=args.device, half=half, verbose=False)[0]

    # 预热：进程里第一次推理会创建 CUDA 上下文、cuDNN 选算法，可能要几秒到几十秒。
    # 不做预热的话，第一个样本会把 infer_ms_mean 拉高到几百毫秒，看起来像"推理很慢"。
    warmup_s = 0.0
    if args.warmup > 0:
        try:
            frame0, _ = grabber.get(timeout=5.0)
        except queue.Empty:
            frame0 = None
        if frame0 is not None:
            t_w = time.perf_counter()
            for _ in range(args.warmup):
                predict(frame0)
            warmup_s = time.perf_counter() - t_w
            grabber.captured = 0          # 预热期间取到的帧不算进统计
            print("[warmup] %d 次预热完成，用时 %.2f s（不计入下面的统计）"
                  % (args.warmup, warmup_s))

    stats = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "args": vars(args), "source": source.info(),
             "model": {"weights": args.weights, "task": model.task,
                       "names": names, "kpt_shape": list(kpt_shape) if kpt_shape else None}}
    counts, conf_sum = Counter(), 0.0
    processed = with_det = 0
    infer_list, grab_list, convert_list = [], [], []
    saved = 0
    fps_ema = 0.0
    last_frame_t = None
    t_begin = time.perf_counter()

    print("[run] 按 q 或 Ctrl+C 退出")
    error = None
    try:
        while not stop_event.is_set():
            try:
                frame, meta = grabber.get(timeout=0.5)
            except queue.Empty:
                if grabber.error is not None:
                    raise grabber.error
                if args.max_frames and processed >= args.max_frames:
                    break
                continue

            now = time.perf_counter()
            if last_frame_t is not None:
                inst_fps = 1.0 / max(1e-6, now - last_frame_t)
                fps_ema = inst_fps if fps_ema == 0 else fps_ema * 0.9 + inst_fps * 0.1
            last_frame_t = now

            t0 = time.perf_counter()
            result = predict(frame)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            dets = draw_detections(frame, result, names, show_kpts=not args.no_keypoints)

            processed += 1
            infer_list.append(infer_ms)
            grab_list.append(meta.get("grab_ms", 0.0))
            convert_list.append(meta.get("convert_ms", 0.0))
            if dets:
                with_det += 1
                for d in dets:
                    counts[d["cls"]] += 1
                    conf_sum += d["conf"]

            if args.display or recorder:
                draw_hud(frame, [
                    "FPS %.1f | infer %.1f ms" % (fps_ema, infer_ms),
                    "%dx%d %s" % (frame.shape[1], frame.shape[0], meta.get("pixel", "-")),
                    "det %d | frame %d" % (len(dets), processed),
                ])
            if recorder:
                recorder.write(frame)
            if save_dir and processed % max(1, args.save_every) == 0:
                cv2.imwrite(str(save_dir / ("frame_%05d.jpg" % processed)), frame)
                saved += 1

            if args.display:
                try:
                    cv2.imshow(args.window, frame)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    print("[warn] 无法创建窗口（%s），改为无显示模式继续" % exc)
                    args.display = False
                    key = -1
                if key in (ord("q"), 27):
                    break

            if processed % 30 == 0:
                now = time.perf_counter()
                print("[progress] %d 帧，近 %.1f FPS，累计目标 %d，检出帧 %d"
                      % (processed, processed / max(1e-6, now - t_begin), sum(counts.values()), with_det))

            if args.max_frames and processed >= args.max_frames:
                break
    except Exception as exc:  # noqa: BLE001
        # 任何异常都先落到 error，让 finally 把相机、录像、窗口都收干净，再汇报
        error = exc
        print("[error] %s" % exc)
    finally:
        stop_event.set()
        # 等取图线程退出再关相机：如果它正卡在 MV_CC_GetImageBuffer 里，
        # 另一个线程同时 CloseDevice/DestroyHandle 会让 SDK 崩溃。
        grabber.join(timeout=args.timeout_ms / 1000.0 + 2.0)
        if grabber.thread.is_alive():
            print("[warn] 取图线程未在超时内退出，跳过关闭相机（进程退出时由系统回收）")
        else:
            try:
                source.close()
            finally:
                sdk_finalize()
        if recorder:
            recorder.close()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_begin
    stats.update({
        "frames_processed": processed,
        "frames_with_detection": with_det,
        "detection_rate": round(with_det / processed, 4) if processed else 0.0,
        "per_class": dict(counts),
        "avg_conf": round(conf_sum / max(1, sum(counts.values())), 4),
        "elapsed_s": round(elapsed, 2),
        "process_fps": round(processed / elapsed, 2) if elapsed else 0.0,
        "captured_frames": grabber.captured,
        "infer_ms_mean": round(float(np.mean(infer_list)), 2) if infer_list else 0.0,
        "infer_ms_p50": round(float(np.percentile(infer_list, 50)), 2) if infer_list else 0.0,
        "infer_ms_p95": round(float(np.percentile(infer_list, 95)), 2) if infer_list else 0.0,
        "infer_ms_max": round(float(np.max(infer_list)), 2) if infer_list else 0.0,
        "warmup_frames": args.warmup,
        "warmup_s": round(warmup_s, 2),
        "grab_ms_mean": round(float(np.mean(grab_list)), 2) if grab_list else 0.0,
        "convert_ms_mean": round(float(np.mean(convert_list)), 2) if convert_list else 0.0,
        "saved_frames": saved,
    })
    if error is not None:
        stats["error"] = repr(error)
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print("[result]\n" + text)
    if args.stats:
        Path(args.stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.stats).write_text(text, encoding="utf-8")
        print("[done] 统计写入 %s" % args.stats)
    if args.record:
        print("[done] 结果视频 %s（%d 帧）" % (args.record, recorder.frames if recorder else 0))
    return 0 if error is None else 1


if __name__ == "__main__":
    sys.exit(main())
