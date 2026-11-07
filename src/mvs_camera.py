#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mvs_camera.py —— 海康机器人（HIKROBOT）MVS 工业相机封装（Linux / WSL2）

为什么需要这一层？
  1. MVS 的 Python 接口是 ctypes 自动生成的：函数名长、参数多、错误码是十六进制数字。
     直接在业务代码里调用，十行有九行是样板代码，出了错也看不懂。
  2. 取图必须成对出现：MV_CC_GetImageBuffer() 借一帧 → MV_CC_FreeImageBuffer() 还一帧。
     漏还一次，SDK 的缓冲池就少一块；跑一会儿就再也取不到图（而且不会报"内存泄漏"）。
     所以要把它包在 try/finally 里，保证异常时也归还。
  3. 参数设置有顺序：先关自动曝光/自动增益，手动值才生效；改像素格式必须在停止取流时。

上层代码只需要记住三件事：
    cam.open()                          # 打开第 0 台相机
    cam.configure(exposure_us=1000, ...)# 设置参数（会自动读回确认）
    frame = cam.grab(); img = cam.to_bgr(frame)

单独运行本文件可以列出所有相机：
    python src/mvs_camera.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from ctypes import POINTER, byref, c_char_p, c_ubyte, cast, memset, sizeof
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# --------------------------------------------------------------------------
# 一、MVS SDK 的加载路径
# --------------------------------------------------------------------------
# MvCameraControl_class.py 内部会用 os.getenv('MVCAM_COMMON_RUNENV') 去拼
#     $MVCAM_COMMON_RUNENV/64/libMvCameraControl.so
# 所以这两个环境变量必须在 import 之前设好；MvImport 目录则要在 sys.path 里。
# 如果你把 MVS 装在了别处，运行时用环境变量覆盖即可：
#     MVS_ROOT=/path/to/MVS python src/mvs_camera.py
MVS_ROOT = os.environ.get("MVS_ROOT") or os.environ.get("MVCAM_SDK_PATH") or "/opt/MVS"
os.environ["MVCAM_SDK_PATH"] = MVS_ROOT
os.environ["MVCAM_COMMON_RUNENV"] = os.path.join(MVS_ROOT, "lib")
_MVIMPORT = os.path.join(MVS_ROOT, "Samples", "64", "Python", "MvImport")
if _MVIMPORT not in sys.path:
    sys.path.insert(0, _MVIMPORT)

try:
    from MvCameraControl_class import *  # noqa: F401,F403
except Exception as exc:  # pragma: no cover - 环境问题，直接给出可操作的提示
    raise SystemExit(
        "无法加载 MVS Python 接口：%r\n"
        "请依次确认：\n"
        "  1) MVS 是否装在 %s（ls /opt/MVS 应能看到 bin/ lib/ Samples/）\n"
        "  2) 是否存在 %s\n"
        "  3) 是否装的是 Linux x86_64 版（Windows 版的 .py 在 WSL 里加载不了 .dll）"
        % (exc, MVS_ROOT, _MVIMPORT)
    )

import cv2  # noqa: E402  （放在 SDK 后面，避免导入顺序干扰）

# --------------------------------------------------------------------------
# 二、错误码 → 文字
# --------------------------------------------------------------------------
# MvErrorDefine_const.py 里所有 MV_E_XXX 都是整数常量，这里反过来做成 码 → 名字 的字典。
# 好处：SDK 升级后新增的错误码会自动带上名字，不用手抄表格。
_ERROR_TEXT = {v: k for k, v in globals().items()
               if k.startswith("MV_E_") and isinstance(v, int)}


def error_text(code: int) -> str:
    """把 SDK 返回的十六进制错误码翻译成人能看懂的名字。"""
    code &= 0xFFFFFFFF
    return _ERROR_TEXT.get(code, "未知错误码")


class CameraError(RuntimeError):
    """MVS 调用失败。message 里已经带上了错误码的中文/英文含义。"""

    def __init__(self, message: str, code: Optional[int] = None):
        self.code = code
        if code is not None:
            message = "%s → %s（ret=0x%08X）" % (message, error_text(code), code & 0xFFFFFFFF)
        super().__init__(message)


class CameraTimeout(CameraError):
    """取流超时：在 timeout 内没有拿到完整帧（曝光太长、相机在触发模式、或掉线）。"""


# --------------------------------------------------------------------------
# 三、常用像素格式
# --------------------------------------------------------------------------
def _pt(name: str, fallback: int) -> int:
    """优先用 SDK 头文件里的常量；万一某个版本没导出，用标准值兜底。"""
    return globals().get("PixelType_Gvsp_" + name, fallback)


PT_MONO8 = _pt("Mono8", 0x01080001)
PT_BAYER_RG8 = _pt("BayerRG8", 0x01080009)
PT_BAYER_GR8 = _pt("BayerGR8", 0x01080008)
PT_BAYER_GB8 = _pt("BayerGB8", 0x0108000A)
PT_BAYER_BG8 = _pt("BayerBG8", 0x0108000B)
PT_RGB8 = _pt("RGB8_Packed", 0x02180014)
PT_BGR8 = _pt("BGR8_Packed", 0x02180015)

PIXEL_TYPES = {
    "Mono8": PT_MONO8,
    "BayerRG8": PT_BAYER_RG8,
    "BayerGR8": PT_BAYER_GR8,
    "BayerGB8": PT_BAYER_GB8,
    "BayerBG8": PT_BAYER_BG8,
    "RGB8": PT_RGB8,
    "BGR8": PT_BGR8,
}
_PIXEL_NAMES = {v: k for k, v in PIXEL_TYPES.items()}

# OpenCV 的去马赛克函数。BayerRG8 = 第一行前两个像素是 R、G（RGGB）。
_BAYER_TO_CV = {
    PT_BAYER_RG8: cv2.COLOR_BayerRG2BGR,
    PT_BAYER_GR8: cv2.COLOR_BayerGR2BGR,
    PT_BAYER_GB8: cv2.COLOR_BayerGB2BGR,
    PT_BAYER_BG8: cv2.COLOR_BayerBG2BGR,
}

# 有些相机把 BGR8 这个枚举叫 "BGR8Packed"，这里给几个别名依次尝试。
_PIXEL_ALIASES = {
    "BGR8": ["BGR8", "BGR8Packed", "BGR8_Packed"],
    "RGB8": ["RGB8", "RGB8Packed", "RGB8_Packed"],
    "Mono8": ["Mono8"],
    "BayerRG8": ["BayerRG8"],
    "BayerGR8": ["BayerGR8"],
    "BayerGB8": ["BayerGB8"],
    "BayerBG8": ["BayerBG8"],
}

# 枚举设备时用到的传输层。GigE = 网口相机，USB = USB3.0 相机。
_TL_TYPES = MV_GIGE_DEVICE | MV_USB_DEVICE
for _name in ("MV_GENTL_GIGE_DEVICE", "MV_GENTL_CAMERALINK_DEVICE",
              "MV_GENTL_CXP_DEVICE", "MV_GENTL_XOF_DEVICE", "MV_GENTL_XOC_DEVICE"):
    _TL_TYPES |= globals().get(_name, 0)

_sdk_ready = False
_open_count = 0


def sdk_init() -> None:
    """初始化 SDK（只需一次）。"""
    global _sdk_ready
    if not _sdk_ready:
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            raise CameraError("MV_CC_Initialize 失败", ret)
        _sdk_ready = True


def sdk_finalize() -> None:
    """反初始化 SDK。只有确认所有相机都关了才调用，否则可能在驱动层崩溃。"""
    global _sdk_ready
    if _sdk_ready and _open_count == 0:
        try:
            MvCamera.MV_CC_Finalize()
        except Exception:
            pass
        _sdk_ready = False


# --------------------------------------------------------------------------
# 四、数据结构
# --------------------------------------------------------------------------
@dataclass
class Device:
    """一台相机的静态信息（不用打开就能读到）。"""
    index: int
    kind: str            # "GigE" 或 "USB"
    model: str
    serial: str
    version: str
    ip: str = ""
    mac: str = ""

    def summary(self) -> str:
        line = "[%d] %-4s %-26s SN=%s" % (self.index, self.kind,
                                          self.model or "(未知型号)", self.serial or "-")
        if self.kind == "GigE":
            line += "  IP=%s" % (self.ip or "-")
        return line


@dataclass
class RawFrame:
    """一帧原始图像：还没做 Bayer → BGR 转换，数据已经复制出 SDK 缓冲区。"""
    width: int
    height: int
    pixel_type: int
    data: bytes
    frame_num: int
    host_ts: float          # time.perf_counter()，用于算延迟
    dev_ts: int             # 相机内部时间戳（微秒）
    lost_packet: int = 0

    @property
    def pixel_name(self) -> str:
        return _PIXEL_NAMES.get(self.pixel_type, "0x%08X" % self.pixel_type)

    @property
    def nbytes(self) -> int:
        return len(self.data)


def _decode_char_array(chars) -> str:
    """ctypes 的 c_char 数组 → Python 字符串（相机里的字符串可能是 GBK）。"""
    raw = bytes(memoryview(chars)).split(b"\x00")[0]
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _enum_raw():
    """枚举设备，返回 [(Device, MV_CC_DEVICE_INFO), ...] 以及设备列表本体。

    设备列表本体（MV_CC_DEVICE_INFO_LIST）必须由调用方保活：
    Device 里的 pDeviceInfo 指向它内部的内存，列表一释放就成了野指针。
    """
    sdk_init()
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(_TL_TYPES, dev_list)
    if ret != 0:
        raise CameraError("MV_CC_EnumDevices 失败", ret)

    items = []
    for i in range(dev_list.nDeviceNum):
        info = cast(dev_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == MV_GIGE_DEVICE:
            gi = info.SpecialInfo.stGigEInfo
            dev = Device(i, "GigE", _decode_char_array(gi.chModelName),
                         _decode_char_array(gi.chSerialNumber),
                         _decode_char_array(gi.chDeviceVersion),
                         ip=_ip_str(gi.nCurrentIp),
                         mac=":".join("%02X" % b for b in bytes(gi.chMacAddress)))
        elif info.nTLayerType == MV_USB_DEVICE:
            ui = info.SpecialInfo.stUsb3VInfo
            dev = Device(i, "USB", _decode_char_array(ui.chModelName),
                         _decode_char_array(ui.chSerialNumber),
                         _decode_char_array(ui.chDeviceVersion))
        else:
            # GenTL 之类的其它传输层：字段位置不同，只记下类型
            dev = Device(i, "GenTL", "", "", "")
        items.append((dev, info))
    return items, dev_list


def _ip_str(n: int) -> str:
    n &= 0xFFFFFFFF
    return "%d.%d.%d.%d" % ((n >> 24) & 0xFF, (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)


def list_devices() -> list[Device]:
    """只列设备，不打开。"""
    items, _keep = _enum_raw()
    return [d for d, _ in items]


# --------------------------------------------------------------------------
# 五、相机对象
# --------------------------------------------------------------------------
class Camera:
    """一台 MVS 相机。

    典型用法：
        with Camera() as cam:
            cam.configure(exposure_us=1000, gain_db=0, pixel_format="BayerRG8", fps=30)
            cam.start()
            frame = cam.grab()
            img = cam.to_bgr(frame)
    """

    def __init__(self, index: int = 0, verbose: bool = True):
        self.index = index
        self.verbose = verbose
        self.device: Optional[Device] = None
        self.last_convert_ms = 0.0

        self._cam = None
        self._dev_list = None      # 保活：见 _enum_raw 的说明
        self._opened = False
        self._grabbing = False
        self._dst_buf = None       # SDK 转换的复用的输出缓冲

    # ---------------- 生命周期 ----------------
    def open(self) -> "Camera":
        global _open_count
        items, dev_list = _enum_raw()
        if not items:
            raise CameraError(
                "没有枚举到任何相机。按相机类型排查：\n"
                "  网口(GigE)相机：WSL 必须用镜像网络（.wslconfig 里 networkingMode=mirrored），"
                "且 Windows 网卡与相机在同一网段\n"
                "  USB3 相机：Windows 里 usbipd attach --wsl 之后，WSL 的 lsusb 应能看到设备"
            )
        if self.index >= len(items):
            raise CameraError("设备序号越界：只找到 %d 台" % len(items))

        dev, info = items[self.index]
        self.device = dev
        self._dev_list = dev_list

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise CameraError("MV_CC_CreateHandle 失败", ret)

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise CameraError(
                "MV_CC_OpenDevice 失败（常见原因：MVS 客户端/另一个 Python 进程正占着相机，"
                "或 USB 相机权限不足）", ret)

        self._cam = cam
        self._opened = True
        _open_count += 1

        # GigE：自动协商最佳包长，能显著减少丢包
        if dev.kind == "GigE":
            size = int(cam.MV_CC_GetOptimalPacketSize())
            if size > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", size)

        # 必须关触发，否则不按软/硬触发就永远没有图
        self.set_enum("TriggerMode", 0, ignore_error=True)  # 0 = Off

        if self.verbose and dev:
            print("[camera] 已打开 %s" % dev.summary())
        return self

    def close(self) -> None:
        """按 停止取流 → 关闭设备 → 销毁句柄 的顺序释放，异常也不跳过后续步骤。"""
        global _open_count
        cam, self._cam = self._cam, None
        if cam is None:
            return
        try:
            if self._grabbing:
                try:
                    cam.MV_CC_StopGrabbing()
                finally:
                    self._grabbing = False
        finally:
            try:
                if self._opened:
                    try:
                        cam.MV_CC_CloseDevice()
                    finally:
                        self._opened = False
            finally:
                cam.MV_CC_DestroyHandle()
                _open_count = max(0, _open_count - 1)

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------------- 参数：读 ----------------
    def get_float(self, key: str) -> tuple[float, float, float]:
        """返回 (当前值, 最小值, 最大值)。"""
        st = MVCC_FLOATVALUE()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetFloatValue(key, st)
        if ret != 0:
            raise CameraError("读取 %s 失败" % key, ret)
        return float(st.fCurValue), float(st.fMin), float(st.fMax)

    def get_int(self, key: str) -> int:
        st = MVCC_INTVALUE_EX()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetIntValueEx(key, st)
        if ret != 0:
            raise CameraError("读取 %s 失败" % key, ret)
        return int(st.nCurValue)

    def get_enum(self, key: str) -> int:
        st = MVCC_ENUMVALUE()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetEnumValue(key, st)
        if ret != 0:
            raise CameraError("读取 %s 失败" % key, ret)
        return int(st.nCurValue)

    def get_string(self, key: str) -> str:
        st = MVCC_STRINGVALUE()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetStringValue(key, st)
        if ret != 0:
            raise CameraError("读取 %s 失败" % key, ret)
        return _decode_char_array(st.chCurValue)

    def read_device_strings(self) -> dict:
        """读型号/序列号/固件版本——实验记录里要求的字段。"""
        out = {}
        for key in ("DeviceModelName", "DeviceSerialNumber", "DeviceFirmwareVersion",
                    "DeviceVersion", "DeviceVendorName", "DeviceUserID"):
            try:
                value = self.get_string(key)
                if value:
                    out[key] = value
            except CameraError:
                pass
        return out

    # ---------------- 参数：写 ----------------
    def set_float(self, key: str, value: float, clamp: bool = True) -> float:
        """设置浮点参数并读回确认；超范围时自动夹到 [min, max] 并提示。"""
        lo = hi = None
        if clamp:
            try:
                _, lo, hi = self.get_float(key)
                value = min(max(value, lo), hi)
            except CameraError:
                pass
        ret = self._cam.MV_CC_SetFloatValue(key, float(value))
        if ret != 0:
            raise CameraError("设置 %s=%s 失败" % (key, value), ret)
        actual = self.get_float(key)[0]
        if abs(actual - value) > max(1e-3, abs(value) * 1e-3):
            print("[warn] %s 设定 %.3f，实际读回 %.3f（相机可能按步进取整）"
                  % (key, value, actual))
        return actual

    def set_int(self, key: str, value: int) -> int:
        ret = self._cam.MV_CC_SetIntValue(key, int(value))
        if ret != 0:
            raise CameraError("设置 %s=%s 失败" % (key, value), ret)
        return self.get_int(key)

    def set_enum(self, key: str, value: int, ignore_error: bool = False) -> None:
        ret = self._cam.MV_CC_SetEnumValue(key, int(value))
        if ret != 0 and not ignore_error:
            raise CameraError("设置 %s=%s 失败" % (key, value), ret)

    def set_enum_str(self, key: str, value: str, ignore_error: bool = False) -> bool:
        ret = self._cam.MV_CC_SetEnumValueByString(key, value)
        if ret != 0:
            if ignore_error:
                return False
            raise CameraError("设置 %s=%s 失败" % (key, value), ret)
        return True

    def set_bool(self, key: str, value: bool, ignore_error: bool = True) -> bool:
        ret = self._cam.MV_CC_SetBoolValue(key, bool(value))
        if ret != 0 and not ignore_error:
            raise CameraError("设置 %s=%s 失败" % (key, value), ret)
        return ret == 0

    # ---------------- 常用参数的语义封装 ----------------
    def set_exposure(self, exposure_us: float) -> float:
        """曝光时间，单位微秒。先关自动曝光，再设手动值。"""
        self.set_enum_str("ExposureAuto", "Off", ignore_error=True)
        for key in ("ExposureTime", "ExposureTimeAbs"):
            try:
                return self.set_float(key, exposure_us)
            except CameraError:
                continue
        raise CameraError("设置曝光失败：相机既没有 ExposureTime 也没有 ExposureTimeAbs 节点")

    def set_gain(self, gain_db: float) -> float:
        """增益，单位 dB。先关自动增益。"""
        self.set_enum_str("GainAuto", "Off", ignore_error=True)
        return self.set_float("Gain", gain_db)

    def set_fps(self, fps: float) -> float:
        """固定采集帧率（打开帧率控制开关，否则相机跑最大帧率）。"""
        self.set_bool("AcquisitionFrameRateEnable", True)
        return self.set_float("AcquisitionFrameRate", fps)

    def set_pixel_format(self, name: str) -> int:
        """设置相机原生输出格式。必须在停止取流时调用。"""
        if self._grabbing:
            raise CameraError("请先 stop()：取流过程中不能改像素格式")
        if name not in PIXEL_TYPES:
            raise CameraError("不认识的像素格式 %r，可选：%s"
                              % (name, ", ".join(PIXEL_TYPES)))
        last = None
        for alias in _PIXEL_ALIASES.get(name, [name]):
            ret = self._cam.MV_CC_SetEnumValueByString("PixelFormat", alias)
            if ret == 0:
                cur = self.get_enum("PixelFormat")
                if cur != PIXEL_TYPES[name]:
                    print("[warn] PixelFormat 设为 %s 后读回 0x%08X，与预期不符" % (alias, cur))
                return cur
            last = ret
        raise CameraError("相机不支持像素格式 %s（可能只支持 Bayer/Mono）" % name, last)

    def set_bayer_quality(self, quality: int) -> None:
        """SDK 去马赛克的质量档位（0 快速 / 1 均衡 / 2 最优 / 3 最优+），会影响转换耗时。"""
        ret = self._cam.MV_CC_SetBayerCvtQuality(int(quality))
        if ret != 0:
            raise CameraError("设置 Bayer 转换质量失败", ret)

    def configure(self, *, exposure_us=None, gain_db=None, pixel_format=None,
                  fps=None, bayer_quality=None) -> "Camera":
        """一次性设置一组参数（顺序已按依赖关系排好）。"""
        if self._grabbing:
            raise CameraError("configure() 必须在 start() 之前调用")
        if pixel_format is not None:
            self.set_pixel_format(pixel_format)
        if bayer_quality is not None:
            self.set_bayer_quality(bayer_quality)
        if fps is not None:
            try:
                self.set_fps(fps)
            except CameraError as exc:
                # 个别相机没有帧率控制节点，固定不了就按相机自己的最大帧率跑
                print("[warn] 设置帧率失败（%s），继续用相机默认帧率" % exc)
        if exposure_us is not None:
            self.set_exposure(exposure_us)
        if gain_db is not None:
            self.set_gain(gain_db)
        return self

    # ---------------- 取流 ----------------
    def start(self) -> "Camera":
        if not self._opened:
            raise CameraError("相机还没打开")
        if not self._grabbing:
            ret = self._cam.MV_CC_StartGrabbing()
            if ret != 0:
                raise CameraError("MV_CC_StartGrabbing 失败", ret)
            self._grabbing = True
        return self

    def stop(self) -> None:
        if self._grabbing and self._cam is not None:
            try:
                self._cam.MV_CC_StopGrabbing()
            finally:
                self._grabbing = False

    def grab(self, timeout_ms: int = 2000) -> RawFrame:
        """取一帧。超时抛 CameraTimeout，其它错误抛 CameraError。

        注意：这里已经把图像数据复制到 Python 的 bytes 里，然后立刻归还 SDK 缓冲，
        所以即使后面转换/推理很慢，也不会拖住相机的采集线程。
        """
        if not self._grabbing:
            raise CameraError("还没开始取流：先调用 start()")

        st = MV_FRAME_OUT()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetImageBuffer(st, int(timeout_ms))
        if ret != 0:
            if ret in (MV_E_NODATA, MV_E_TIMEOUT, MV_E_GC_TIMEOUT):
                raise CameraTimeout("取流超时（timeout=%d ms）" % timeout_ms, ret)
            raise CameraError("MV_CC_GetImageBuffer 失败", ret)

        try:
            info = st.stFrameInfo
            n = int(info.nFrameLen)
            data = ctypes.string_at(st.pBufAddr, n) if n else b""
            if not data:
                raise CameraError("拿到空帧（nFrameLen=0）")
            return RawFrame(
                width=int(info.nWidth),
                height=int(info.nHeight),
                pixel_type=int(info.enPixelType),
                data=data,
                frame_num=int(info.nFrameNum),
                host_ts=time.perf_counter(),
                dev_ts=(int(info.nDevTimeStampHigh) << 32) | int(info.nDevTimeStampLow),
                lost_packet=int(getattr(info, "nLostPacket", 0)),
            )
        finally:
            # 关键：无论上面发生什么，都要把这一帧还给 SDK
            self._cam.MV_CC_FreeImageBuffer(st)

    # ---------------- 教学/诊断用的低层接口 ----------------
    def grab_unreleased(self, timeout_ms: int = 2000):
        """借一帧但**不归还**，仅供"漏归还缓冲会怎样"的实验使用。

        正常代码请一律用 grab()，它内部用 try/finally 保证归还。
        """
        st = MV_FRAME_OUT()
        memset(byref(st), 0, sizeof(st))
        ret = self._cam.MV_CC_GetImageBuffer(st, int(timeout_ms))
        if ret != 0:
            if ret in (MV_E_NODATA, MV_E_TIMEOUT, MV_E_GC_TIMEOUT):
                raise CameraTimeout("取流超时（timeout=%d ms）" % timeout_ms, ret)
            raise CameraError("MV_CC_GetImageBuffer 失败", ret)
        return st

    def release_frame(self, frame_out) -> None:
        """归还 grab_unreleased() 借出的帧。"""
        self._cam.MV_CC_FreeImageBuffer(frame_out)

    def try_set_enum_str(self, key: str, value: str) -> int:
        """设置枚举但**不抛异常**，返回原始错误码（实验里用来演示"错误格式"）。"""
        return int(self._cam.MV_CC_SetEnumValueByString(key, value))

    # ---------------- 格式转换 ----------------
    def to_bgr(self, frame: RawFrame, mode: str = "sdk") -> np.ndarray:
        """原始帧 → OpenCV BGR 图像。

        mode:
          "sdk"  用 MVS 自带的 MV_CC_ConvertPixelTypeEx 转换（SIMD 优化，通常最快）
          "cv2"  Bayer 数据交给 cv2.cvtColor 去马赛克（用来对比"转换流程"的耗时）
        """
        t0 = time.perf_counter()
        img = self._to_bgr(frame, mode)
        self.last_convert_ms = (time.perf_counter() - t0) * 1000.0
        return img

    def _to_bgr(self, frame: RawFrame, mode: str) -> np.ndarray:
        pt, w, h = frame.pixel_type, frame.width, frame.height

        if pt == PT_BGR8:  # 相机直出 BGR，不需要转换
            return np.frombuffer(frame.data, np.uint8).reshape(h, w, 3).copy()

        if pt == PT_RGB8:
            rgb = np.frombuffer(frame.data, np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if pt == PT_MONO8:
            gray = np.frombuffer(frame.data, np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if mode == "cv2" and pt in _BAYER_TO_CV:
            gray = np.frombuffer(frame.data, np.uint8).reshape(h, w)
            return cv2.cvtColor(gray, _BAYER_TO_CV[pt])

        # 其余情况（Bayer 用 SDK 转、或 10/12bit 等）都交给 SDK
        return self._convert_with_sdk(frame, PT_BGR8)

    def _convert_with_sdk(self, frame: RawFrame, dst_type: int) -> np.ndarray:
        w, h = frame.width, frame.height
        need = w * h * 3
        if self._dst_buf is None or sizeof(self._dst_buf) < need:
            self._dst_buf = (c_ubyte * need)()

        # 直接把 Python bytes 的地址交给 SDK（只读，零拷贝）。
        # frame.data 在本函数返回前一直有效，所以指针不会悬空。
        src = cast(c_char_p(frame.data), POINTER(c_ubyte))
        param = MV_CC_PIXEL_CONVERT_PARAM_EX()
        memset(byref(param), 0, sizeof(param))
        param.nWidth = w
        param.nHeight = h
        param.enSrcPixelType = frame.pixel_type
        param.pSrcData = src
        param.nSrcDataLen = len(frame.data)
        param.enDstPixelType = dst_type
        param.pDstBuffer = self._dst_buf
        param.nDstBufferSize = need

        ret = self._cam.MV_CC_ConvertPixelTypeEx(param)
        if ret != 0:
            raise CameraError("MV_CC_ConvertPixelTypeEx 失败（源格式 %s）"
                              % frame.pixel_name, ret)

        out = np.frombuffer(self._dst_buf, np.uint8, count=int(param.nDstLen))
        return out.reshape(h, w, 3).copy()

    # ---------------- 杂项 ----------------
    def snapshot_info(self) -> dict:
        """当前参数快照，写实验记录用。"""
        info: dict[str, Any] = {}
        if self.device:
            info.update({"kind": self.device.kind, "model": self.device.model,
                         "serial": self.device.serial, "device_version": self.device.version,
                         "ip": self.device.ip})
        try:
            info["exposure_us"] = round(self.get_float("ExposureTime")[0], 3)
        except CameraError:
            pass
        try:
            info["gain_db"] = round(self.get_float("Gain")[0], 3)
        except CameraError:
            pass
        try:
            pf = self.get_enum("PixelFormat")
            info["pixel_format"] = _PIXEL_NAMES.get(pf, "0x%08X" % pf)
        except CameraError:
            pass
        try:
            info["width"] = self.get_int("Width")
            info["height"] = self.get_int("Height")
        except CameraError:
            pass
        try:
            info["fps"] = round(self.get_float("AcquisitionFrameRate")[0], 2)
        except CameraError:
            pass
        try:
            info.update(self.read_device_strings())
        except Exception:
            pass
        return info


# --------------------------------------------------------------------------
# 六、命令行自检：python src/mvs_camera.py
# --------------------------------------------------------------------------
def main() -> int:
    print("MVS_ROOT = %s" % MVS_ROOT)
    try:
        devs = list_devices()
    except CameraError as exc:
        print("[error] %s" % exc)
        return 2
    if not devs:
        print("没有发现相机（0 台）。")
        print("  网口相机 → 检查 .wslconfig 的 networkingMode=mirrored 与网段")
        print("  USB 相机 → Windows 里 usbipd list / attach --wsl")
        return 1
    for dev in devs:
        print(dev.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
