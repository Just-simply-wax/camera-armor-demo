#!/usr/bin/env bash
# 环境自检：把输出整段发给别人，就能判断问题出在哪一层。
# 用法：bash tools/probe_env.sh

echo "================ 1. 系统 ================"
lsb_release -a 2>/dev/null || (cat /etc/os-release | head -3)
uname -r
echo

echo "================ 2. Python ================"
python3 -V
which python3
python3 -c "import sys; print(sys.executable)"
echo

echo "================ 3. MVS SDK ================"
if [ -d /opt/MVS ]; then
  echo "/opt/MVS 存在"
  ls /opt/MVS
  echo "--- 动态库 ---"
  ls -l /opt/MVS/lib/64/libMvCameraControl.so 2>/dev/null || echo "!! 找不到 libMvCameraControl.so"
  echo "--- Python 接口 ---"
  ls /opt/MVS/Samples/64/Python/MvImport 2>/dev/null | head -5 || echo "!! 找不到 MvImport"
else
  echo "!! /opt/MVS 不存在，MVS 没装或装到了别处"
fi
echo "MVCAM_SDK_PATH      = ${MVCAM_SDK_PATH:-<未设置>}"
echo "MVCAM_COMMON_RUNENV = ${MVCAM_COMMON_RUNENV:-<未设置>}"
echo

echo "================ 4. 显示（WSLg） ================"
echo "DISPLAY=${DISPLAY:-<空，无法弹窗>}"
ls /tmp/.X11-unix 2>/dev/null | head -3
echo

echo "================ 5. 网络与 USB ================"
echo "--- 网卡 ---"
ip -4 addr show 2>/dev/null | grep -E "^[0-9]+:|inet " | head -20
echo "--- 路由 ---"
ip route 2>/dev/null | head -5
echo "--- USB ---"
lsusb 2>/dev/null | head -20 || echo "没有 lsusb（sudo apt install usbutils）"
echo

echo "================ 6. Python 依赖 ================"
python3 - <<'PY'
mods = ["cv2", "numpy", "torch", "ultralytics"]
for name in mods:
    try:
        mod = __import__(name)
        extra = ""
        if name == "torch":
            extra = " cuda_available=%s" % mod.cuda.is_available()
        print("%-12s %s%s" % (name, getattr(mod, "__version__", "?"), extra))
    except Exception as exc:
        print("%-12s 未安装/导入失败: %r" % (name, exc))
PY
echo

echo "================ 7. 相机枚举 ================"
cd "$(dirname "$0")/.." 2>/dev/null || true
python3 src/mvs_camera.py 2>&1 | tail -20
