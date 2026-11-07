#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render.py —— 把推理结果画到画面上（检测框 / 类别 / 置信度 / 关键点 / 中心点 / HUD）

单独拆出来的原因：
  app.py（实时识别）和 lab.py（参数实验）都要画同一套东西。
  画法保持一致，实验截图和演示视频才可比。

注意：画面上的文字一律用英文/数字。
  OpenCV 的 putText 不支持中文（会显示成 ????），要写中文得用 PIL 贴图，没必要。
"""

from __future__ import annotations

import cv2
import numpy as np

# BGR 调色板，按类别 id 循环取色
PALETTE = [
    (0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255),
    (0, 255, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]


def _font_scale(frame: np.ndarray) -> float:
    """按画面大小自动调字号，1080p 和 640×480 都能看清。"""
    h, w = frame.shape[:2]
    return max(0.45, min(1.0, min(w, h) / 900.0))


def put_label(frame, text, org, color, scale, thickness=1):
    """带底色的小标签，避免文字和背景糊在一起。"""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                     max(1, thickness))
    x, y = org
    x = max(0, min(x, frame.shape[1] - tw - 4))
    y = max(th + 4, min(y, frame.shape[0] - 2))
    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + base - 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), max(1, thickness), cv2.LINE_AA)


def draw_detections(frame: np.ndarray, result, names: dict,
                    show_kpts: bool = True) -> list[dict]:
    """在 frame 上画检测结果，返回结构化列表（方便写 CSV / JSON）。

    每个元素：
        {"cls": 类别名, "conf": 置信度, "xyxy": [x1,y1,x2,y2],
         "center": (cx, cy), "kpts": [(x,y), ...] 或 []}
    """
    dets: list[dict] = []
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return dets

    scale = _font_scale(frame)
    kpts_all = getattr(result, "keypoints", None)

    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        color = PALETTE[cls_id % len(PALETTE)]
        label = "%s %.2f" % (names.get(cls_id, str(cls_id)), conf)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        put_label(frame, label, (x1, max(0, y1 - 6)), color, scale)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 12, 2)
        cv2.putText(frame, "(%d,%d)" % (cx, cy), (cx + 8, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, color, 1, cv2.LINE_AA)

        pts = []
        if show_kpts and kpts_all is not None and len(kpts_all.xy) > i:
            xy = kpts_all.xy[i].cpu().numpy() if hasattr(kpts_all.xy[i], "cpu") \
                else np.asarray(kpts_all.xy[i])
            if xy.ndim == 2 and xy.shape[0] >= 2:
                pts = [(float(px), float(py)) for px, py in xy]
                if len(pts) >= 4:
                    cv2.polylines(frame, [np.array(pts, np.int32)], True, color, 2,
                                  cv2.LINE_AA)
                for j, (px, py) in enumerate(pts):
                    cv2.circle(frame, (int(px), int(py)), 4, color, -1, cv2.LINE_AA)
                    cv2.putText(frame, "P%d" % (j + 1), (int(px) + 6, int(py) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.7, color, 1,
                                cv2.LINE_AA)

        dets.append({"cls": names.get(cls_id, str(cls_id)), "conf": conf,
                     "xyxy": [x1, y1, x2, y2], "center": (cx, cy), "kpts": pts})
    return dets


def draw_hud(frame: np.ndarray, lines: list[str]) -> None:
    """左上角信息条：FPS / 推理耗时 / 分辨率等。"""
    if not lines:
        return
    scale = _font_scale(frame)
    pad, lh = 8, int(22 * scale) + 6
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                for t in lines) + pad * 2
    h = lh * len(lines) + pad
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + lh * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 1, cv2.LINE_AA)
