#!/usr/bin/env python3
"""YOLO11-pose RKNN test for FPGA HDMI water-meter pointer dials."""

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import usb_rknn_detect as det
from hdmi_rknn_detect import HDMIFrameSource

LABELS = ["10^-1", "10^-2", "10^-3", "10^-4"]
COLORS = [(60, 220, 60), (60, 180, 255), (255, 180, 60), (255, 80, 180)]


@dataclass
class PoseDet:
    box: np.ndarray
    score: float
    cls_id: int
    kpts: np.ndarray


def pointer_angle_deg(center, tip):
    angle = math.degrees(math.atan2(tip[1] - center[1], tip[0] - center[0]))
    return angle + 360.0 if angle < 0 else angle


class AngleStabilizer:
    def __init__(self, deadband_deg=1.0, alpha=0.30):
        self.deadband_deg = max(0.0, float(deadband_deg))
        self.alpha = min(1.0, max(0.01, float(alpha)))
        self._angle_by_key = {}

    @staticmethod
    def _diff_deg(new_angle, old_angle):
        return (new_angle - old_angle + 180.0) % 360.0 - 180.0

    def update(self, key, raw_angle):
        old_angle = self._angle_by_key.get(key)
        if old_angle is None:
            stable = raw_angle % 360.0
        else:
            diff = self._diff_deg(raw_angle, old_angle)
            if abs(diff) < self.deadband_deg:
                stable = old_angle
            else:
                stable = (old_angle + self.alpha * diff) % 360.0
        self._angle_by_key[key] = stable
        return stable


def parse_args():
    p = argparse.ArgumentParser(description="YOLO11-pose RKNN water-meter HDMI test")
    p.add_argument("--model", default="/home/demo/water_meter/module/water_meter_yolo11n_pose_fp.rknn")
    p.add_argument("--image", default="", help="Run one image instead of HDMI")
    p.add_argument("--device", default="/dev/video73")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--input-width", type=int, default=640)
    p.add_argument("--input-height", type=int, default=640)
    p.add_argument("--input-layout", default="nhwc", choices=("auto", "nchw", "nhwc"))
    p.add_argument("--input-dtype", default="uint8", choices=("auto", "uint8", "int8", "float32"))
    p.add_argument("--color", default="rgb", choices=("bgr", "rgb"))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--print-every", type=int, default=30)
    p.add_argument("--save-output", default="/home/demo/water_meter/pose_test_output.jpg")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--debug-output", action="store_true")
    p.add_argument("--angle-deadband", type=float, default=1.0, help="Do not update displayed angle below this delta in degrees")
    p.add_argument("--angle-alpha", type=float, default=0.30, help="Displayed angle smoothing factor after the deadband")
    return p.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def nms(boxes, scores, iou_thr):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union
        order = order[np.where(iou <= iou_thr)[0] + 1]
    return keep


def unwrap_pose_output(outputs):
    if outputs is None or len(outputs) == 0:
        return None
    arrs = [np.asarray(o) for o in outputs if np.asarray(o).size]
    if not arrs:
        return None
    arr = arrs[0]
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1) if arr.shape[0] <= 64 else arr.reshape(-1, arr.shape[-1])
    if arr.shape[0] == 14 and arr.shape[1] != 14:
        arr = arr.T
    elif arr.shape[1] != 14 and arr.shape[0] > arr.shape[1]:
        pass
    elif arr.shape[1] != 14 and arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] < 14:
        return None
    return arr[:, :14].astype(np.float32, copy=False)


def postprocess_pose(outputs, conf_thr, iou_thr, input_w, input_h):
    pred = unwrap_pose_output(outputs)
    if pred is None or pred.size == 0:
        return []

    # YOLO11 pose export: x,y,w,h, cls0..cls3, center(x,y,score), tip(x,y,score)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:8]
    if cls_scores.min() < 0.0 or cls_scores.max() > 1.0:
        cls_scores = sigmoid(cls_scores)
    cls_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
    scores = cls_scores[np.arange(len(cls_ids)), cls_ids]

    keep = scores >= conf_thr
    if not np.any(keep):
        return []
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    kpts = pred[keep, 8:14].reshape(-1, 2, 3)
    if kpts[:, :, 2].min() < 0.0 or kpts[:, :, 2].max() > 1.0:
        kpts[:, :, 2] = sigmoid(kpts[:, :, 2])

    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, input_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, input_h - 1)

    selected = []
    for c in np.unique(cls_ids):
        inds = np.where(cls_ids == c)[0]
        for local_i in nms(boxes[inds], scores[inds], iou_thr):
            i = inds[local_i]
            selected.append(PoseDet(boxes[i], float(scores[i]), int(cls_ids[i]), kpts[i].copy()))
    selected.sort(key=lambda d: (d.cls_id, -d.score))
    return selected


def scale_pose_to_src(dets: List[PoseDet], ratio, pad_x, pad_y, src_w, src_h):
    out = []
    for d in dets:
        box = d.box.copy()
        box[[0, 2]] = (box[[0, 2]] - pad_x) / ratio
        box[[1, 3]] = (box[[1, 3]] - pad_y) / ratio
        box[[0, 2]] = box[[0, 2]].clip(0, src_w - 1)
        box[[1, 3]] = box[[1, 3]].clip(0, src_h - 1)
        kpts = d.kpts.copy()
        kpts[:, 0] = ((kpts[:, 0] - pad_x) / ratio).clip(0, src_w - 1)
        kpts[:, 1] = ((kpts[:, 1] - pad_y) / ratio).clip(0, src_h - 1)
        out.append(PoseDet(box, d.score, d.cls_id, kpts))
    return out


def draw_pose(frame, dets: List[PoseDet], fps=0.0, infer_ms=0.0, stabilizer: Optional[AngleStabilizer] = None):
    for d in dets:
        color = COLORS[d.cls_id % len(COLORS)]
        x1, y1, x2, y2 = d.box.astype(int).tolist()
        center = d.kpts[0, :2]
        tip = d.kpts[1, :2]
        raw_angle = pointer_angle_deg(center, tip)
        angle = stabilizer.update(d.cls_id, raw_angle) if stabilizer is not None else raw_angle
        radius = max(1.0, float(np.linalg.norm(tip - center)))
        rad = math.radians(angle)
        tip_draw = center + np.array([math.cos(rad) * radius, math.sin(rad) * radius], dtype=np.float32)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, tuple(center.astype(int)), 4, (0, 255, 255), -1)
        cv2.circle(frame, tuple(tip_draw.astype(int)), 5, (0, 0, 255), -1)
        cv2.line(frame, tuple(center.astype(int)), tuple(tip_draw.astype(int)), (255, 0, 0), 2)
        label = LABELS[d.cls_id] if d.cls_id < len(LABELS) else str(d.cls_id)
        text = f"{label} {d.score:.2f} {angle:.1f}deg"
        cv2.putText(frame, text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(frame, (0, 0), (470, 96), (0, 0, 0), -1)
    cv2.putText(frame, f"YOLO11-pose det={len(dets)} FPS={fps:.1f}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(frame, f"Infer={infer_ms:.1f}ms", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)


def prepare_input(frame, input_w, input_h, color, layout, dtype):
    img, ratio, pad_x, pad_y = det.letterbox(frame, (input_h, input_w))
    if color == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    model_in = det.build_model_input(img, layout)
    model_in = det.convert_input_dtype(model_in, dtype)
    return model_in, ratio, pad_x, pad_y


def infer_frame(rknn, frame, args, input_w, input_h, layout, dtype, debug=False):
    src_h, src_w = frame.shape[:2]
    model_in, ratio, pad_x, pad_y = prepare_input(frame, input_w, input_h, args.color, layout, dtype)
    t0 = time.time()
    outputs = rknn.inference(inputs=[model_in], data_type=dtype, data_format=layout)
    infer_ms = (time.time() - t0) * 1000.0
    if debug:
        print("[DEBUG] output summary:")
        for i, out in enumerate(outputs or []):
            arr = np.asarray(out)
            if arr.size:
                print(f"  out{i}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.5f}, max={arr.max():.5f}")
            else:
                print(f"  out{i}: shape={arr.shape}, dtype={arr.dtype}, empty")
    dets = postprocess_pose(outputs, args.conf, args.iou, input_w, input_h)
    return scale_pose_to_src(dets, ratio, pad_x, pad_y, src_w, src_h), infer_ms


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[ERROR] model not found: {args.model}")
        return 1
    rknn = RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print(f"[ERROR] load_rknn failed: {ret}")
        return 1
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"[ERROR] init_runtime failed: {ret}")
        return 1

    input_w, input_h = args.input_width, args.input_height
    layout, dtype = args.input_layout, args.input_dtype
    if layout == "auto" or dtype == "auto":
        probed = det.probe_input_config(rknn, input_w, input_h, layout, dtype, args.model)
        if probed is None:
            print("[ERROR] failed to probe input config")
            return 1
        input_w, input_h, layout, dtype = probed
    print(f"[INFO] model={args.model}")
    print(f"[INFO] input={input_w}x{input_h} {layout}/{dtype}")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"[ERROR] cannot read image: {args.image}")
            return 1
        dets, infer_ms = infer_frame(rknn, frame, args, input_w, input_h, layout, dtype, debug=True)
        print(f"[RESULT] detections={len(dets)} infer_ms={infer_ms:.2f}")
        for d in dets:
            c = d.kpts[0, :2]
            t = d.kpts[1, :2]
            ang = pointer_angle_deg(c, t)
            print(f"  {LABELS[d.cls_id]} score={d.score:.3f} box={d.box.round(1).tolist()} center={c.round(1).tolist()} tip={t.round(1).tolist()} angle={ang:.1f}")
        out = frame.copy()
        draw_pose(out, dets, infer_ms=infer_ms, stabilizer=AngleStabilizer(args.angle_deadband, args.angle_alpha))
        if args.save_output:
            cv2.imwrite(args.save_output, out)
            print(f"[INFO] saved {args.save_output}")
        return 0

    source = HDMIFrameSource(args.device, args.width, args.height, args.fps)
    source.open()
    frame_count = 0
    t0 = time.time()
    last = t0
    fps_smooth = 0.0
    debug_once = args.debug_output
    angle_stabilizer = AngleStabilizer(args.angle_deadband, args.angle_alpha)
    try:
        while True:
            frame = source.read(timeout_sec=2.0)
            if frame is None:
                print("[WARN] HDMI timeout")
                continue
            dets, infer_ms = infer_frame(rknn, frame, args, input_w, input_h, layout, dtype, debug=debug_once)
            debug_once = False
            frame_count += 1
            now = time.time()
            dt = now - last
            last = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth else 1.0 / dt
            annotated = frame.copy()
            draw_pose(annotated, dets, fps=fps_smooth, infer_ms=infer_ms, stabilizer=angle_stabilizer)
            if args.save_output:
                cv2.imwrite(args.save_output, annotated)
            if args.print_every and (frame_count == 1 or frame_count % args.print_every == 0):
                print(f"[STAT] frame={frame_count} fps={frame_count/max(now-t0,1e-6):.2f} smooth={fps_smooth:.2f} infer={infer_ms:.1f}ms det={len(dets)}")
            if not args.no_display:
                cv2.imshow("FPGA HDMI YOLO11 Pose", annotated)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
            if args.max_frames and frame_count >= args.max_frames:
                break
    finally:
        source.close()
        rknn.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
