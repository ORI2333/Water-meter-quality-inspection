#!/usr/bin/env python3
"""USB camera + RKNN (RKNNLite) realtime detection demo for RK3588."""

# python3 /home/demo/water_meter/code/usb_rknn_detect.py \
#   --model /home/demo/water_meter/module/best_640_fp.rknn \
#   --input-dtype float32 \
#   --input-layout auto \
#   --conf 0.01 \
#   --debug-output


import argparse
import os
import time
from typing import List, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB camera RKNN detection")
    parser.add_argument(
        "--model",
        default="/home/demo/water_meter/module/best_640_fp.rknn",
        help="Path to .rknn model file",
    )
    parser.add_argument(
        "--device",
        default="/dev/video74",
        help="Camera device path or index, e.g. /dev/video74 or 0",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera fps")
    parser.add_argument("--color", default="rgb", choices=("bgr", "rgb"), help="Input color order for model")
    parser.add_argument("--input-width", type=int, default=640, help="Model input width")
    parser.add_argument("--input-height", type=int, default=640, help="Model input height")
    parser.add_argument(
        "--input-layout",
        default="auto",
        choices=("auto", "nchw", "nhwc"),
        help="Model input tensor layout, auto will prefer nhwc",
    )
    parser.add_argument(
        "--input-dtype",
        default="auto",
        choices=("auto", "uint8", "int8", "float32"),
        help="RKNN input dtype, auto will try int8 then uint8 then float32",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--classes", type=int, default=1, help="Number of classes for YOLO-like output")
    parser.add_argument(
        "--labels",
        default="",
        help="Optional label file, one class name per line",
    )
    parser.add_argument("--debug-output", action="store_true", help="Print RKNN output tensor summary once")
    return parser.parse_args()


def parse_device(device_arg: str):
    if str(device_arg).isdigit():
        return int(device_arg)
    return device_arg


def pick_dtype_order(req_dtype: str, model_path: str) -> List[str]:
    if req_dtype != "auto":
        return [req_dtype]
    name = os.path.basename(model_path).lower()
    if ("_fp" in name) or ("float" in name) or ("fp16" in name) or ("fp32" in name):
        return ["float32", "uint8", "int8"]
    return ["uint8", "int8", "float32"]


def load_labels(path: str, num_classes: int) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [x.strip() for x in f if x.strip()]
        if labels:
            return labels
    return [f"cls_{i}" for i in range(num_classes)]


def letterbox(image: np.ndarray, new_shape: Tuple[int, int], color=(0, 0, 0)):
    ih, iw = image.shape[:2]
    h, w = new_shape
    r = min(w / iw, h / ih)

    new_unpad_w, new_unpad_h = int(round(iw * r)), int(round(ih * r))
    dw, dh = w - new_unpad_w, h - new_unpad_h
    dw /= 2
    dh /= 2

    if (iw, ih) != (new_unpad_w, new_unpad_h):
        image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, left, top


def xywh_to_xyxy(x):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = (xx2 - xx1).clip(0)
        h = (yy2 - yy1).clip(0)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def _squeeze_output(out: np.ndarray) -> np.ndarray:
    arr = np.array(out)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def _looks_like_rk_yolo11(outputs, num_classes: int) -> bool:
    if outputs is None or len(outputs) < 6 or (len(outputs) % 3) != 0:
        return False

    for i in range(0, len(outputs), 3):
        box = _squeeze_output(outputs[i])
        cls = _squeeze_output(outputs[i + 1])
        score_sum = _squeeze_output(outputs[i + 2])

        if box.ndim != 3 or cls.ndim != 3 or score_sum.ndim != 3:
            return False
        if box.shape[1:] != cls.shape[1:] or cls.shape[1:] != score_sum.shape[1:]:
            return False
        if box.shape[0] % 4 != 0 or score_sum.shape[0] != 1:
            return False
        if num_classes > 0 and cls.shape[0] != num_classes:
            return False

    return True


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _yolo11_dfl(position: np.ndarray) -> np.ndarray:
    channels, grid_h, grid_w = position.shape
    bins = channels // 4
    pos = position.reshape(4, bins, grid_h, grid_w)
    pos = _softmax(pos, axis=1)
    acc = np.arange(bins, dtype=np.float32).reshape(1, bins, 1, 1)
    return (pos * acc).sum(axis=1)


def _yolo11_box_process(position: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    grid_h, grid_w = position.shape[1:3]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    grid = np.stack((col, row), axis=0).astype(np.float32)
    stride = np.array([img_w // grid_w, img_h // grid_h], dtype=np.float32).reshape(2, 1, 1)

    position = _yolo11_dfl(position)
    box_xy = grid + 0.5 - position[0:2, :, :]
    box_xy2 = grid + 0.5 + position[2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=0)
    return xyxy.transpose(1, 2, 0).reshape(-1, 4)


def _flatten_chw(x: np.ndarray) -> np.ndarray:
    return x.transpose(1, 2, 0).reshape(-1, x.shape[0])


def postprocess_rk_yolo11(outputs, conf_thr: float, iou_thr: float, num_classes: int, img_h: int, img_w: int):
    boxes_per_branch = []
    cls_per_branch = []

    for i in range(0, len(outputs), 3):
        box = _squeeze_output(outputs[i]).astype(np.float32, copy=False)
        cls = _squeeze_output(outputs[i + 1]).astype(np.float32, copy=False)
        boxes_per_branch.append(_yolo11_box_process(box, img_h=img_h, img_w=img_w))
        cls_per_branch.append(_flatten_chw(cls))

    boxes = np.concatenate(boxes_per_branch, axis=0)
    cls_prob = np.concatenate(cls_per_branch, axis=0)
    classes = np.argmax(cls_prob, axis=1).astype(np.int32)
    scores = cls_prob[np.arange(len(classes)), classes]

    keep = scores >= conf_thr
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    all_keep = []
    for c in np.unique(classes):
        inds = np.where(classes == c)[0]
        kept = nms(boxes[inds], scores[inds], iou_thr)
        all_keep.extend(inds[k] for k in kept)

    all_keep = np.array(all_keep, dtype=np.int32)
    return boxes[all_keep], scores[all_keep], classes[all_keep]


def postprocess_auto(outputs, conf_thr: float, iou_thr: float, num_classes: int, img_h: int, img_w: int):
    if outputs is None:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    if _looks_like_rk_yolo11(outputs, num_classes):
        return postprocess_rk_yolo11(outputs, conf_thr, iou_thr, num_classes, img_h=img_h, img_w=img_w)

    candidates = []

    for out in outputs:
        arr = np.array(out)
        if arr.size == 0:
            continue
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 1:
            continue
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1) if arr.shape[0] <= 256 else arr.reshape(-1, arr.shape[-1])

        if arr.ndim == 2:
            # Common RKNN YOLOv8 output is [C, N], e.g. [8, 8400], transpose to [N, C].
            if arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
                arr = arr.T
            if arr.shape[1] >= 5:
                candidates.append(arr.astype(np.float32, copy=False))

    if not candidates:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    pred = np.concatenate(candidates, axis=0)

    # Format A: [x1,y1,x2,y2,score,class]
    if pred.shape[1] in (6, 7):
        box = pred[:, :4]
        score = pred[:, 4]
        cls = pred[:, 5].astype(np.int32)

        # If boxes look like xywh, convert to xyxy.
        if np.mean(box[:, 2] > box[:, 0]) < 0.5:
            box = xywh_to_xyxy(box)

        keep = score > conf_thr
        box, score, cls = box[keep], score[keep], cls[keep]
    else:
        # Format B1 (YOLO-like): [x,y,w,h,obj,cls0,cls1,...]
        # Format B2 (YOLOv8): [x,y,w,h,cls0,cls1,...] (no objectness)
        box = pred[:, :4]
        extra = pred[:, 4:]
        if extra.shape[1] == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

        noobj_prob = extra
        if noobj_prob.min() < 0.0 or noobj_prob.max() > 1.0:
            noobj_prob = 1.0 / (1.0 + np.exp(-np.clip(noobj_prob, -50.0, 50.0)))
        noobj_cls = np.argmax(noobj_prob, axis=1).astype(np.int32)
        noobj_score = noobj_prob[np.arange(len(noobj_cls)), noobj_cls]

        use_obj_branch = False
        obj_score = None
        obj_cls = None
        if extra.shape[1] >= 2:
            obj = extra[:, 0]
            obj_prob = extra[:, 1:]
            if obj_prob.min() < 0.0 or obj_prob.max() > 1.0:
                obj_prob = 1.0 / (1.0 + np.exp(-np.clip(obj_prob, -50.0, 50.0)))
            if obj.min() < 0.0 or obj.max() > 1.0:
                obj = 1.0 / (1.0 + np.exp(-np.clip(obj, -50.0, 50.0)))
            obj_cls = np.argmax(obj_prob, axis=1).astype(np.int32)
            obj_cls_score = obj_prob[np.arange(len(obj_cls)), obj_cls]
            obj_score = obj * obj_cls_score

            # Prefer explicit channel-count match; otherwise pick the branch with better top scores.
            if num_classes > 0 and extra.shape[1] == num_classes + 1:
                use_obj_branch = True
            elif num_classes > 0 and extra.shape[1] == num_classes:
                use_obj_branch = False
            else:
                use_obj_branch = float(np.percentile(obj_score, 95)) > float(np.percentile(noobj_score, 95)) * 1.1

        if use_obj_branch and obj_score is not None and obj_cls is not None:
            score = obj_score
            cls = obj_cls
        else:
            score = noobj_score
            cls = noobj_cls

        keep = score > conf_thr
        box, score, cls = box[keep], score[keep], cls[keep]
        box = xywh_to_xyxy(box)

    if len(box) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32)

    all_keep = []
    for c in np.unique(cls):
        inds = np.where(cls == c)[0]
        kept = nms(box[inds], score[inds], iou_thr)
        all_keep.extend(inds[k] for k in kept)

    all_keep = np.array(all_keep, dtype=np.int32)
    return box[all_keep], score[all_keep], cls[all_keep]


def estimate_max_score(outputs) -> float:
    if outputs is None:
        return 0.0
    if _looks_like_rk_yolo11(outputs, num_classes=0):
        max_score = 0.0
        for i in range(1, len(outputs), 3):
            arr = _squeeze_output(outputs[i])
            if arr.size == 0:
                continue
            max_score = max(max_score, float(np.max(arr)))
        return max_score

    max_score = 0.0
    for out in outputs:
        arr = np.array(out)
        if arr.size == 0:
            continue
        if arr.ndim >= 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            continue
        if arr.shape[0] <= 256 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] <= 4:
            continue
        cls_prob = arr[:, 4:]
        if cls_prob.size == 0:
            continue
        if cls_prob.min() < 0.0 or cls_prob.max() > 1.0:
            cls_prob = 1.0 / (1.0 + np.exp(-np.clip(cls_prob, -50.0, 50.0)))
        max_score = max(max_score, float(cls_prob.max()))
    return max_score


def scale_boxes_to_src(boxes, ratio, pad_x, pad_y, src_w, src_h):
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_w - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_h - 1)
    return boxes


def build_model_input(rgb_img: np.ndarray, layout: str):
    if layout == "nchw":
        return np.ascontiguousarray(np.transpose(rgb_img, (2, 0, 1))[None, ...])
    return np.ascontiguousarray(rgb_img[None, ...])


def try_infer_once(rknn, input_w: int, input_h: int, layout: str, dtype: str):
    dummy = np.zeros((input_h, input_w, 3), dtype=np.uint8)
    model_in = build_model_input(dummy, layout)
    if dtype == "float32":
        model_in = np.ascontiguousarray(model_in.astype(np.float32) / 255.0)
    elif dtype == "int8":
        model_in = np.ascontiguousarray(model_in.astype(np.int16) - 128).astype(np.int8, copy=False)
    out = rknn.inference(inputs=[model_in], data_type=dtype, data_format=layout)
    return out is not None


def probe_input_config(rknn, req_w: int, req_h: int, req_layout: str, req_dtype: str, model_path: str):
    dtype_order = pick_dtype_order(req_dtype, model_path)
    if req_layout == "auto":
        layout_order = ["nhwc", "nchw"]
    else:
        layout_order = [req_layout, "nhwc" if req_layout == "nchw" else "nchw"]
    size_candidates = [(req_w, req_h)]

    common_sizes = [(640, 640), (640, 480), (416, 416), (320, 320), (1280, 736), (960, 960)]
    for sz in common_sizes:
        if sz not in size_candidates:
            size_candidates.append(sz)

    for w, h in size_candidates:
        for layout in layout_order:
            for dtype in dtype_order:
                try:
                    if try_infer_once(rknn, w, h, layout, dtype):
                        return w, h, layout, dtype
                except Exception:
                    continue
    return None


def convert_input_dtype(model_input: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "float32":
        return np.ascontiguousarray((model_input.astype(np.float32) / 255.0).astype(np.float32, copy=False))
    if dtype == "int8":
        return np.ascontiguousarray((model_input.astype(np.int16) - 128).astype(np.int8, copy=False))
    return np.ascontiguousarray(model_input)


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels, args.classes)

    if not os.path.exists(args.model):
        print(f"[ERROR] RKNN model not found: {args.model}")
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

    input_h = int(args.input_height)
    input_w = int(args.input_width)
    input_layout = args.input_layout.lower()
    need_nchw = input_layout == "nchw"

    cap = cv2.VideoCapture(parse_device(args.device), cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {args.device}")
        rknn.release()
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Camera: {args.device}")
    print("[INFO] Controls: q=quit")
    if input_layout != "auto" and args.input_dtype != "auto":
        forced_dtype = args.input_dtype
        print(f"[INFO] Input config (fixed): width={input_w}, height={input_h}, layout={input_layout}, dtype={forced_dtype}")
    else:
        print("[INFO] Probing model input config...")
        probed = probe_input_config(rknn, input_w, input_h, input_layout, args.input_dtype, args.model)
        if probed is None:
            print("[ERROR] Failed to probe a valid model input config.")
            print("[HINT] Try a model-specific demo script or verify the model input preprocessing.")
            cap.release()
            cv2.destroyAllWindows()
            rknn.release()
            return 1
        input_w, input_h, input_layout, forced_dtype = probed
        need_nchw = input_layout == "nchw"
        print(f"[INFO] Input config: width={input_w}, height={input_h}, layout={input_layout}, dtype={forced_dtype}")

    prev_t = time.time()
    fps = 0.0
    selected_dtype = forced_dtype
    selected_format = input_layout
    infer_cfg_printed = False
    output_debug_printed = False

    def infer_with_fallback(model_input):
        nonlocal selected_dtype, selected_format, infer_cfg_printed

        # Fixed mode: do not fallback, avoid repeated mismatched probing.
        if input_layout != "auto" and args.input_dtype != "auto":
            inp = convert_input_dtype(model_input, selected_dtype)
            try:
                return rknn.inference(
                    inputs=[inp],
                    data_type=selected_dtype,
                    data_format=selected_format,
                )
            except Exception as e:
                print(f"[ERROR] Inference failed with fixed config: {e}")
                print(
                    f"[HINT] Current fixed config is layout={selected_format}, dtype={selected_dtype}, "
                    f"tensor_dtype={inp.dtype}, input={input_w}x{input_h}."
                )
                return None

        if selected_dtype is not None:
            try:
                inp = convert_input_dtype(model_input, selected_dtype)
                return rknn.inference(
                    inputs=[inp],
                    data_type=selected_dtype,
                    data_format=selected_format,
                )
            except Exception:
                selected_dtype = None
                selected_format = None

        dtype_order = pick_dtype_order(args.input_dtype, args.model)
        format_order = [input_layout]
        alt_format = "nhwc" if input_layout == "nchw" else "nchw"
        if alt_format not in format_order:
            format_order.append(alt_format)

        for dtype in dtype_order:
            inp = convert_input_dtype(model_input, dtype)

            for fmt in format_order:
                try:
                    out = rknn.inference(inputs=[inp], data_type=dtype, data_format=fmt)
                    if out is not None:
                        selected_dtype = dtype
                        selected_format = fmt
                        if not infer_cfg_printed:
                            print(f"[INFO] Inference config selected: dtype={dtype}, data_format={fmt or 'default'}")
                            infer_cfg_printed = True
                        return out
                except Exception:
                    continue

        return None

    def print_output_debug_once(outputs):
        nonlocal output_debug_printed
        if output_debug_printed or outputs is None:
            return
        parts = []
        for i, out in enumerate(outputs):
            arr = np.array(out)
            parts.append(f"out{i}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.5f}, max={arr.max():.5f}")
        if parts:
            print("[DEBUG] RKNN output summary:")
            for line in parts:
                print(f"  {line}")
        output_debug_printed = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            src_h, src_w = frame.shape[:2]
            img, ratio, pad_x, pad_y = letterbox(frame, (input_h, input_w))
            if args.color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            model_in = build_model_input(img, "nchw" if need_nchw else "nhwc")

            outputs = infer_with_fallback(model_in)
            if args.debug_output:
                print_output_debug_once(outputs)
            if outputs is None:
                cv2.putText(
                    frame,
                    "Inference failed (check input layout/dtype)",
                    (20, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            elif selected_dtype is not None:
                cv2.putText(
                    frame,
                    f"RKNN in: {selected_dtype}/{selected_format or 'default'}",
                    (20, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
            max_score = estimate_max_score(outputs)
            boxes, scores, classes = postprocess_auto(outputs, args.conf, args.iou, args.classes, input_h, input_w)

            if len(boxes) > 0:
                boxes = scale_boxes_to_src(boxes, ratio, pad_x, pad_y, src_w, src_h).astype(np.int32)
                for box, score, cls_id in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = box.tolist()
                    cls_name = labels[int(cls_id)] if int(cls_id) < len(labels) else str(int(cls_id))
                    text = f"{cls_name} {score:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"MaxScore: {max_score:.3f}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("RKNN Detection", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        rknn.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
