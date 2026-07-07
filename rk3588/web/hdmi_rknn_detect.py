#!/usr/bin/env python3
"""HDMI RX + RKNN water-meter detection for RK3588.

This entry reads FPGA HDMI from rk_hdmirx (/dev/video73) with GStreamer
appsink, then reuses usb_rknn_detect.py preprocessing/postprocessing helpers.
"""

import argparse
import os
import sys
import time
from typing import Optional

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import usb_rknn_detect as det


class HDMIFrameSource:
    def __init__(self, device: str, width: int, height: int, fps: int, fmt: str = "BGR"):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.fmt = fmt.upper()
        self.pipeline = None
        self.sink = None

    def open(self):
        Gst.init(None)
        # rk_hdmirx reports BGR3 in v4l2; GStreamer exposes it as BGR.
        caps = f"video/x-raw,format={self.fmt},width={self.width},height={self.height},framerate={self.fps}/1"
        desc = (
            f"v4l2src device={self.device} ! {caps} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
        )
        self.pipeline = Gst.parse_launch(desc)
        self.sink = self.pipeline.get_by_name("sink")
        if self.sink is None:
            raise RuntimeError("failed to create appsink")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("failed to set HDMI capture pipeline to PLAYING")
        print(f"[INFO] HDMI pipeline: {desc}")

    def _check_bus_error(self):
        if self.pipeline is None:
            return
        bus = self.pipeline.get_bus()
        msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
        if msg is None:
            return
        err, debug = msg.parse_error() if msg.type == Gst.MessageType.ERROR else msg.parse_warning()
        level = "ERROR" if msg.type == Gst.MessageType.ERROR else "WARNING"
        print(f"[{level}] GStreamer: {err}; {debug}")
        if msg.type == Gst.MessageType.ERROR:
            raise RuntimeError(str(err))

    def read(self, timeout_sec: float = 2.0) -> Optional[np.ndarray]:
        if self.sink is None:
            raise RuntimeError("source is not open")
        self._check_bus_error()
        sample = self.sink.emit("try-pull-sample", int(timeout_sec * 1_000_000_000))
        if sample is None:
            self._check_bus_error()
            return None

        caps = sample.get_caps()
        st = caps.get_structure(0)
        ok_w, cap_w = st.get_int("width")
        ok_h, cap_h = st.get_int("height")
        if ok_w and ok_h:
            width, height = cap_w, cap_h
        else:
            width, height = self.width, self.height

        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            data = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = width * height * 3
            if data.size < expected:
                raise RuntimeError(f"short HDMI frame: got {data.size}, expected {expected}")
            frame = data[:expected].reshape((height, width, 3)).copy()
            return frame
        finally:
            buf.unmap(map_info)

    def close(self):
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.sink = None


def parse_args():
    p = argparse.ArgumentParser(description="FPGA HDMI RX RKNN water-meter detection")
    p.add_argument("--model", default="/home/demo/water_meter/module/best_640_fp.rknn")
    p.add_argument("--device", default="/dev/video73")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--input-width", type=int, default=640)
    p.add_argument("--input-height", type=int, default=640)
    p.add_argument("--input-layout", default="auto", choices=("auto", "nchw", "nhwc"))
    p.add_argument("--input-dtype", default="auto", choices=("auto", "uint8", "int8", "float32"))
    p.add_argument("--color", default="rgb", choices=("bgr", "rgb"), help="Color order expected by the model")
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--classes", type=int, default=1)
    p.add_argument("--labels", default="")
    p.add_argument("--no-display", action="store_true", help="Run headless and print stats only")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means forever")
    p.add_argument("--save-output", default="", help="Optional path to save the latest annotated frame")
    p.add_argument("--debug-output", action="store_true", help="Print RKNN output tensor summary once")
    p.add_argument("--print-every", type=int, default=30, help="Print one status line every N frames")
    return p.parse_args()


def draw_detections(frame, boxes, scores, classes, labels):
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box.astype(np.int32).tolist()
        cls_i = int(cls_id)
        cls_name = labels[cls_i] if cls_i < len(labels) else str(cls_i)
        text = f"{cls_name} {float(score):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, text, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[ERROR] RKNN model not found: {args.model}")
        return 1

    labels = det.load_labels(args.labels, args.classes)

    rknn = det.RKNNLite()
    ret = rknn.load_rknn(args.model)
    if ret != 0:
        print(f"[ERROR] load_rknn failed: {ret}")
        return 1
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"[ERROR] init_runtime failed: {ret}")
        rknn.release()
        return 1

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] HDMI input: {args.device} {args.width}x{args.height}@{args.fps}")

    input_w, input_h = int(args.input_width), int(args.input_height)
    input_layout = args.input_layout.lower()
    selected_dtype = args.input_dtype.lower()
    if input_layout == "auto" or selected_dtype == "auto":
        print("[INFO] Probing model input config...")
        probed = det.probe_input_config(rknn, input_w, input_h, input_layout, selected_dtype, args.model)
        if probed is None:
            print("[ERROR] Failed to probe model input config")
            rknn.release()
            return 1
        input_w, input_h, input_layout, selected_dtype = probed
    print(f"[INFO] RKNN input: {input_w}x{input_h}, layout={input_layout}, dtype={selected_dtype}")

    source = HDMIFrameSource(args.device, args.width, args.height, args.fps)
    try:
        source.open()
    except Exception as e:
        print(f"[ERROR] Cannot open HDMI input {args.device}: {e}")
        print("[HINT] If /dev/video73 is busy, stop the old HDMI preview first, for example: fuser -k /dev/video73")
        rknn.release()
        return 1

    frame_count = 0
    det_count_sum = 0
    t0 = time.time()
    last_t = t0
    fps_smooth = 0.0
    output_debug_printed = False
    last_frame = None

    try:
        while True:
            frame = source.read(timeout_sec=2.0)
            if frame is None:
                print("[WARN] HDMI frame timeout")
                continue

            src_h, src_w = frame.shape[:2]
            img, ratio, pad_x, pad_y = det.letterbox(frame, (input_h, input_w))
            if args.color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            model_in = det.build_model_input(img, input_layout)
            model_in = det.convert_input_dtype(model_in, selected_dtype)
            infer_t0 = time.time()
            outputs = rknn.inference(inputs=[model_in], data_type=selected_dtype, data_format=input_layout)
            infer_ms = (time.time() - infer_t0) * 1000.0

            if args.debug_output and (not output_debug_printed) and outputs is not None:
                print("[DEBUG] RKNN output summary:")
                for i, out in enumerate(outputs):
                    arr = np.array(out)
                    if arr.size:
                        print(f"  out{i}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.5f}, max={arr.max():.5f}")
                    else:
                        print(f"  out{i}: shape={arr.shape}, dtype={arr.dtype}, empty")
                output_debug_printed = True

            boxes, scores, classes = det.postprocess_auto(outputs, args.conf, args.iou, args.classes, input_h, input_w)
            if len(boxes) > 0:
                boxes = det.scale_boxes_to_src(boxes, ratio, pad_x, pad_y, src_w, src_h)

            frame_count += 1
            det_count_sum += int(len(boxes))
            now = time.time()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt) if fps_smooth > 0 else 1.0 / dt

            annotated = frame
            if (not args.no_display) or args.save_output:
                annotated = frame.copy()
                draw_detections(annotated, boxes, scores, classes, labels)
                max_score = det.estimate_max_score(outputs)
                cv2.putText(annotated, f"FPS: {fps_smooth:.1f}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(annotated, f"Infer: {infer_ms:.1f} ms", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(annotated, f"MaxScore: {max_score:.3f}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                last_frame = annotated

            if args.print_every > 0 and (frame_count == 1 or frame_count % args.print_every == 0):
                elapsed = max(now - t0, 1e-6)
                print(
                    f"[STAT] frame={frame_count} fps={frame_count/elapsed:.2f} "
                    f"smooth={fps_smooth:.2f} infer_ms={infer_ms:.1f} det={len(boxes)}"
                )

            if not args.no_display:
                cv2.imshow("FPGA HDMI RKNN Detection", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            if args.max_frames > 0 and frame_count >= args.max_frames:
                break
    except Exception as e:
        print(f"[ERROR] HDMI/RKNN loop stopped: {e}")
        if "busy" in str(e).lower() or "resource" in str(e).lower():
            print("[HINT] /dev/video73 is probably occupied by another preview/detect process.")
        return_code = 1
    else:
        return_code = 0
    finally:
        if args.save_output and last_frame is not None:
            cv2.imwrite(args.save_output, last_frame)
            print(f"[INFO] Saved annotated frame: {args.save_output}")
        source.close()
        cv2.destroyAllWindows()
        rknn.release()

    elapsed = max(time.time() - t0, 1e-6)
    print(f"[DONE] frames={frame_count}, avg_fps={frame_count/elapsed:.2f}, avg_det={det_count_sum/max(frame_count,1):.2f}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
