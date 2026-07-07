#!/usr/bin/env python3
"""Open and preview a USB camera on Linux (rk3588/Ubuntu)."""
# cd /home/demo/water_meter/code
# python3 usb_camera_view.py


import argparse
import glob
import os
import sys
import time

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV is not installed. Install with: pip install opencv-python")
    sys.exit(1)


def list_video_devices() -> list[str]:
    by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*"))
    real_paths = []
    for path in by_id_paths:
        try:
            real_paths.append(f"{path} -> {os.path.realpath(path)}")
        except OSError:
            pass

    plain_paths = sorted(glob.glob("/dev/video*"))
    merged = real_paths + [p for p in plain_paths if p not in {x.split(" -> ")[-1] for x in real_paths}]
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB camera preview tool")
    parser.add_argument("--device", default="/dev/video74", help="Camera device path or index, e.g. /dev/video74 or 0")
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--no-display", action="store_true", help="Run capture loop without GUI")
    parser.add_argument("--save-dir", default=".", help="Directory for snapshots")
    return parser.parse_args()


def parse_device(device_arg: str):
    if device_arg.isdigit():
        return int(device_arg)
    return device_arg


def main() -> int:
    args = parse_args()

    devices = list_video_devices()
    if devices:
        print("[INFO] Found video devices:")
        for dev in devices:
            print(f"  - {dev}")
    else:
        print("[WARN] No /dev/video* devices found.")

    device = parse_device(str(args.device))
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {args.device}")
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Camera opened: {args.device}")
    print(f"[INFO] Actual stream: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS")

    os.makedirs(args.save_dir, exist_ok=True)
    print("[INFO] Controls: press 's' to save snapshot, 'q' to quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame, retrying...")
                time.sleep(0.05)
                continue

            if not args.no_display:
                cv2.imshow("USB Camera", frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = -1
                # Light throttling in headless mode to avoid maxing CPU.
                time.sleep(0.001)

            if key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = os.path.join(args.save_dir, f"snapshot_{ts}.jpg")
                cv2.imwrite(out_path, frame)
                print(f"[INFO] Snapshot saved: {out_path}")
            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("[INFO] Camera closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
