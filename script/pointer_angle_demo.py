#!/usr/bin/env python3
"""
Pure-OpenCV demo: estimate pointer angle from labeled dial ROIs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


POINTER_RE = re.compile(r"^\s*10\^-?\d+\s*$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = (script_dir.parent / "data" / "original_dataset").resolve()
    parser = argparse.ArgumentParser(description="Pointer angle demo (traditional CV).")
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--image-dir", default="fig")
    parser.add_argument("--label-dir", default="label")
    parser.add_argument("--stem", default=None, help="Sample stem, e.g. 00201")
    parser.add_argument("--image-path", type=Path, default=None)
    parser.add_argument("--label-path", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output preview path. Default: <dataset_root>/demo_angle/<stem>_angle_demo.jpg",
    )
    return parser.parse_args()


def try_load_json(path: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Cannot decode json: {path}")


def normalize_angle_deg(deg: float) -> float:
    out = deg % 360.0
    return out if out >= 0 else out + 360.0


def segment_center_distance(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    px, py = x2 - x1, y2 - y1
    norm2 = px * px + py * py
    if norm2 < 1e-6:
        return math.hypot(cx - x1, cy - y1)
    u = ((cx - x1) * px + (cy - y1) * py) / norm2
    u = max(0.0, min(1.0, u))
    qx, qy = x1 + u * px, y1 + u * py
    return math.hypot(cx - qx, cy - qy)


def detect_pointer_line(roi_bgr: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    h, w = roi_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(10, int(min(w, h) * 0.22)),
        maxLineGap=8,
    )
    if lines is None:
        return None

    best = None
    best_score = -1e18
    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, item.tolist())
        length = math.hypot(x2 - x1, y2 - y1)
        center_dist = segment_center_distance(cx, cy, x1, y1, x2, y2)
        # Prefer long lines passing near ROI center.
        score = length - 2.0 * center_dist
        if score > best_score:
            best_score = score
            best = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
    return best


def line_to_angle_deg(
    p1: Tuple[int, int], p2: Tuple[int, int], roi_shape: Tuple[int, int, int]
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    h, w = roi_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    d1 = math.hypot(p1[0] - cx, p1[1] - cy)
    d2 = math.hypot(p2[0] - cx, p2[1] - cy)
    tip = p1 if d1 > d2 else p2
    center = (int(round(cx)), int(round(cy)))

    dx = tip[0] - cx
    dy = cy - tip[1]  # flip y to math coordinate
    deg = normalize_angle_deg(math.degrees(math.atan2(dy, dx)))
    return deg, center, tip


def find_pointer_boxes(label_json: dict) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    out: List[Tuple[str, Tuple[int, int, int, int]]] = []
    for shp in label_json.get("shapes", []):
        label = str(shp.get("label", "")).strip()
        if not POINTER_RE.fullmatch(label):
            continue
        if str(shp.get("shape_type", "")).lower() != "rectangle":
            continue
        pts = shp.get("points", [])
        if not isinstance(pts, list) or len(pts) != 2:
            continue
        try:
            (x1, y1), (x2, y2) = pts
            x0, y0 = int(min(x1, x2)), int(min(y1, y2))
            x1i, y1i = int(max(x1, x2)), int(max(y1, y2))
        except Exception:
            continue
        out.append((label, (x0, y0, x1i, y1i)))
    return out


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, str]:
    if args.image_path and args.label_path:
        return args.image_path.resolve(), args.label_path.resolve(), args.image_path.stem

    dataset_root = args.dataset_root.resolve()
    img_dir = (dataset_root / args.image_dir).resolve()
    lbl_dir = (dataset_root / args.label_dir).resolve()

    if args.stem:
        stem = Path(args.stem).stem
        return img_dir / f"{stem}.jpg", lbl_dir / f"{stem}.json", stem

    stems = sorted(set(p.stem for p in img_dir.iterdir()) & set(p.stem for p in lbl_dir.glob("*.json")))
    if not stems:
        raise RuntimeError("No paired image+label found.")
    stem = stems[0]
    return img_dir / f"{stem}.jpg", lbl_dir / f"{stem}.json", stem


def main() -> None:
    args = parse_args()
    image_path, label_path, stem = resolve_paths(args)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    label_json = try_load_json(label_path)
    boxes = find_pointer_boxes(label_json)
    if not boxes:
        raise RuntimeError("No pointer rectangles (10^-n) found in label.")

    vis = img.copy()
    results: Dict[str, float] = {}
    h_img, w_img = img.shape[:2]

    for label, (x0, y0, x1, y1) in boxes:
        x0 = max(0, min(w_img - 1, x0))
        y0 = max(0, min(h_img - 1, y0))
        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        if x1 <= x0 or y1 <= y0:
            continue

        roi = img[y0:y1, x0:x1]
        line = detect_pointer_line(roi)

        cv2.rectangle(vis, (x0, y0), (x1, y1), (60, 220, 80), 2)
        if line is None:
            cv2.putText(vis, f"{label}: no-line", (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue

        p1, p2 = line
        angle_deg, center, tip = line_to_angle_deg(p1, p2, roi.shape)
        results[label] = angle_deg

        cxg, cyg = x0 + center[0], y0 + center[1]
        txg, tyg = x0 + tip[0], y0 + tip[1]
        cv2.line(vis, (cxg, cyg), (txg, tyg), (255, 60, 60), 3)
        cv2.circle(vis, (cxg, cyg), 4, (0, 255, 255), -1)
        cv2.circle(vis, (txg, tyg), 4, (255, 60, 60), -1)
        cv2.putText(
            vis,
            f"{label}: {angle_deg:.1f} deg",
            (x0, max(24, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 60, 60),
            2,
        )

    out_path = args.out.resolve() if args.out else (args.dataset_root.resolve() / "demo_angle" / f"{stem}_angle_demo.jpg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

    print("=== Pointer Angle Demo ===")
    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print("Angles (deg, 0=right, 90=up):")
    for k in sorted(results.keys()):
        print(f"  {k}: {results[k]:.2f}")
    print(f"Preview: {out_path}")


if __name__ == "__main__":
    main()
