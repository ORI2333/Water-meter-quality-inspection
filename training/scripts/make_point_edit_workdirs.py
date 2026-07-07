#!/usr/bin/env python3
"""
Create point-only Labelme workdirs for safer manual editing.

Each generated workdir keeps only one editable point label in JSON, while the
image has reference boxes and the paired point burned in. After editing, use
merge_point_edit_workdir.py to copy edited point coordinates back to the full
Labelme JSON files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


POINTER_LABELS = {"10^-1", "10^-2", "10^-3", "10^-4"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_full = root / "data" / "original_dataset" / "labelme_pose_work_copy"
    parser = argparse.ArgumentParser(description="Create center/tip point-only Labelme workdirs.")
    parser.add_argument("--full-work-dir", type=Path, default=default_full)
    parser.add_argument("--output-root", type=Path, default=default_full.parent)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def imread(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def imwrite(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Cannot encode image: {path}")
    buf.tofile(str(path))


def rect_xyxy(points: List[List[float]]) -> Tuple[int, int, int, int]:
    (x1, y1), (x2, y2) = points
    return int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))


def collect_points(shapes: List[dict]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out = {}
    for s in shapes:
        if s.get("shape_type") != "point":
            continue
        label = str(s.get("label", ""))
        gid = s.get("group_id")
        pts = s.get("points") or []
        if label in {"center", "tip"} and isinstance(gid, int) and pts:
            out[(label, gid)] = (float(pts[0][0]), float(pts[0][1]))
    return out


def draw_reference(image: np.ndarray, shapes: List[dict], target_label: str) -> np.ndarray:
    vis = image.copy()
    points = collect_points(shapes)

    for s in shapes:
        if s.get("shape_type") != "rectangle":
            continue
        label = str(s.get("label", ""))
        if label not in POINTER_LABELS:
            continue
        x0, y0, x1, y1 = rect_xyxy(s.get("points", []))
        cv2.rectangle(vis, (x0, y0), (x1, y1), (70, 220, 70), 2)
        cv2.putText(vis, label, (x0, max(24, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 220, 70), 2)

    for (label, gid), (x, y) in points.items():
        if label == target_label:
            continue
        color = (0, 230, 255) if label == "center" else (255, 80, 80)
        cv2.circle(vis, (int(round(x)), int(round(y))), 5, color, -1)

    return vis


def filtered_shapes(shapes: List[dict], target_label: str) -> List[dict]:
    out = []
    for s in shapes:
        if s.get("shape_type") == "point" and s.get("label") == target_label:
            ns = dict(s)
            ns["description"] = f"point_only_edit:{target_label}"
            ns["flags"] = dict(ns.get("flags") or {})
            ns["flags"]["point_only_edit"] = True
            out.append(ns)
    return out


def build_workdir(full_dir: Path, out_dir: Path, target_label: str, overwrite: bool, jpeg_quality: int) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for json_path in sorted(full_dir.glob("*.json")):
        data = read_json(json_path)
        image_path = full_dir / f"{json_path.stem}.jpg"
        if not image_path.exists():
            continue
        image = imread(image_path)
        vis = draw_reference(image, data.get("shapes", []), target_label=target_label)

        out_img = out_dir / image_path.name
        out_json = out_dir / json_path.name
        imwrite(out_img, vis, jpeg_quality=jpeg_quality)

        out_data = dict(data)
        out_data["imagePath"] = out_img.name
        out_data["imageData"] = None
        out_data["shapes"] = filtered_shapes(data.get("shapes", []), target_label=target_label)
        out_json.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    print(f"{target_label}: wrote {count} samples -> {out_dir}")


def main() -> None:
    args = parse_args()
    full_dir = args.full_work_dir.resolve()
    output_root = args.output_root.resolve()
    if not full_dir.exists():
        raise FileNotFoundError(full_dir)

    build_workdir(full_dir, output_root / "labelme_tip_edit", "tip", args.overwrite, args.jpeg_quality)
    build_workdir(full_dir, output_root / "labelme_center_edit", "center", args.overwrite, args.jpeg_quality)


if __name__ == "__main__":
    main()
