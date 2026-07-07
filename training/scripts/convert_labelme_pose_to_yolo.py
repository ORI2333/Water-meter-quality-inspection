#!/usr/bin/env python3
"""
Convert merged Labelme pointer annotations to Ultralytics YOLO pose format.

Each pointer dial is exported as:
  cls cx cy w h center_x center_y 2 tip_x tip_y 2

All coordinates are normalized to image width/height.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


POINTER_LABELS = ["10^-1", "10^-2", "10^-3", "10^-4"]
CLASS_MAP = {name: idx for idx, name in enumerate(POINTER_LABELS)}
IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Convert Labelme center/tip pointer labels to YOLO11-pose dataset."
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=root / "data" / "original_dataset" / "labelme_pose_work_copy",
        help="Merged Labelme directory containing images and JSON files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "yolo11_pointer_pose",
        help="Output YOLO pose dataset directory.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing output directory before conversion.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_image(src_dir: Path, data: dict, stem: str) -> Path:
    image_path = data.get("imagePath")
    candidates: List[Path] = []
    if image_path:
        candidates.append(src_dir / image_path)
        candidates.append(src_dir / Path(image_path).name)
    for ext in IMAGE_EXTS:
        candidates.append(src_dir / f"{stem}{ext}")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for {stem}")


def xyxy_from_rectangle(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def point_from_shape(shape: dict) -> Tuple[float, float]:
    pts = shape.get("points") or []
    if len(pts) != 1 or len(pts[0]) != 2:
        raise ValueError("point shape must contain exactly one [x, y] point")
    return float(pts[0][0]), float(pts[0][1])


def clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def yolo_line(
    cls_id: int,
    box: Tuple[float, float, float, float],
    center: Tuple[float, float],
    tip: Tuple[float, float],
    width: int,
    height: int,
) -> str:
    x0, y0, x1, y1 = box
    cx = ((x0 + x1) / 2.0) / width
    cy = ((y0 + y1) / 2.0) / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    values = [
        cls_id,
        clamp01(cx),
        clamp01(cy),
        clamp01(bw),
        clamp01(bh),
        clamp01(center[0] / width),
        clamp01(center[1] / height),
        2,
        clamp01(tip[0] / width),
        clamp01(tip[1] / height),
        2,
    ]
    return " ".join(str(v) if isinstance(v, int) else f"{v:.8f}" for v in values)


def iter_objects(data: dict) -> Iterable[Tuple[int, str, dict, dict, dict]]:
    by_group: Dict[int, dict] = {}
    for shape in data.get("shapes", []):
        gid = shape.get("group_id")
        if not isinstance(gid, int):
            continue
        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "")).strip()
        group = by_group.setdefault(gid, {})
        if label in CLASS_MAP and shape_type == "rectangle":
            group["rect"] = shape
            group["class_label"] = label
        elif label == "center" and shape_type == "point":
            group["center"] = shape
        elif label == "tip" and shape_type == "point":
            group["tip"] = shape

    for gid, group in sorted(by_group.items()):
        if {"rect", "center", "tip", "class_label"} - set(group):
            raise ValueError(f"incomplete group_id {gid}: {sorted(group)}")
        yield gid, group["class_label"], group["rect"], group["center"], group["tip"]


def write_yaml(out_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(POINTER_LABELS))
    content = f"""# Water meter pointer pose dataset
path: .
train: images/train
val: images/val

kpt_shape: [2, 3]
flip_idx: [0, 1]

names:
{names}
"""
    (out_dir / "water_meter_pose.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    src_dir = args.src_dir.resolve()
    out_dir = args.out_dir.resolve()
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1")
    if out_dir.exists() and args.clean:
        shutil.rmtree(out_dir)

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files in {src_dir}")

    rng = random.Random(args.seed)
    shuffled = json_files[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * args.val_ratio)))
    val_set = {p.stem for p in shuffled[:val_count]}

    stats = {"train": 0, "val": 0, "objects": 0}
    for json_path in json_files:
        data = load_json(json_path)
        width = int(data.get("imageWidth") or 0)
        height = int(data.get("imageHeight") or 0)
        if width <= 0 or height <= 0:
            raise ValueError(f"{json_path.name}: invalid imageWidth/imageHeight")

        image_path = find_image(src_dir, data, json_path.stem)
        split = "val" if json_path.stem in val_set else "train"
        dst_image = out_dir / "images" / split / image_path.name
        dst_label = out_dir / "labels" / split / f"{json_path.stem}.txt"

        lines = []
        for _gid, class_label, rect, center_shape, tip_shape in iter_objects(data):
            box = xyxy_from_rectangle(rect.get("points") or [])
            center = point_from_shape(center_shape)
            tip = point_from_shape(tip_shape)
            lines.append(yolo_line(CLASS_MAP[class_label], box, center, tip, width, height))

        if len(lines) != 4:
            raise ValueError(f"{json_path.name}: expected 4 pointer objects, got {len(lines)}")

        shutil.copy2(image_path, dst_image)
        dst_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        stats[split] += 1
        stats["objects"] += len(lines)

    write_yaml(out_dir)
    print("=== Labelme to YOLO pose done ===")
    print(f"Source: {src_dir}")
    print(f"Output: {out_dir}")
    print(f"Train images: {stats['train']}")
    print(f"Val images: {stats['val']}")
    print(f"Objects: {stats['objects']}")
    print(f"YAML: {out_dir / 'water_meter_pose.yaml'}")


if __name__ == "__main__":
    main()
