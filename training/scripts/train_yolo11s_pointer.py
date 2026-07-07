#!/usr/bin/env python3
"""
Prepare and train YOLO11s for water-meter pointer detection (4 classes).

Input dataset (Labelme):
  data/original_dataset/
    |- fig/*.jpg
    |- label/*.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


CLASS_NAMES = ["10^-1", "10^-2", "10^-3", "10^-4"]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_dataset_root = (repo_root / "data" / "original_dataset").resolve()
    default_output_root = (repo_root / "data" / "yolo11s_pointer_detect").resolve()

    parser = argparse.ArgumentParser(
        description="Prepare Labelme annotations and train YOLO11s pointer detector."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root)
    parser.add_argument("--image-dir", type=str, default="fig")
    parser.add_argument("--label-dir", type=str, default="label")
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--model", type=str, default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="cpu: force CPU; cuda: force GPU(0); auto: pick CUDA if available.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare YOLO dataset; do not start training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root before preparing.",
    )
    return parser.parse_args()


def try_load_json(path: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode json: {path}")


def collect_image_size(image_path: Path) -> Tuple[int, int]:
    from PIL import Image  # type: ignore

    with Image.open(image_path) as img:
        w, h = img.size
    return w, h


def rect_points_to_xyxy(points: List[List[float]]) -> Tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = points
    x_min = min(float(x1), float(x2))
    y_min = min(float(y1), float(y2))
    x_max = max(float(x1), float(x2))
    y_max = max(float(y1), float(y2))
    return x_min, y_min, x_max, y_max


def to_yolo_xywh(
    x_min: float, y_min: float, x_max: float, y_max: float, width: int, height: int
) -> Tuple[float, float, float, float]:
    box_w = max(0.0, x_max - x_min)
    box_h = max(0.0, y_max - y_min)
    cx = x_min + box_w / 2.0
    cy = y_min + box_h / 2.0
    return cx / width, cy / height, box_w / width, box_h / height


def make_dirs(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)


def prepare_dataset(args: argparse.Namespace) -> Tuple[Path, Dict[str, int]]:
    dataset_root = args.dataset_root.resolve()
    image_dir = (dataset_root / args.image_dir).resolve()
    label_dir = (dataset_root / args.label_dir).resolve()
    output_root = args.output_root.resolve()

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing dirs: image_dir={image_dir}, label_dir={label_dir}")

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    make_dirs(output_root)

    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    image_paths = {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}}
    label_paths = {p.stem: p for p in label_dir.glob("*.json")}
    stems = sorted(set(image_paths.keys()) & set(label_paths.keys()))

    valid_items: List[Tuple[str, List[str]]] = []
    skipped_no_pointer = 0
    skipped_bad_label = 0

    for stem in stems:
        image_path = image_paths[stem]
        label_path = label_paths[stem]

        try:
            data = try_load_json(label_path)
            width, height = collect_image_size(image_path)
        except Exception:
            skipped_bad_label += 1
            continue

        lines: List[str] = []
        for shp in data.get("shapes", []):
            if not isinstance(shp, dict):
                continue
            label = str(shp.get("label", "")).strip()
            if label not in class_to_id:
                continue
            if str(shp.get("shape_type", "")).lower() != "rectangle":
                continue
            points = shp.get("points", [])
            if not (isinstance(points, list) and len(points) == 2):
                continue
            try:
                x_min, y_min, x_max, y_max = rect_points_to_xyxy(points)  # type: ignore[arg-type]
                x, y, w, h = to_yolo_xywh(x_min, y_min, x_max, y_max, width, height)
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            lines.append(f"{class_to_id[label]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if not lines:
            skipped_no_pointer += 1
            continue
        valid_items.append((stem, lines))

    random.Random(args.seed).shuffle(valid_items)
    val_count = int(len(valid_items) * args.val_ratio)
    val_set = set(stem for stem, _ in valid_items[:val_count])

    train_count = 0
    out_class_counter = {k: 0 for k in CLASS_NAMES}
    for stem, lines in valid_items:
        split = "val" if stem in val_set else "train"
        image_src = image_paths[stem]
        image_dst = output_root / "images" / split / image_src.name
        label_dst = output_root / "labels" / split / f"{stem}.txt"

        shutil.copy2(image_src, image_dst)
        label_dst.write_text("\n".join(lines) + "\n", encoding="utf-8")

        for line in lines:
            class_id = int(line.split()[0])
            out_class_counter[CLASS_NAMES[class_id]] += 1
        if split == "train":
            train_count += 1

    yaml_path = output_root / "dataset.yaml"
    names_dict = {i: name for i, name in enumerate(CLASS_NAMES)}
    yaml_text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {names_dict}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    report = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "total_paired": len(stems),
        "total_used": len(valid_items),
        "train_count": train_count,
        "val_count": len(valid_items) - train_count,
        "skipped_no_pointer": skipped_no_pointer,
        "skipped_bad_label": skipped_bad_label,
        "class_box_count": out_class_counter,
        "classes": CLASS_NAMES,
        "dataset_yaml": str(yaml_path),
    }
    (output_root / "prepare_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return yaml_path, report


def resolve_device(device_arg: str):
    if device_arg == "cpu":
        return "cpu", "cpu"

    if device_arg == "cuda":
        import torch  # type: ignore

        if not torch.cuda.is_available():
            print("[WARN] --device cuda set, but CUDA unavailable. Fallback to CPU.")
            return "cpu", "cpu"
        return 0, "cuda:0"

    # auto
    import torch  # type: ignore

    if torch.cuda.is_available():
        return 0, "cuda:0"
    return "cpu", "cpu"


def train(args: argparse.Namespace, yaml_path: Path) -> None:
    from ultralytics import YOLO  # type: ignore

    device_for_ultralytics, device_show = resolve_device(args.device)
    print(f"Training device: {device_show}")

    model = YOLO(args.model)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device_for_ultralytics,
        project=str((args.output_root / "runs").resolve()),
        name="yolo11s_pointer_detect",
    )


def main() -> None:
    args = parse_args()
    yaml_path, report = prepare_dataset(args)

    print("=== Dataset prepared ===")
    print(f"Used samples: {report['total_used']} (train={report['train_count']}, val={report['val_count']})")
    print(f"Skipped no-pointer: {report['skipped_no_pointer']}, bad-label: {report['skipped_bad_label']}")
    print(f"Class box count: {report['class_box_count']}")
    print(f"YAML: {yaml_path}")

    if args.prepare_only:
        print("Prepare only mode, skip training.")
        return

    train(args, yaml_path)


if __name__ == "__main__":
    main()
