#!/usr/bin/env python3
"""
Bootstrap Labelme pose points for water-meter pointer dials.

The script keeps existing rectangle labels and appends two point shapes for each
pointer ROI:
  - center: dial rotation center, initialized from the rectangle center
  - tip: pointer tip, initialized by a rough Hough/dark-pixel heuristic

By default it writes to a new label folder so the original labels stay intact.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


POINTER_LABELS = ["10^-1", "10^-2", "10^-3", "10^-4"]
AUTO_POINT_LABELS = {"center", "tip"}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    default_root = (repo_root / "data" / "original_dataset").resolve()

    parser = argparse.ArgumentParser(description="Bootstrap center/tip points for Labelme pointer labels.")
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--image-dir", default="fig")
    parser.add_argument("--label-dir", default="label")
    parser.add_argument("--output-label-dir", default="label_pose_bootstrap")
    parser.add_argument("--preview-dir", default="pose_bootstrap_preview")
    parser.add_argument("--pointer-labels", nargs="*", default=POINTER_LABELS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output label/preview folders.")
    parser.add_argument("--in-place", action="store_true", help="Write back into --label-dir. Use with care.")
    parser.add_argument("--no-vis", action="store_true", help="Do not export preview images.")
    parser.add_argument("--samples", type=int, default=40, help="Number of preview images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--padding-ratio", type=float, default=0.04, help="Temporary ROI padding used for tip estimation.")
    return parser.parse_args()


def try_load_json(path: Path) -> dict:
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode json: {path}")


def imread_unicode(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def image_map(image_dir: Path) -> Dict[str, Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return {p.stem: p for p in image_dir.iterdir() if p.suffix.lower() in exts}


def rect_xyxy(points: List[List[float]]) -> Tuple[float, float, float, float]:
    (x1, y1), (x2, y2) = points
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clamp_point(x: float, y: float, w: int, h: int) -> Tuple[float, float]:
    return max(0.0, min(float(w - 1), x)), max(0.0, min(float(h - 1), y))


def segment_center_distance(cx: float, cy: float, x1: float, y1: float, x2: float, y2: float) -> float:
    px, py = x2 - x1, y2 - y1
    norm2 = px * px + py * py
    if norm2 < 1e-6:
        return math.hypot(cx - x1, cy - y1)
    u = ((cx - x1) * px + (cy - y1) * py) / norm2
    u = max(0.0, min(1.0, u))
    qx, qy = x1 + u * px, y1 + u * py
    return math.hypot(cx - qx, cy - qy)


def estimate_tip_by_hough(roi: np.ndarray, cx: float, cy: float) -> Optional[Tuple[float, float, float]]:
    h, w = roi.shape[:2]
    if min(w, h) < 24:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(12, int(min(w, h) * 0.08)),
        minLineLength=max(12, int(min(w, h) * 0.18)),
        maxLineGap=max(5, int(min(w, h) * 0.04)),
    )
    if lines is None:
        return None

    best_tip: Optional[Tuple[float, float]] = None
    best_score = -1e18
    best_conf = 0.0

    for item in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, item.tolist())
        length = math.hypot(x2 - x1, y2 - y1)
        center_dist = segment_center_distance(cx, cy, x1, y1, x2, y2)
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        tip = (x1, y1) if d1 >= d2 else (x2, y2)
        far = max(d1, d2)
        # Prefer long lines that pass near the dial center and extend outward.
        score = length + 0.55 * far - 2.4 * center_dist
        if score > best_score:
            best_score = score
            best_tip = tip
            best_conf = max(0.0, min(1.0, (length / max(min(w, h), 1)) * (1.0 - center_dist / max(min(w, h), 1))))

    if best_tip is None:
        return None
    return best_tip[0], best_tip[1], best_conf


def estimate_tip_by_dark_pixels(roi: np.ndarray, cx: float, cy: float) -> Optional[Tuple[float, float, float]]:
    h, w = roi.shape[:2]
    if min(w, h) < 24:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    threshold = np.percentile(blur, 28)
    mask = blur <= threshold

    yy, xx = np.nonzero(mask)
    if len(xx) < 20:
        return None

    dx = xx.astype(np.float32) - float(cx)
    dy = yy.astype(np.float32) - float(cy)
    dist = np.sqrt(dx * dx + dy * dy)
    keep = (dist > min(w, h) * 0.12) & (dist < min(w, h) * 0.58)
    if int(keep.sum()) < 20:
        return None

    xx, yy, dist = xx[keep], yy[keep], dist[keep]
    # Pick a high-distance dark point cluster as a rough needle tip.
    cutoff = np.percentile(dist, 92)
    far = dist >= cutoff
    if int(far.sum()) == 0:
        return None
    tx = float(np.mean(xx[far]))
    ty = float(np.mean(yy[far]))
    conf = max(0.0, min(0.55, float(cutoff / max(min(w, h), 1))))
    return tx, ty, conf


def estimate_tip(image: np.ndarray, box: Tuple[float, float, float, float], padding_ratio: float) -> Tuple[float, float, str, float]:
    h_img, w_img = image.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    pad = max(2, int(min(bw, bh) * padding_ratio))

    rx0 = max(0, int(math.floor(x0)) - pad)
    ry0 = max(0, int(math.floor(y0)) - pad)
    rx1 = min(w_img, int(math.ceil(x1)) + pad)
    ry1 = min(h_img, int(math.ceil(y1)) + pad)
    roi = image[ry0:ry1, rx0:rx1]

    cx = (x0 + x1) / 2.0 - rx0
    cy = (y0 + y1) / 2.0 - ry0

    for name, fn in (("hough", estimate_tip_by_hough), ("dark", estimate_tip_by_dark_pixels)):
        out = fn(roi, cx, cy)
        if out is None:
            continue
        tx, ty, conf = out
        gx, gy = clamp_point(tx + rx0, ty + ry0, w_img, h_img)
        return gx, gy, name, conf

    # Fallback: put the tip to the right of the center. It is intentionally
    # visible and easy to drag to the true needle tip in Labelme.
    gx, gy = clamp_point((x0 + x1) / 2.0 + bw * 0.32, (y0 + y1) / 2.0, w_img, h_img)
    return gx, gy, "fallback", 0.0


def next_group_id(used: set) -> int:
    gid = 1
    while gid in used:
        gid += 1
    used.add(gid)
    return gid


def make_point_shape(label: str, point: Tuple[float, float], group_id: int, pointer_label: str, method: str, conf: float) -> dict:
    return {
        "label": label,
        "points": [[float(point[0]), float(point[1])]],
        "group_id": group_id,
        "description": f"auto_bootstrap:{pointer_label}:{method}:conf={conf:.3f}",
        "shape_type": "point",
        "flags": {"auto_bootstrap": True},
        "mask": None,
    }


def remove_old_auto_points(shapes: Iterable[dict]) -> List[dict]:
    out: List[dict] = []
    for shp in shapes:
        label = str(shp.get("label", "")).strip()
        shape_type = str(shp.get("shape_type", "")).lower()
        desc = str(shp.get("description", "") or "")
        flags = shp.get("flags", {}) if isinstance(shp.get("flags", {}), dict) else {}
        is_auto = bool(flags.get("auto_bootstrap")) or desc.startswith("auto_bootstrap:")
        if shape_type == "point" and label in AUTO_POINT_LABELS and is_auto:
            continue
        out.append(shp)
    return out


def bootstrap_one(
    label_path: Path,
    image_path: Path,
    out_path: Path,
    pointer_labels: set,
    padding_ratio: float,
) -> dict:
    data = try_load_json(label_path)
    image = imread_unicode(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    shapes = remove_old_auto_points(data.get("shapes", []))
    used_group_ids = {int(s["group_id"]) for s in shapes if isinstance(s, dict) and isinstance(s.get("group_id"), int)}

    added = 0
    methods: Dict[str, int] = {}
    for shp in shapes:
        if not isinstance(shp, dict):
            continue
        label = str(shp.get("label", "")).strip()
        if label not in pointer_labels:
            continue
        if str(shp.get("shape_type", "")).lower() != "rectangle":
            continue
        pts = shp.get("points", [])
        if not (isinstance(pts, list) and len(pts) == 2):
            continue

        try:
            x0, y0, x1, y1 = rect_xyxy(pts)  # type: ignore[arg-type]
        except Exception:
            continue
        if x1 <= x0 or y1 <= y0:
            continue

        gid = shp.get("group_id")
        if not isinstance(gid, int):
            gid = next_group_id(used_group_ids)
            shp["group_id"] = gid

        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        tip_x, tip_y, method, conf = estimate_tip(image, (x0, y0, x1, y1), padding_ratio)
        shapes.append(make_point_shape("center", center, gid, label, "box_center", 1.0))
        shapes.append(make_point_shape("tip", (tip_x, tip_y), gid, label, method, conf))
        added += 2
        methods[method] = methods.get(method, 0) + 1

    data["shapes"] = shapes
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"stem": label_path.stem, "added_points": added, "methods": methods}


def draw_previews(
    records: List[dict],
    image_paths: Dict[str, Path],
    out_label_dir: Path,
    preview_dir: Path,
    samples: int,
    seed: int,
) -> None:
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    pool = [r["stem"] for r in records if r.get("added_points", 0) > 0]
    random.Random(seed).shuffle(pool)
    for stem in pool[: max(0, samples)]:
        image = imread_unicode(image_paths[stem])
        if image is None:
            continue
        data = try_load_json(out_label_dir / f"{stem}.json")
        vis = image.copy()

        group_centers: Dict[int, Tuple[int, int]] = {}
        group_tips: Dict[int, Tuple[int, int]] = {}
        for shp in data.get("shapes", []):
            if not isinstance(shp, dict):
                continue
            label = str(shp.get("label", "")).strip()
            gid = shp.get("group_id")
            pts = shp.get("points", [])
            shape_type = str(shp.get("shape_type", "")).lower()

            if shape_type == "rectangle" and label in POINTER_LABELS and isinstance(pts, list) and len(pts) == 2:
                x0, y0, x1, y1 = rect_xyxy(pts)  # type: ignore[arg-type]
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (60, 220, 80), 2)
                cv2.putText(vis, label, (int(x0), max(24, int(y0) - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 220, 80), 2)
            elif shape_type == "point" and label in AUTO_POINT_LABELS and isinstance(gid, int) and isinstance(pts, list) and pts:
                x, y = int(round(pts[0][0])), int(round(pts[0][1]))
                if label == "center":
                    group_centers[gid] = (x, y)
                    cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)
                elif label == "tip":
                    group_tips[gid] = (x, y)
                    cv2.circle(vis, (x, y), 6, (255, 60, 60), -1)

        for gid, center in group_centers.items():
            tip = group_tips.get(gid)
            if tip:
                cv2.line(vis, center, tip, (255, 60, 60), 2)

        imwrite_unicode(preview_dir / f"{stem}_pose_bootstrap.jpg", vis)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    image_dir = (dataset_root / args.image_dir).resolve()
    label_dir = (dataset_root / args.label_dir).resolve()
    out_label_dir = label_dir if args.in_place else (dataset_root / args.output_label_dir).resolve()
    preview_dir = (dataset_root / args.preview_dir).resolve()

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing dataset dirs: image_dir={image_dir}, label_dir={label_dir}")
    if out_label_dir.exists() and not args.overwrite and not args.in_place:
        raise FileExistsError(f"Output label dir exists, use --overwrite: {out_label_dir}")
    if out_label_dir.exists() and args.overwrite and not args.in_place:
        shutil.rmtree(out_label_dir)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = image_map(image_dir)
    pointer_labels = set(args.pointer_labels)
    records: List[dict] = []
    missing_images: List[str] = []

    for label_path in sorted(label_dir.glob("*.json")):
        image_path = images.get(label_path.stem)
        if image_path is None:
            missing_images.append(label_path.stem)
            continue
        out_path = out_label_dir / label_path.name
        rec = bootstrap_one(
            label_path=label_path,
            image_path=image_path,
            out_path=out_path,
            pointer_labels=pointer_labels,
            padding_ratio=float(args.padding_ratio),
        )
        records.append(rec)

    report = {
        "dataset_root": str(dataset_root),
        "label_dir": str(label_dir),
        "output_label_dir": str(out_label_dir),
        "total_labels": len(records),
        "missing_images": missing_images[:200],
        "added_points": sum(r.get("added_points", 0) for r in records),
        "method_counts": {},
        "records": records,
    }
    method_counts: Dict[str, int] = {}
    for rec in records:
        for method, count in rec.get("methods", {}).items():
            method_counts[method] = method_counts.get(method, 0) + int(count)
    report["method_counts"] = method_counts

    report_path = dataset_root / "pose_bootstrap_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.no_vis:
        draw_previews(records, images, out_label_dir, preview_dir, args.samples, args.seed)

    print("=== Pose bootstrap done ===")
    print(f"Dataset root: {dataset_root}")
    print(f"Input labels: {label_dir}")
    print(f"Output labels: {out_label_dir}")
    print(f"Label files processed: {len(records)}")
    print(f"Added points: {report['added_points']}")
    print(f"Tip methods: {method_counts}")
    print(f"Missing images: {len(missing_images)}")
    print(f"Report: {report_path}")
    if not args.no_vis:
        print(f"Preview: {preview_dir}")


if __name__ == "__main__":
    main()
