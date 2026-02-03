#!/usr/bin/env python3
"""
Validate partially labeled Labelme dataset and optionally export visual previews.

Default dataset layout:
  data/original_dataset/
    |- fig/
    |- label/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = (script_dir.parent / "data" / "original_dataset").resolve()

    parser = argparse.ArgumentParser(
        description="Validate Labelme annotations and check pointer labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help="Dataset root folder (contains fig/ and label/).",
    )
    parser.add_argument("--image-dir", default="fig", help="Image folder name.")
    parser.add_argument("--label-dir", default="label", help="Label folder name.")
    parser.add_argument(
        "--pointer-labels",
        nargs="*",
        default=[],
        help="Exact pointer label names. If empty, infer by keywords/pattern.",
    )
    parser.add_argument(
        "--pointer-keywords",
        nargs="*",
        default=["pointer", "needle"],
        help="Keywords for auto-detecting pointer labels.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of preview images to save.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for preview sampling."
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Output report JSON path. Default: <dataset-root>/validation_report.json",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=None,
        help="Visualization output folder. Default: <dataset-root>/validation_preview",
    )
    parser.add_argument(
        "--no-vis",
        action="store_true",
        help="Disable visualization export.",
    )
    parser.add_argument(
        "--vis-problems-only",
        action="store_true",
        help="Only export preview images for samples with issues.",
    )
    parser.add_argument(
        "--check-stems",
        nargs="*",
        default=[],
        help="Only validate these image ids/stems, e.g. 00001 00201.",
    )
    parser.add_argument(
        "--check-stems-file",
        type=Path,
        default=None,
        help="Text file of stems to validate (one stem or filename per line).",
    )
    parser.add_argument(
        "--label-font-size",
        type=int,
        default=34,
        help="Preview label font size.",
    )
    return parser.parse_args()


def try_load_json(path: Path) -> Tuple[Optional[dict], Optional[str]]:
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb18030")
    for enc in encodings:
        try:
            return json.loads(path.read_text(encoding=enc)), None
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as e:
            return None, f"JSON decode error: {e}"
        except Exception as e:  # pragma: no cover
            return None, f"Read error: {e}"
    return None, "Unable to decode file with utf-8/gbk encodings"


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def shape_bounds(points: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def contains_keyword(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


def looks_like_pointer_label(label: str) -> bool:
    # Common water-meter pointer labels often look like 10^-1/10^-2/...
    return bool(re.fullmatch(r"\s*10\^-?\d+\s*", label))


def is_pointer_label(label: str, pointer_labels_exact: set, pointer_keywords: List[str]) -> bool:
    if pointer_labels_exact:
        return label in pointer_labels_exact
    return contains_keyword(label, pointer_keywords) or looks_like_pointer_label(label)


def normalize_to_stem(text: str) -> str:
    s = text.strip().strip('"').strip("'")
    if not s:
        return ""
    return Path(s).stem


def load_requested_stems(cli_stems: List[str], stems_file: Optional[Path]) -> set:
    requested = set()
    for s in cli_stems:
        stem = normalize_to_stem(s)
        if stem:
            requested.add(stem)

    if stems_file is not None:
        file_path = stems_file.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"--check-stems-file not found: {file_path}")

        for line in file_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            stem = normalize_to_stem(raw)
            if stem:
                requested.add(stem)

    return requested


def validate_shapes(
    shapes: list,
    image_size: Optional[Tuple[int, int]],
    pointer_labels_exact: set,
    pointer_keywords: List[str],
) -> Tuple[dict, List[dict], Counter, int]:
    stats = {
        "shape_count": 0,
        "invalid_shape_count": 0,
        "out_of_bounds_points": 0,
        "zero_area_shapes": 0,
    }
    issues: List[dict] = []
    labels_counter: Counter = Counter()
    pointer_shape_count = 0

    width, height = (image_size or (None, None))

    for idx, shape in enumerate(shapes):
        stats["shape_count"] += 1

        if not isinstance(shape, dict):
            stats["invalid_shape_count"] += 1
            issues.append({"shape_index": idx, "type": "shape_not_dict"})
            continue

        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "")).strip().lower()
        points = shape.get("points", [])
        labels_counter[label] += 1

        if is_pointer_label(label, pointer_labels_exact, pointer_keywords):
            pointer_shape_count += 1

        if not label:
            stats["invalid_shape_count"] += 1
            issues.append({"shape_index": idx, "type": "empty_label"})

        if not isinstance(points, list) or not points:
            stats["invalid_shape_count"] += 1
            issues.append({"shape_index": idx, "type": "invalid_points"})
            continue

        valid_points: List[List[float]] = []
        for p in points:
            if (
                isinstance(p, list)
                and len(p) == 2
                and is_number(p[0])
                and is_number(p[1])
            ):
                valid_points.append([float(p[0]), float(p[1])])
            else:
                stats["invalid_shape_count"] += 1
                issues.append({"shape_index": idx, "type": "bad_point_format"})
                valid_points = []
                break

        if not valid_points:
            continue

        if shape_type == "rectangle" and len(valid_points) != 2:
            stats["invalid_shape_count"] += 1
            issues.append({"shape_index": idx, "type": "rectangle_points_not_2"})

        x0, y0, x1, y1 = shape_bounds(valid_points)
        if x0 == x1 or y0 == y1:
            stats["zero_area_shapes"] += 1
            issues.append({"shape_index": idx, "type": "zero_area"})

        if width is not None and height is not None:
            for px, py in valid_points:
                if px < 0 or py < 0 or px > width or py > height:
                    stats["out_of_bounds_points"] += 1
                    issues.append({"shape_index": idx, "type": "point_out_of_bounds"})

    return stats, issues, labels_counter, pointer_shape_count


def collect_image_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def draw_previews(
    records: List[dict],
    image_map: Dict[str, Path],
    pointer_labels_exact: set,
    pointer_keywords: List[str],
    out_dir: Path,
    sample_count: int,
    seed: int,
    label_font_size: int,
    problems_only: bool = False,
) -> str:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception:
        return "skip: Pillow not installed."

    if problems_only:
        pool = [r for r in records if r.get("issues")]
    else:
        pointer_records = [r for r in records if r.get("pointer_shape_count", 0) > 0]
        fallback_records = [r for r in records if r.get("shape_count", 0) > 0]
        pool = pointer_records if pointer_records else fallback_records

    if not pool:
        return "skip: no problematic samples found." if problems_only else "skip: no labeled records to preview."

    random.seed(seed)
    selected = random.sample(pool, k=min(sample_count, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)

    font = None
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size=max(12, label_font_size))
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for rec in selected:
        stem = rec["stem"]
        img_path = image_map.get(stem)
        if not img_path:
            continue

        data = rec["raw_json"]
        shapes = data.get("shapes", [])

        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                draw = ImageDraw.Draw(img)

                for shp in shapes:
                    label = str(shp.get("label", "")).strip()
                    points = shp.get("points", [])
                    if not isinstance(points, list) or len(points) < 2:
                        continue

                    color = (
                        (255, 60, 60)
                        if is_pointer_label(label, pointer_labels_exact, pointer_keywords)
                        else (60, 220, 80)
                    )

                    pts = []
                    valid = True
                    for p in points:
                        if not (isinstance(p, list) and len(p) == 2 and is_number(p[0]) and is_number(p[1])):
                            valid = False
                            break
                        pts.append((float(p[0]), float(p[1])))
                    if not valid:
                        continue

                    shape_type = str(shp.get("shape_type", "")).lower()
                    if shape_type == "rectangle" and len(pts) >= 2:
                        x0, y0 = pts[0]
                        x1, y1 = pts[1]
                        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)

                        tx = min(x0, x1)
                        ty = min(y0, y1)
                        tb = draw.textbbox((tx, ty), label, font=font)
                        draw.rectangle(tb, fill=(0, 0, 0))
                        draw.text((tx, ty), label, fill=color, font=font)
                    else:
                        draw.polygon(pts, outline=color)

                img.save(out_dir / f"{stem}_preview.jpg", quality=95)
        except Exception:
            continue

    return f"ok: saved previews to {out_dir}"


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root.resolve()
    image_dir = (dataset_root / args.image_dir).resolve()
    label_dir = (dataset_root / args.label_dir).resolve()
    report_path = (
        args.report_json.resolve()
        if args.report_json
        else (dataset_root / "validation_report.json").resolve()
    )
    vis_dir = (
        args.vis_dir.resolve()
        if args.vis_dir
        else (dataset_root / "validation_preview").resolve()
    )

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing folder(s). image_dir={image_dir}, label_dir={label_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    image_files = [p for p in image_dir.iterdir() if p.suffix.lower() in image_exts]
    label_files = sorted(label_dir.glob("*.json"))
    image_map = {p.stem: p for p in image_files}
    label_map = {p.stem: p for p in label_files}

    image_stems = set(image_map)
    label_stems = set(label_map)
    paired_stems = sorted(image_stems & label_stems)
    missing_label = sorted(image_stems - label_stems)
    missing_image = sorted(label_stems - image_stems)

    requested_stems = load_requested_stems(args.check_stems, args.check_stems_file)
    requested_missing_in_dataset: List[str] = []
    if requested_stems:
        requested_missing_in_dataset = sorted(
            stem for stem in requested_stems if stem not in image_stems and stem not in label_stems
        )
        paired_stems = sorted(set(paired_stems) & requested_stems)
        missing_label = sorted(set(missing_label) & requested_stems)
        missing_image = sorted(set(missing_image) & requested_stems)

    pointer_labels_exact = set(args.pointer_labels)
    pointer_keywords = args.pointer_keywords

    global_labels = Counter()
    records: List[dict] = []
    file_errors = []

    total_shape_count = 0
    total_invalid_shape_count = 0
    total_out_of_bounds_points = 0
    total_zero_area = 0
    total_pointer_shapes = 0
    files_with_pointer = 0
    files_without_shapes = 0

    for stem in paired_stems:
        label_path = label_map[stem]
        image_path = image_map[stem]

        data, err = try_load_json(label_path)
        if err or data is None:
            file_errors.append({"file": str(label_path), "error": err})
            continue

        shapes = data.get("shapes", [])
        if not isinstance(shapes, list):
            file_errors.append({"file": str(label_path), "error": "shapes is not list"})
            continue

        image_size = collect_image_size(image_path)
        stats, issues, labels_counter, pointer_count = validate_shapes(
            shapes=shapes,
            image_size=image_size,
            pointer_labels_exact=pointer_labels_exact,
            pointer_keywords=pointer_keywords,
        )

        global_labels.update(labels_counter)
        total_shape_count += stats["shape_count"]
        total_invalid_shape_count += stats["invalid_shape_count"]
        total_out_of_bounds_points += stats["out_of_bounds_points"]
        total_zero_area += stats["zero_area_shapes"]
        total_pointer_shapes += pointer_count

        if pointer_count > 0:
            files_with_pointer += 1
        if stats["shape_count"] == 0:
            files_without_shapes += 1

        records.append(
            {
                "stem": stem,
                "shape_count": stats["shape_count"],
                "pointer_shape_count": pointer_count,
                "issues": issues,
                "raw_json": data,
            }
        )

    if not pointer_labels_exact:
        inferred_pointer_labels = sorted(
            [
                name
                for name in global_labels
                if contains_keyword(name, pointer_keywords) or looks_like_pointer_label(name)
            ]
        )
    else:
        inferred_pointer_labels = sorted(pointer_labels_exact)

    report_records = [
        {
            "stem": r["stem"],
            "shape_count": r["shape_count"],
            "pointer_shape_count": r["pointer_shape_count"],
            "issues": r["issues"][:50],
        }
        for r in records
    ]

    vis_message = "skip: disabled by --no-vis"
    if not args.no_vis:
        vis_message = draw_previews(
            records=records,
            image_map=image_map,
            pointer_labels_exact=pointer_labels_exact,
            pointer_keywords=pointer_keywords,
            out_dir=vis_dir,
            sample_count=args.samples,
            seed=args.seed,
            label_font_size=args.label_font_size,
            problems_only=args.vis_problems_only,
        )

    report = {
        "dataset_root": str(dataset_root),
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "summary": {
            "total_images": len(image_files),
            "total_labels": len(label_files),
            "paired_items": len(paired_stems),
            "images_missing_label": len(missing_label),
            "labels_missing_image": len(missing_image),
            "json_file_errors": len(file_errors),
            "files_without_shapes": files_without_shapes,
            "total_shapes": total_shape_count,
            "invalid_shapes": total_invalid_shape_count,
            "out_of_bounds_points": total_out_of_bounds_points,
            "zero_area_shapes": total_zero_area,
            "files_with_pointer": files_with_pointer,
            "pointer_shape_total": total_pointer_shapes,
        },
        "pointer_rules": {
            "exact_labels": sorted(pointer_labels_exact),
            "keywords": pointer_keywords,
            "resolved_pointer_labels": inferred_pointer_labels,
        },
        "top_labels": global_labels.most_common(30),
        "missing_label_examples": missing_label[:30],
        "missing_image_examples": missing_image[:30],
        "requested_stems": sorted(requested_stems)[:5000],
        "requested_missing_in_dataset": requested_missing_in_dataset[:200],
        "file_errors": file_errors[:100],
        "records": report_records[:5000],
        "visualization": vis_message,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Validation done ===")
    print(f"Dataset root: {dataset_root}")
    if requested_stems:
        print(f"Requested stems: {len(requested_stems)}, matched pairs: {len(paired_stems)}")
    print(f"Images: {len(image_files)}, Labels: {len(label_files)}, Paired: {len(paired_stems)}")
    print(
        "Missing labels: "
        f"{len(missing_label)}, Missing images: {len(missing_image)}, JSON errors: {len(file_errors)}"
    )
    if requested_missing_in_dataset:
        print(f"Requested but not in dataset: {len(requested_missing_in_dataset)}")
    print(
        "Shapes: "
        f"{total_shape_count}, Invalid: {total_invalid_shape_count}, "
        f"Out-of-bounds points: {total_out_of_bounds_points}, Zero-area: {total_zero_area}"
    )
    print(
        "Pointer: "
        f"files with pointer={files_with_pointer}, pointer shape total={total_pointer_shapes}"
    )
    print(f"Resolved pointer labels: {inferred_pointer_labels}")
    print(f"Report JSON: {report_path}")
    print(f"Visualization: {vis_message}")


if __name__ == "__main__":
    main()
