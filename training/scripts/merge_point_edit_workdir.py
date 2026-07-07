#!/usr/bin/env python3
"""
Merge edited point-only Labelme JSON files back into the full Labelme JSON set.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    base = root / "data" / "original_dataset"
    parser = argparse.ArgumentParser(description="Merge edited center/tip point coordinates back to full JSON.")
    parser.add_argument("--full-work-dir", type=Path, default=base / "labelme_pose_work_copy")
    parser.add_argument("--edit-dir", type=Path, required=True)
    parser.add_argument("--label", required=True, choices=("center", "tip"))
    parser.add_argument("--backup", action="store_true", help="Backup full JSON dir before merging.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def point_map(data: dict, label: str) -> Dict[int, Tuple[float, float]]:
    out = {}
    for s in data.get("shapes", []):
        if s.get("shape_type") != "point" or s.get("label") != label:
            continue
        gid = s.get("group_id")
        pts = s.get("points") or []
        if isinstance(gid, int) and pts:
            out[gid] = (float(pts[0][0]), float(pts[0][1]))
    return out


def backup_dir(src: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = src.with_name(f"{src.name}_backup_{stamp}")
    shutil.copytree(src, dst)
    return dst


def main() -> None:
    args = parse_args()
    full_dir = args.full_work_dir.resolve()
    edit_dir = args.edit_dir.resolve()
    if not full_dir.exists():
        raise FileNotFoundError(full_dir)
    if not edit_dir.exists():
        raise FileNotFoundError(edit_dir)

    if args.backup and not args.dry_run:
        print(f"Backup: {backup_dir(full_dir)}")

    files = sorted(edit_dir.glob("*.json"))
    updated_files = 0
    updated_points = 0
    missing_full = []
    missing_points = []

    for edit_json in files:
        full_json = full_dir / edit_json.name
        if not full_json.exists():
            missing_full.append(edit_json.name)
            continue

        edited = point_map(read_json(edit_json), args.label)
        if not edited:
            missing_points.append(edit_json.name)
            continue

        full_data = read_json(full_json)
        changed = 0
        for s in full_data.get("shapes", []):
            if s.get("shape_type") != "point" or s.get("label") != args.label:
                continue
            gid = s.get("group_id")
            if not isinstance(gid, int) or gid not in edited:
                continue
            x, y = edited[gid]
            old = s.get("points") or []
            if old and len(old[0]) == 2 and float(old[0][0]) == x and float(old[0][1]) == y:
                continue
            s["points"] = [[x, y]]
            changed += 1

        if changed:
            updated_files += 1
            updated_points += changed
            if not args.dry_run:
                full_json.write_text(json.dumps(full_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Merge point edit done ===")
    print(f"Label: {args.label}")
    print(f"Edit dir: {edit_dir}")
    print(f"Full dir: {full_dir}")
    print(f"Input files: {len(files)}")
    print(f"Updated files: {updated_files}")
    print(f"Updated points: {updated_points}")
    print(f"Missing full JSON: {len(missing_full)}")
    print(f"Missing edited points: {len(missing_points)}")
    if args.dry_run:
        print("Dry run only, no files changed.")


if __name__ == "__main__":
    main()
