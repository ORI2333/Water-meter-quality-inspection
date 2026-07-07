import argparse
import json
from pathlib import Path

import cv2


def collect_images(source_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def main() -> None:
    parser = argparse.ArgumentParser(description="Use a .pt model to validate images in water/val.")
    parser.add_argument("--weights", default=str(Path("..") / "module" / "best(1).pt"), help="Path to .pt model")
    parser.add_argument("--source", default=str(Path("..") / "val"), help="Image directory to validate")
    parser.add_argument("--out-dir", default=str(Path("..") / "val_best_check"), help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="", help="Device: cpu/cuda/0/0,1 ...")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics") from e

    script_dir = Path(__file__).resolve().parent
    weights = (script_dir / args.weights).resolve()
    source = (script_dir / args.source).resolve()
    out_dir = (script_dir / args.out_dir).resolve()
    vis_dir = out_dir / "images"
    vis_dir.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        raise FileNotFoundError(f"Model not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source dir not found: {source}")

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No images found in: {source}")

    model = YOLO(str(weights))
    summary = []

    print(f"Model: {weights}")
    print(f"Source: {source}")
    print(f"Images: {len(images)}")
    print(f"Output: {out_dir}")

    for idx, img_path in enumerate(images, start=1):
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        r0 = results[0]
        annotated = r0.plot()
        out_img = vis_dir / img_path.name
        cv2.imwrite(str(out_img), annotated)

        item = {"file": img_path.name, "boxes": []}
        if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
            xyxy = r0.boxes.xyxy.detach().cpu().tolist()
            confs = r0.boxes.conf.detach().cpu().tolist()
            clss = r0.boxes.cls.detach().cpu().tolist()
            names = getattr(r0, "names", {}) or {}
            for box, cf, cls_id in zip(xyxy, confs, clss):
                cls_int = int(cls_id)
                item["boxes"].append(
                    {
                        "cls_id": cls_int,
                        "cls_name": names.get(cls_int, str(cls_int)),
                        "conf": float(cf),
                        "xyxy": [float(v) for v in box],
                    }
                )

        summary.append(item)
        print(f"[{idx}/{len(images)}] {img_path.name}: {len(item['boxes'])} boxes")

    summary_path = out_dir / "predictions.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved visualizations to: {vis_dir}")
    print(f"Done. Saved summary json to: {summary_path}")


if __name__ == "__main__":
    main()
