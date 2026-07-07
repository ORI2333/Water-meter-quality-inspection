import argparse
import shutil
from pathlib import Path


def export_pt_to_onnx(
    pt_path: Path,
    onnx_path: Path,
    imgsz: int | tuple[int, int],
    opset: int,
) -> None:
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ultralytics not installed. Run: pip install ultralytics") from e

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(pt_path))
    export_out = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=False,
            simplify=False,
        )
    )

    # Ultralytics may return either the ONNX file path or a directory.
    if export_out.suffix.lower() == ".onnx" and export_out.exists():
        produced = export_out
    else:
        export_dir = export_out.parent if export_out.is_file() else export_out
        produced = export_dir / f"{pt_path.stem}.onnx"
        if not produced.exists():
            candidates = list(export_dir.glob("*.onnx"))
            if not candidates:
                raise FileNotFoundError(f"Cannot find exported onnx under: {export_dir}")
            produced = candidates[0]

    shutil.copyfile(produced, onnx_path)
    print(f"ONNX saved: {onnx_path}")


def _require_rknn():
    try:
        from rknn.api import RKNN  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "rknn-toolkit2 is not available.\n"
            "Please run this step in x86_64 Linux with RKNN-Toolkit2 installed."
        ) from e
    return RKNN


def convert_onnx_to_rknn(
    onnx_path: Path,
    rknn_path: Path,
    target_platform: str,
    do_quant: bool,
    dataset_txt: Path | None,
) -> None:
    RKNN = _require_rknn()
    rknn = RKNN(verbose=True)

    # Keep preprocessing explicit in runtime script; do not bake mean/std here.
    rknn.config(target_platform=target_platform)

    print(f"Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"rknn.load_onnx failed: {ret}")

    build_kwargs = {"do_quantization": do_quant}
    if do_quant:
        if dataset_txt is None:
            raise ValueError("--dataset-txt is required when --quant is enabled.")
        build_kwargs["dataset"] = str(dataset_txt)
        print(f"Building INT8 RKNN with dataset: {dataset_txt}")
    else:
        print("Building floating-point RKNN (no quantization).")

    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError(f"rknn.build failed: {ret}")

    rknn_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed: {ret}")

    rknn.release()
    print(f"RKNN saved: {rknn_path}")
    if do_quant:
        print("[HINT] Quantized model built. Runtime should usually use uint8/int8 input.")
    else:
        print("[HINT] Floating-point model built. Runtime should prefer float32 input (with /255 normalization).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export water/module/best(1).pt to ONNX and convert to RKNN for RK3588."
    )
    parser.add_argument("--pt", default=str(Path("..") / "module" / "best(1).pt"), help="Path to .pt model")
    parser.add_argument(
        "--onnx-out",
        default=str(Path("..") / "module" / "onnx" / "best_640.onnx"),
        help="Output .onnx path",
    )
    parser.add_argument(
        "--rknn-out",
        default=str(Path("..") / "module" / "best_640_fp.rknn"),
        help="Output .rknn path",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO export size N for NxN")
    parser.add_argument("--img-h", type=int, default=0, help="Optional non-square export height")
    parser.add_argument("--img-w", type=int, default=0, help="Optional non-square export width")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset, RKNN usually uses <=12/13")
    parser.add_argument("--platform", default="rk3588", help="RKNN target platform")
    parser.add_argument("--quant", action="store_true", help="Enable INT8 quantization")
    parser.add_argument("--dataset-txt", default="", help="Calibration dataset txt for --quant")
    parser.add_argument(
        "--only-export-onnx",
        action="store_true",
        help="Only export ONNX; skip RKNN conversion",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    pt_path = (script_dir / args.pt).resolve()
    onnx_out = (script_dir / args.onnx_out).resolve()
    rknn_out = (script_dir / args.rknn_out).resolve()
    dataset_txt = (script_dir / args.dataset_txt).resolve() if args.dataset_txt else None

    if not pt_path.exists():
        raise FileNotFoundError(f"PT model not found: {pt_path}")
    if dataset_txt and not dataset_txt.exists():
        raise FileNotFoundError(f"Dataset txt not found: {dataset_txt}")

    imgsz: int | tuple[int, int] = args.imgsz
    if args.img_h and args.img_w:
        img_h = ((int(args.img_h) + 31) // 32) * 32
        img_w = ((int(args.img_w) + 31) // 32) * 32
        if (img_h, img_w) != (args.img_h, args.img_w):
            print(
                f"Input size rounded to stride-32 multiple: {(args.img_h, args.img_w)} -> {(img_h, img_w)}"
            )
        imgsz = (img_h, img_w)

    print(f"PT model: {pt_path}")
    print(f"ONNX out: {onnx_out}")
    print(f"RKNN out: {rknn_out}")

    export_pt_to_onnx(
        pt_path=pt_path,
        onnx_path=onnx_out,
        imgsz=imgsz,
        opset=args.opset,
    )

    if args.only_export_onnx:
        print("Skip RKNN conversion because --only-export-onnx is set.")
        return

    convert_onnx_to_rknn(
        onnx_path=onnx_out,
        rknn_path=rknn_out,
        target_platform=args.platform,
        do_quant=args.quant,
        dataset_txt=dataset_txt,
    )


if __name__ == "__main__":
    main()
