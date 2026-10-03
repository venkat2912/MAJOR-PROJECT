"""Builds a TensorRT engine for the C++ engine's TensorRT backend.

The engine is tied to the GPU model and TensorRT version it is built with, so
build it on the machine that will run it.
"""
import argparse
from pathlib import Path

import tensorrt as trt
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/waste.pt")
    p.add_argument("--out", default="models/waste.engine")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--fp32", action="store_true", help="keep full precision instead of FP16")
    args = p.parse_args()

    model = YOLO(args.weights)
    onnx_path = model.export(format="onnx", imgsz=args.size, batch=args.batch)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"cannot parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if not args.fp32:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"building {'FP32' if args.fp32 else 'FP16'} engine with TensorRT {trt.__version__} "
          "(this takes a few minutes) ...")
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("TensorRT failed to build the engine")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(blob)
    names = model.names
    out.with_suffix(".names").write_text("\n".join(names[i] for i in sorted(names)) + "\n")
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB, batch {args.batch})")


if __name__ == "__main__":
    main()
