#!/usr/bin/env bash
# Runs the project end to end on a Colab or Kaggle GPU notebook.
#   bash scripts/run_all.sh smoke   build and run with the stock COCO model (no training)
#   bash scripts/run_all.sh train   download TACO, fine-tune, export
#   bash scripts/run_all.sh bench   engine vs Python baseline on one image
#   bash scripts/run_all.sh trt     add the TensorRT FP16 backend and benchmark it
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-smoke}"
BATCH="${BATCH:-8}"
EPOCHS="${EPOCHS:-60}"
REPEAT="${REPEAT:-50}"
SUDO=""
[ "$(id -u)" -eq 0 ] || SUDO="sudo"

setup() {
    pip install -q -r requirements.txt
    if ! command -v cmake >/dev/null || ! pkg-config --exists opencv4; then
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq cmake pkg-config libopencv-dev >/dev/null
    fi
}

# TensorRT libraries come from pip; the matching headers come from NVIDIA's
# open-source repository, so the two are always the same version.
trt_library() {
    python3 - <<'EOF'
import glob, sysconfig
libs = sorted(glob.glob(sysconfig.get_paths()["purelib"] + "/tensorrt*libs*/libnvinfer.so.*"))
print(libs[0] if libs else "")
EOF
}

setup_trt() {
    if ! python3 -c 'import tensorrt' 2>/dev/null; then
        local cu
        cu="$(python3 -c 'import torch; print(torch.version.cuda.split(".")[0])')"
        pip install -q "tensorrt-cu${cu}" onnx onnxslim
    fi
    # TensorRT 11 needs ModelOpt to write FP16 into the ONNX graph.
    if [ "$(python3 -c 'import tensorrt; print(tensorrt.__version__.split(".")[0])')" -ge 11 ] &&
        ! python3 -c 'import modelopt.onnx' 2>/dev/null; then
        pip install -q "nvidia-modelopt[onnx]>=0.44"
    fi
    if [ ! -d third_party/TensorRT/include ]; then
        local ver
        ver="$(python3 -c 'import tensorrt; print(".".join(tensorrt.__version__.split(".")[:2]))')"
        git clone -q --depth 1 -b "release/${ver}" https://github.com/NVIDIA/TensorRT third_party/TensorRT
    fi
}

build() {
    local extra=()
    local lib
    lib="$(trt_library)"
    if [ -d third_party/TensorRT/include ] && [ -n "$lib" ]; then
        extra+=(-DWASTE_WITH_TENSORRT=ON
                -DTENSORRT_INCLUDE_DIR="$PWD/third_party/TensorRT/include"
                -DTENSORRT_LIBRARY="$lib")
    fi
    cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
        "${extra[@]}"
    cmake --build build -j"$(nproc)"
}

# run_engine MODEL IMAGE TAG
run_engine() {
    mkdir -p out
    ./build/waste_engine --model "$1" --classes "${1%.*}.names" --input "$2" \
        --batch "$BATCH" --repeat "$REPEAT" \
        --json "out/engine_$3.json" --out "out/annotated_$3.jpg"
}

# compare TORCHSCRIPT_MODEL IMAGE: engine and baseline on the same model file,
# plus the TensorRT engine next to it when there is one.
compare() {
    local model="$1" image="$2"
    mkdir -p out
    echo "== C++ engine, TorchScript FP32 =="
    run_engine "$model" "$image" torchscript
    echo "== Python baseline, TorchScript FP32 =="
    python3 python/baseline.py --model "$model" --input "$image" --batch "$BATCH" \
        --repeat "$REPEAT" --json out/baseline.json
    echo "== engine (TorchScript) vs baseline =="
    python3 tools/compare.py out/engine_torchscript.json out/baseline.json
    if [ -f "${model%.torchscript}.engine" ]; then
        echo "== C++ engine, TensorRT =="
        run_engine "${model%.torchscript}.engine" "$image" tensorrt
        echo "== engine (TensorRT) vs baseline =="
        python3 tools/compare.py out/engine_tensorrt.json out/baseline.json
    fi
}

sample_image() {
    python3 -c "from ultralytics.utils import ASSETS; print(ASSETS / 'bus.jpg')"
}

case "$STAGE" in
smoke)
    setup
    python3 train/export.py --weights yolo11s.pt --out models/coco.torchscript --batch "$BATCH"
    build
    compare models/coco.torchscript "${IMAGE:-$(sample_image)}"
    ;;
train)
    setup
    # MIN_FRACTION=0 trains on whatever was downloaded instead of stopping;
    # MAX_WAIT caps the seconds spent waiting on a throttling image host.
    python3 train/prepare_taco.py --min-fraction "${MIN_FRACTION:-0.9}" \
        --max-wait "${MAX_WAIT:-1800}"
    python3 train/train.py --epochs "$EPOCHS"
    python3 train/export.py --batch "$BATCH"
    ;;
bench)
    [ -x build/waste_engine ] || { setup; build; }
    IMAGE="${IMAGE:-$(ls data/taco/images/val/*.jpg | head -n 1)}"
    compare "${MODEL:-models/waste.torchscript}" "$IMAGE"
    ;;
trt)
    setup
    setup_trt
    # Uses the trained waste model when there is one, the stock model otherwise.
    if [ -f models/waste.pt ]; then
        WEIGHTS=models/waste.pt NAME=waste
        IMAGE="${IMAGE:-$(ls data/taco/images/val/*.jpg | head -n 1)}"
    else
        WEIGHTS=yolo11s.pt NAME=coco
        IMAGE="${IMAGE:-$(sample_image)}"
    fi
    if [ ! -f "models/$NAME.torchscript" ]; then
        python3 train/export.py --weights "$WEIGHTS" --out "models/$NAME.torchscript" --batch "$BATCH"
    fi
    # FP32=1 builds a full-precision engine instead of FP16.
    python3 train/build_trt.py --weights "$WEIGHTS" --out "models/$NAME.engine" --batch "$BATCH" \
        ${FP32:+--fp32}
    build
    compare "models/$NAME.torchscript" "$IMAGE"
    ;;
*)
    echo "unknown stage: $STAGE (use smoke, train, bench or trt)" >&2
    exit 1
    ;;
esac
