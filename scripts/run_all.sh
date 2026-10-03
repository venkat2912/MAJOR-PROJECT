#!/usr/bin/env bash
# Runs the project end to end on a Colab or Kaggle GPU notebook.
#   bash scripts/run_all.sh smoke   build and run with the stock COCO model (no training)
#   bash scripts/run_all.sh train   download TACO, fine-tune, export
#   bash scripts/run_all.sh bench   engine vs Python baseline on one image
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

build() {
    cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"
    cmake --build build -j"$(nproc)"
}

compare() {
    local model="$1" image="$2"
    mkdir -p out
    ./build/waste_engine --model "$model" --classes "${model%.torchscript}.names" \
        --input "$image" --batch "$BATCH" --repeat "$REPEAT" \
        --json out/engine.json --out out/annotated.jpg
    python3 python/baseline.py --model "$model" --input "$image" --batch "$BATCH" \
        --repeat "$REPEAT" --json out/baseline.json
    python3 tools/compare.py out/engine.json out/baseline.json
}

case "$STAGE" in
smoke)
    setup
    python3 train/export.py --weights yolo11s.pt --out models/coco.torchscript --batch "$BATCH"
    build
    IMAGE="${IMAGE:-$(python3 -c "from ultralytics.utils import ASSETS; print(ASSETS / 'bus.jpg')")}"
    compare models/coco.torchscript "$IMAGE"
    ;;
train)
    setup
    python3 train/prepare_taco.py
    python3 train/train.py --epochs "$EPOCHS"
    python3 train/export.py --batch "$BATCH"
    ;;
bench)
    [ -x build/waste_engine ] || { setup; build; }
    IMAGE="${IMAGE:-$(ls data/taco/images/val/*.jpg | head -n 1)}"
    compare "${MODEL:-models/waste.torchscript}" "$IMAGE"
    ;;
*)
    echo "unknown stage: $STAGE (use smoke, train or bench)" >&2
    exit 1
    ;;
esac
