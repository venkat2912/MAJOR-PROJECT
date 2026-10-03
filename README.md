# MAJOR-PROJECT
AI BASED ENGINE FOR WASTE CLASSIFICATION

A GPU inference engine for drone cleanliness surveys of ghats and water-body
banks. A high-resolution frame is cut into overlapping tiles, each tile goes
through a waste detector, and the results are merged into per-class counts and
a coverage score for the frame.

Python is used only to train and export the model. Everything that runs on a
frame is C++ and CUDA.

## Layout

| Path | What it is |
|---|---|
| `engine/src/gpu_pipeline.cu` | CUDA kernels: fused tile preprocessing, output decoding, NMS mask, coverage |
| `engine/src/main.cpp` | C++ engine: tiling, LibTorch inference on the kernel's buffer, timing, output |
| `python/baseline.py` | The same pipeline in plain Python, used as the benchmark baseline |
| `tools/compare.py` | Checks the two agree and prints the per-stage speedup |
| `train/` | TACO download and conversion, fine-tuning, TorchScript export |
| `scripts/run_all.sh` | End-to-end driver for Colab and Kaggle |
| `CNN.py`, `yolo.cfg`, `coco.names` | The original YOLOv3 demo, kept for reference |

## Running on Colab or Kaggle

Use a GPU runtime (on Kaggle, also turn internet on).

```
!git clone https://github.com/venkat2912/MAJOR-PROJECT.git
%cd MAJOR-PROJECT
!bash scripts/run_all.sh smoke
```

`smoke` needs no training: it exports the stock COCO model, builds the engine,
runs it and the baseline on a sample image, and prints the comparison.

```
!bash scripts/run_all.sh train
!bash scripts/run_all.sh bench
```

`train` downloads TACO, fine-tunes YOLO11s on 6 material classes (plastic,
paper/cardboard, metal, glass, organic, other) and exports it. `bench` compares
the engine with the baseline on a validation image. `EPOCHS`, `BATCH`, `REPEAT`
and `IMAGE` can be set as environment variables.

The training set is each image plus the 640-pixel tiles the engine would cut
from it. The image host throttles bulk downloads, so `train` stops before
training if fewer than 90% of the images arrived; run it again and it fetches
only the missing ones, or set `MIN_FRACTION=0` to train on what is there.

```
!bash scripts/run_all.sh trt
```

`trt` installs TensorRT, builds an FP16 engine from the model, rebuilds the C++
engine with the TensorRT backend and benchmarks it next to the TorchScript one.
A `.engine` file passed as `--model` runs through TensorRT; anything else runs
through TorchScript.

## Running the engine directly

```
./build/waste_engine --model models/waste.torchscript --classes models/waste.names \
    --input survey.jpg --out annotated.jpg --json result.json
```

A video file as `--input` gives per-frame scores with `--csv scores.csv`.
`--batch` must match the batch size the model was exported with.

## How a frame is processed

1. The frame is copied to the GPU once.
2. One kernel crops every tile, resizes and letterboxes it, converts BGR to
   RGB, normalises and writes it in CHW layout into the batch buffer.
3. LibTorch runs the model on that buffer without copying it.
4. A kernel thresholds the raw output and maps boxes back to frame coordinates.
5. Candidates from all tiles are sorted on the GPU and a kernel builds the
   pairwise overlap mask; the final scan over the mask runs on the CPU.
6. A kernel measures the fraction of the frame covered by the kept boxes.

## Limits

- Coverage is measured from boxes, so it overstates the true waste area. Masks
  from a segmentation model would fix that.
- TACO is ground-level litter. Drone footage and ghat-specific waste such as
  floral offerings need local images added to the training set.
- INT8 quantisation, pinned memory with CUDA streams, and a PyTorch extension
  for the kernels come next.
