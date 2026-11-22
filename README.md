# Visual Perception Pipeline

This project processes images or videos with two independent Ultralytics models and writes one machine-readable JSON record per source frame. It combines non-person object detection, person detection with pose estimation, and optional ByteTrack IDs in a JSON Lines (`.jsonl`) output.

## 1. Pipeline overview

For each decoded frame, `run_pipeline.py`:

1. Reads the image/video and preserves the source frame index.
2. Runs the YOLO26s detector for COCO classes **except `person`**. Class agnostic NMS is enabled. The default confidence threshold is `0.10` (the runner allows `0.10–0.15`).
3. Independently runs YOLO Pose on the same full frame. It detects people itself and estimates 17 COCO body keypoints. The default person threshold is `0.25`.
4. For videos, applies two separate persistent ByteTrack streams: one for non-person objects and one for people. Images are not tracked. Tracker IDs are independent between streams; ByteTrack does not provide ReID embeddings here.
5. Writes processed frames as JSONL. If the video decoder reports a frame-index gap, the output includes explicit `skipped` records for the missing indices.

Both models process the same ordered frame batches. The main command writes `<input-name>_predictions.jsonl` into the selected output directory. `predict_media.py` is a separate convenience script that saves an annotated YOLO detection, tracking and posing video; it does not produce the pipeline JSONL schema.

## 2. Output schema

The contract is [`schemas/frame_record.schema.json`](schemas/frame_record.schema.json). Each line is a complete JSON object; metadata is repeated on every line.

| Field | Meaning |
|---|---|
| `schema_version` | Currently `"1.0"`. |
| `status` | `"processed"` or `"skipped"`. |
| `camera_id` | Stable source/camera identifier supplied by the caller. |
| `frame_id`, `timestamp` | Source frame index and Unix epoch seconds. For video: start epoch + frame index / source FPS. For a still image: supplied capture epoch. |
| `timestamp_source` | Indicates which timestamp rule was used. |
| `frame_size` | `[width, height]` in pixels. |
| `frame_gap_before`, `skipped_frames_before` | Whether and how many source indices were missing before this record. |
| `model_info` | Model weight paths, input dimensions, confidence/NMS settings, classes and keypoint convention. |
| `tracker_info` | Separate tracker types, configuration and settings for objects and people; `enabled` is false for still images. |
| `models_run` | Whether detector and pose inference ran for this record. Both are false for skipped frames. |
| `detections` | Non-person YOLO26s object detections. |
| `people` | People detected by the independent YOLO Pose model. |
| `poses` | One keypoint result per pose detection, linked to the matching person by `detection_index`. |
| `skip_reason` | Present on skipped records; currently `"decoder_frame_gap"`. |

Each detection/person object has `index`, `class_id`, `class_name`, `confidence`, `xyxy`, `track_id`, and `track_source`. `xyxy` is `[x1, y1, x2, y2]` in original-frame pixels. A missing track ID is JSON `null`. The non-person detector uses COCO class IDs 1–79; bag-related IDs are backpack `24`, handbag `26`, suitcase `28`. Person entries use class ID `0` and source `yolo_pose`.

Each pose has `detection_index`, `track_id`, and `keypoints`. `keypoints` has exactly 17 `[x, y, confidence]` triples in original-frame pixel coordinates, ordered as `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`.

JSON Schema checks record shapes and types. The supplied validator also checks pose-to-person links, unique person indices, increasing frame IDs, reported frame gaps, monotonic timestamps, and a constant camera ID. Run it with:

```bash
python tools/validate_jsonl.py results/example_predictions.jsonl
```

## 3. Requirements and model files

- Python **3.10+** for the project code (Python 3.11 or 3.12 is a practical choice for broader TensorRT wheel availability).
- Packages in [`requirements.txt`](requirements.txt): `ultralytics`, `opencv-python-headless`, `numpy`, `PyYAML`, and `jsonschema`.
- For GPU inference: an NVIDIA GPU, working NVIDIA driver, CUDA-enabled PyTorch build, and CUDA/TensorRT versions compatible with that driver and GPU. CPU inference is available without TensorRT.
- Detector weights: `models/detection/yolo26s.pt`.
- Pose weights: `models/pose/yolo26s-pose.pt`.

The ZIP includes model files, but model weights and generated TensorRT `.engine` files are excluded from Git. Keep the weights at those paths or provide paths with `--detector-weights` and `--pose-weights`.

## 4. Run locally

### 4.1 Get the project and create an environment

Extract the project ZIP and open a terminal in its root directory (the directory containing `run_pipeline.py`). Then create and activate a virtual environment.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 4.2 Install PyTorch and project requirements

Install the PyTorch build suited to the machine **first**. For NVIDIA GPU use, select the CUDA build supported by the installed driver using the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). For CPU-only use, choose the CPU build. Then install project packages:

```bash
python -m pip install -r requirements.txt
```

Check GPU availability when using CUDA:

```bash
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"
nvidia-smi
```

### 4.3 Run an image or video

Use the actual Unix epoch capture time for the image or the video's first frame. For videos, frame timestamps are calculated from this time and the source FPS. If video FPS metadata is missing or incorrect, pass `--source-fps`.

```bash
python run_pipeline.py data/example.mp4 \
  --camera-id camera_01 \
  --capture-start-epoch 1795363200.0 \
  --detector-weights models/detection/yolo26s.pt \
  --pose-weights models/pose/yolo26s-pose.pt \
  --person-tracker configs/trackers/bytetrack_buffer_60.yaml \
  --object-tracker bytetrack.yaml \
  --output results
```

PowerShell equivalent (use backticks for line continuation):

```powershell
python run_pipeline.py data/example.mp4 `
  --camera-id camera_01 `
  --capture-start-epoch 1795363200.0 `
  --detector-weights models/detection/yolo26s.pt `
  --pose-weights models/pose/yolo26s-pose.pt `
  --person-tracker configs/trackers/bytetrack_buffer_60.yaml `
  --object-tracker bytetrack.yaml `
  --output results
```

For a still image, give its capture epoch in the same argument and use an image path. Useful options:

| Option | Default | Purpose |
|---|---:|---|
| `--device` | `0` | Ultralytics device (`0` for first GPU, or `cpu`). |
| `--imgsz` | `640` | Inference image size. |
| `--batch-size` | `4` | Frames per inference batch; must be at least 1. |
| `--det-conf` | `0.10` | Object confidence threshold (allowed range `0.10–0.15`). |
| `--pose-conf` | `0.25` | Person confidence threshold (minimum `0.25`). |
| `--nms-iou` | `0.70` | NMS IoU threshold. |
| `--max-detections` | `300` | Maximum detections per frame. |
| `--start-frame` | `0` | Starting source frame index. Output IDs retain source numbering. |
| `--max-frames` | all | Limit processed source frames. |
| `--source-fps` | video metadata | Override the video's reported frame rate. |
| `--tracker` | built-in defaults | Set one tracker config for both streams; `--object-tracker` and `--person-tracker` can override separately. |

Available person tracker configs use buffers of 30, 60, or 90 **processed tracker frames** (not seconds): `configs/trackers/bytetrack_buffer_{30,60,90}.yaml`. The object tracker defaults to Ultralytics' built-in `bytetrack.yaml`.

The command prints a summary with processed/skipped counts, inference and end-to-end FPS, elapsed time, and output path.

## 5. TensorRT GPU optimization (optional)

TensorRT can improve NVIDIA GPU inference speed. It is optional: `.pt` weights run directly through PyTorch. TensorRT requires an NVIDIA GPU and compatible driver/runtime. Build engines on the target GPU, with the TensorRT and Ultralytics versions intended for inference; serialized engines are hardware/runtime/platform dependent and should be rebuilt when those change. See the [Ultralytics TensorRT guide](https://docs.ultralytics.com/integrations/tensorrt/) and the [NVIDIA TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).

### 5.1 Install TensorRT

First get the CUDA-enabled PyTorch installation working and verify `torch.cuda.is_available()` and `nvidia-smi`. Install the TensorRT Python package matching the CUDA **major version** reported by `torch.version.cuda`. For example:

```bash
# CUDA 12.x PyTorch
python -m pip install --upgrade tensorrt-cu12

# CUDA 13.x PyTorch (use instead of the CUDA 12 command above)
python -m pip install --upgrade tensorrt-cu13
```

On Windows, follow NVIDIA's current [TensorRT pip installation guide](https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-pip.html) for supported Python, driver and wheel combinations. On Linux, the same CUDA-matched pip wheel is the straightforward Python setup. Jetson users should use the TensorRT version supplied for their JetPack release. Confirm the installation:

```bash
python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"
```

Do not install a TensorRT wheel solely because it is the newest version: its CUDA variant, Python version, OS, GPU, and driver must fit the environment.

### 5.2 Export both project models to FP16 engines

Export on the GPU that will run inference. These settings match the project defaults and the provided Colab notebook (`imgsz=640`, dynamic batch up to 4, FP16):

```python
from ultralytics import YOLO

detector_engine = YOLO("models/detection/yolo26s.pt").export(
    format="engine", imgsz=640, batch=4, dynamic=True,
    half=True, device=0, workspace=4,
)
pose_engine = YOLO("models/pose/yolo26s-pose.pt", task="pose").export(
    format="engine", imgsz=640, batch=4, dynamic=True,
    half=True, device=0, workspace=4,
)
print(detector_engine, pose_engine)
```

The exported engine paths are printed; pass them to `--detector-weights` and `--pose-weights`. An engine with dynamic batch size 4 supports batch sizes from 1 through 4 within its exported shape profile. Keep `--imgsz` and `--batch-size` within the engine's profile. If export fails, verify the GPU/runtime setup and consult the compatibility and troubleshooting notes in the Ultralytics guide.

Example using the resulting engines:

```bash
python run_pipeline.py data/example.mp4 \
  --camera-id camera_01 \
  --capture-start-epoch 1795363200.0 \
  --detector-weights models/detection/yolo26s.engine \
  --pose-weights models/pose/yolo26s-pose.engine \
  --output results --device 0 --imgsz 640 --batch-size 4
```

### 5.3 Colab step by step

The project includes [`Notebook/Colab_TensorRT_Test.ipynb`](Notebook/Colab_TensorRT_Test.ipynb), which automates the steps below, including GPU checks, CUDA-matched TensorRT install, FP16 export/cache, pipeline execution, JSONL validation, and copying the result to Drive.

1. Upload/extract the project to Google Drive at `MyDrive/visual_perception`. Put the video/image in `work/` and confirm both `.pt` weight files exist in `models/`.
2. Open `Notebook/Colab_TensorRT_Test.ipynb` in Colab.
3. Select **Runtime → Change runtime type → GPU**, reconnect, then run the GPU check cell. Stop if `torch.cuda.is_available()` is false.
4. Run the Drive mount/settings cell. Change `INPUT_DRIVE`, `CAMERA_ID`, and `CAPTURE_START_EPOCH` to match your input and real capture time. Adjust `MAX_FRAMES` if you want a shorter run.
5. Run dependency/TensorRT installation. The notebook retains Colab's existing CUDA-enabled PyTorch and chooses `tensorrt-cu12` or `tensorrt-cu13` based on its CUDA major version. A fresh Colab runtime needs installation again.
6. Run the project staging cell. It copies code, input media, and model files to `/content` so inference does not read frame by frame from mounted Drive.
7. Run the engine cell. It exports both models to dynamic FP16 TensorRT engines, or reuses a cached engine only when its weights, task, settings, GPU, TensorRT, and Ultralytics versions match.
8. Run the pipeline cell. The default run is limited to 600 frames; set `MAX_FRAMES = None` to process the complete input. Output is created in the configured Drive output folder.
9. Run the final validation/copy cell. It validates the JSONL and saves the result under `MyDrive/visual_perception/outputs/colab_tensorrt/`.

The generated `.engine` files are specific to their build environment. The notebook rebuilds or selects a matching cache when Colab's GPU or runtime stack changes.

## Project layout

```text
visual_perception/
├── configs/trackers/       # ByteTrack buffer variants
├── models/                 # Detector/pose weights (local files)
├── Notebook/               # Colab + TensorRT workflow
├── schemas/                # JSON Schema for frame records
├── src/detection/          # Object detection adapter
├── src/pose/               # Full-frame YOLO Pose adapter
├── src/tracking/           # Persistent ByteTrack adapter
├── tools/validate_jsonl.py # JSONL/schema/link/order validation
├── run_pipeline.py         # Main JSONL pipeline
├── predict_media.py        # Annotated YOLO-only helper
└── requirements.txt
```

