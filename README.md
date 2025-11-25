# Vehicle Counting with Ultralytics YOLO11

This repo contains a minimal Python pipeline that uses **Ultralytics YOLO11** to count vehicles (cars, motorcycles, buses, and trucks) in a video stream. The script keeps track IDs provided by the built-in ByteTrack tracker so each vehicle is counted exactly once while the video plays.

## Prerequisites

- Python 3.9+
- ffmpeg installed on the system (required by OpenCV for some codecs)

## Setup

```bash
cd /home/elio/my_workspace/object_counting
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python vehicle_counter.py \
  --video path/to/input.mp4 \
  --model yolo11n.pt \
  --conf 0.25
```

Arguments:

- `--video` (required) path to the input video or camera stream. Pass `0` for a webcam or capture card index.
- `--model` (optional) YOLO11 checkpoint to use. Defaults to the lightweight `yolo11n.pt`. Any Ultralytics YOLO11 variant is supported.
- `--conf` (optional) confidence threshold (default `0.25`).

While the video plays, an OpenCV window titled “Vehicle Counter” shows the detections and running totals. Press `q` to stop the stream; no annotated video file is written. A summary of counts is printed when the stream ends.

## Architecture (MVC)

The application now follows a lightweight Model-View-Controller layout:

1. **Model (`VehicleCounterModel`)** loads the YOLO11 checkpoint and streams ByteTrack-enabled detections for the desired COCO vehicle classes.
2. **View (`VehicleCounterView`)** overlays running totals on each frame and displays them in real time (press `q` to stop).
3. **Controller (`VehicleCounterController`)** keeps track IDs, updates per-class counts once per vehicle, and orchestrates the full pipeline before printing the summary.

## Notes

- The default COCO-trained checkpoints already know about common vehicles. For domain-specific videos, swap in a fine-tuned YOLO11 model via `--model`.
- Tracking quality depends heavily on video resolution, lighting, and occlusion. Try different YOLO11 variants (`yolo11s.pt`, `yolo11m.pt`, etc.) and adjust `--conf` for your footage.

