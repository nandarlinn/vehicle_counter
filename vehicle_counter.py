#!/usr/bin/env python3
"""
Vehicle counting script powered by Ultralytics YOLO11 tracking.

This module now follows a minimal MVC structure:
  * Model (`VehicleCounterModel`) loads YOLO11 and streams tracked detections.
  * View (`VehicleCounterView`) draws overlays and displays real-time frames.
  * Controller (`VehicleCounterController`) coordinates model + view to count vehicles.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Union

import cv2
from ultralytics import YOLO

# COCO class IDs that correspond to vehicles we care about.
VEHICLE_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count vehicles in a video using Ultralytics YOLO11 tracking."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file or camera index (e.g., 0).",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="YOLO11 model checkpoint to use (default: yolo11n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    return parser.parse_args()


def resolve_video_source(raw: str) -> Union[int, Path]:
    """Return an integer camera index or a validated file path."""
    if raw.isdigit():
        return int(raw)
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


class VehicleCounterModel:
    """Model layer responsible for running YOLO11 tracking."""

    def __init__(self, model_path: str, conf: float) -> None:
        self._model = YOLO(model_path)
        self._conf = conf

    def track(self, source: Union[int, Path]) -> Iterable:
        src = source if isinstance(source, int) else str(source)
        return self._model.track(
            source=src,
            stream=True,
            tracker="bytetrack.yaml",
            classes=list(VEHICLE_CLASS_MAP.keys()),
            conf=self._conf,
            verbose=False,
        )


class VehicleCounterView:
    """View layer that draws overlays and displays frames in real time."""

    def __init__(self, window_name: str = "Vehicle Counter") -> None:
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def _overlay_counts(self, frame, counts: Dict[str, int]) -> None:
        y = 30
        for label in sorted(counts.keys()):
            text = f"{label.title()}: {counts[label]}"
            cv2.putText(
                frame,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 30

        total = sum(counts.values())
        cv2.putText(
            frame,
            f"Total: {total}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 215, 255),
            2,
            cv2.LINE_AA,
        )

    def render(self, result_frame, counts: Dict[str, int]) -> None:
        frame = result_frame
        self._overlay_counts(frame, counts)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            raise KeyboardInterrupt("User requested exit (q pressed).")

    def close(self) -> None:
        cv2.destroyWindow(self.window_name)


class VehicleCounterController:
    """Controller layer orchestrating the full counting workflow."""

    def __init__(
        self,
        model: VehicleCounterModel,
        view: VehicleCounterView,
    ) -> None:
        self.model = model
        self.view = view
        self.counts: Dict[str, int] = {label: 0 for label in VEHICLE_CLASS_MAP.values()}
        self._counted_track_ids: Dict[int, str] = {}

    def run(self, video_source: Union[int, Path]) -> None:
        for result in self.model.track(video_source):
            boxes = result.boxes
            if boxes.id is None:
                continue

            track_ids = boxes.id.int().tolist()
            class_ids = boxes.cls.int().tolist()

            for track_id, class_id in zip(track_ids, class_ids):
                if class_id not in VEHICLE_CLASS_MAP:
                    continue
                if track_id in self._counted_track_ids:
                    continue
                label = VEHICLE_CLASS_MAP[class_id]
                self._counted_track_ids[track_id] = label
                self.counts[label] += 1

            frame = result.plot()
            self.view.render(frame, self.counts)

    def summarize(self) -> None:
        print("Vehicle counts:")
        for label in sorted(self.counts.keys()):
            print(f"  {label.title():<12} {self.counts[label]}")
        print(f"  {'Total':<12} {sum(self.counts.values())}")


def main() -> None:
    args = parse_args()

    video_source = resolve_video_source(args.video)

    model = VehicleCounterModel(args.model, args.conf)
    view = VehicleCounterView()
    controller = VehicleCounterController(model, view)

    try:
        controller.run(video_source)
    except KeyboardInterrupt as exc:
        print(str(exc))
    finally:
        view.close()

    controller.summarize()

if __name__ == "__main__":
    main()

