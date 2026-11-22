"""Batched full-frame YOLO Pose adapter with optional persistent ByteTrack IDs."""

from pathlib import Path
from typing import Any

import numpy as np
from ultralytics import YOLO

from src.tracking.tracking import ByteTrack


class YoloPoseAloneEstimator:
    """Detect people and estimate keypoints independently on full video frames."""

    def __init__(
        self,
        weights: str | Path | None = None,
        imgsz: int = 640,
        device: str | int = "0",
        confidence: float = 0.25,
        max_detections: int = 300,
        nms_iou: float = 0.7,
        *,
        model: Any | None = None,
        tracker: ByteTrack | None = None,
    ):
        """Load YOLO Pose and store inference and optional tracking settings."""
        if model is None:
            if weights is None:
                raise ValueError("Provide pose weights or an already-loaded YOLO model.")
            weights = Path(weights)
            if not weights.is_file():
                raise FileNotFoundError(f"YOLO pose weights not found: {weights}")
            model = YOLO(str(weights))
        self.model = model
        self.imgsz = imgsz
        self.device = device
        self.confidence = confidence
        self.max_detections = max_detections
        self.nms_iou = nms_iou
        self.tracker = tracker

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        """Predict on one full frame; no detector boxes are passed in."""
        return self.predict_batch([frame])[0]

    def predict_batch(self, frames: list[np.ndarray]) -> list[dict[str, Any]]:
        """Batch full frames through YOLO Pose, which detects its own people."""
        if not frames:
            return []

        if self.tracker is None:
            results = self.model.predict(
                source=frames, batch=len(frames), imgsz=self.imgsz,
                conf=self.confidence, device=self.device,
                max_det=self.max_detections,
                iou=self.nms_iou,
                verbose=False, stream=False,
            )
            results = list(results)
        else:
            results = self.tracker.track_batch(
                self.model, frames, imgsz=self.imgsz,
                conf=self.confidence, device=self.device,
                max_detections=self.max_detections,
                nms_iou=self.nms_iou,
                agnostic_nms=False,
            )

        outputs = []
        for result in results:
            detections: list[dict[str, Any]] = []
            poses: list[dict[str, Any]] = []
            boxes = result.boxes
            keypoints = result.keypoints
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                class_ids = boxes.cls.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                point_data = keypoints.data.detach().cpu().numpy() if keypoints is not None else None
                track_ids = self.tracker.ids_for(result, len(xyxy)) if self.tracker else [None] * len(xyxy)

                for index, (box, class_id, confidence) in enumerate(zip(xyxy, class_ids, confidences)):
                    detection = {
                        "index": index,
                        "class_id": int(class_id),
                        "class_name": str(result.names[int(class_id)]),
                        "confidence": float(confidence),
                        "xyxy": [float(value) for value in box],
                        "track_id": track_ids[index],
                        "track_source": "yolo_pose",
                    }
                    detections.append(detection)
                    if point_data is not None and index < len(point_data):
                        poses.append({
                            "detection_index": index,
                            "track_id": track_ids[index],
                            "keypoints": np.asarray(point_data[index], dtype=float),
                        })
            outputs.append({
                "result": result,
                "detections": detections,
                "people": detections,
                "poses": poses,
            })
        return outputs
