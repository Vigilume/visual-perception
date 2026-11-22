"""Reusable YOLO26s detection helpers for single frames and frame batches."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.tracking.tracking import ByteTrack

PERSON_CLASS_ID = 0  # COCO class index for person.


def load_detector(weights: str | Path) -> YOLO:
    """Load the all-class YOLO detector."""
    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"YOLO detection weights not found: {weights}")
    return YOLO(str(weights))


def detect_frame(
    detector: YOLO,
    frame: Any,
    imgsz: int = 640,
    conf: float = 0.1,
    device: str | int = "0",
    classes: list[int] | None = None,
    tracker: ByteTrack | None = None,
    agnostic_nms: bool = True,
    nms_iou: float = 0.7,
    max_detections: int = 300,
) -> tuple[Any, list[dict], list[dict]]:
    """Run detection on one frame and return its results and detections."""
    results, detections, people = detect_batch(
        detector, [frame], imgsz=imgsz, conf=conf, device=device,
        classes=classes, tracker=tracker, agnostic_nms=agnostic_nms,
        nms_iou=nms_iou,
        max_detections=max_detections,
    )
    return results[0], detections[0], people[0]


def detect_batch(
    detector: YOLO,
    frames: list[Any],
    imgsz: int = 640,
    conf: float = 0.1,
    device: str | int = "0",
    classes: list[int] | None = None,
    tracker: ByteTrack | None = None,
    agnostic_nms: bool = True,
    nms_iou: float = 0.7,
    max_detections: int = 300,
) -> tuple[list[Any], list[list[dict]], list[list[dict]]]:
    """Infer over multiple frames and return aligned per-frame detections.

    If ``classes`` excludes COCO class 0, this returns only non-person classes.
    Passing a ByteTrack instance enables persistent IDs across chronological
    calls. Keep a separate tracker/model stream for each independent model.
    """
    if not frames:
        return [], [], []

    if tracker is None:
        results = detector.predict(
            source=frames, batch=len(frames), imgsz=imgsz, conf=conf,
            device=device, classes=classes, agnostic_nms=agnostic_nms,
            iou=nms_iou, max_det=max_detections, verbose=False, stream=False,
        )
        results = list(results)
    else:
        results = tracker.track_batch(
            detector, frames, imgsz=imgsz, conf=conf, device=device, classes=classes,
            agnostic_nms=agnostic_nms,
            nms_iou=nms_iou,
            max_detections=max_detections,
        )

    detections_by_frame, people_by_frame = [], []
    for result in results:
        detections, people = [], []
        boxes_obj = result.boxes
        if boxes_obj is not None:
            boxes = boxes_obj.xyxy.detach().cpu().numpy()
            class_ids = boxes_obj.cls.detach().cpu().numpy().astype(int)
            confidences = boxes_obj.conf.detach().cpu().numpy()
            track_ids = tracker.ids_for(result, len(boxes)) if tracker else [None] * len(boxes)
            for index, (box, class_id, confidence) in enumerate(zip(boxes, class_ids, confidences)):
                detection = {
                    "index": index,
                    "class_id": int(class_id),
                    "class_name": str(result.names[int(class_id)]),
                    "confidence": float(confidence),
                    "xyxy": [float(value) for value in box],
                    "track_id": track_ids[index],
                    "track_source": "yolo26s",
                }
                detections.append(detection)
                if int(class_id) == PERSON_CLASS_ID:
                    people.append(detection)
        detections_by_frame.append(detections)
        people_by_frame.append(people)

    return results, detections_by_frame, people_by_frame
