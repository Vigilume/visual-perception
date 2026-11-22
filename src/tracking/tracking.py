"""ByteTrack adapter for persistent tracking across frame batches."""

from pathlib import Path
from typing import Any

import yaml


class ByteTrack:
    """Run an Ultralytics tracker and expose its settings as output metadata."""

    def __init__(self, config: str = "bytetrack.yaml") -> None:
        """Resolve and read the selected tracker YAML configuration."""
        self.config = str(config)
        config_path = Path(config)
        if not config_path.is_file():
            from ultralytics.utils.checks import check_yaml

            config_path = Path(check_yaml(config))
        with config_path.open("r", encoding="utf-8-sig") as stream:
            self._settings = yaml.safe_load(stream) or {}

    def metadata(self) -> dict[str, Any]:
        """Return the tracker type, config name, and settings for each frame."""
        keys = (
            "track_high_thresh", "track_low_thresh", "new_track_thresh",
            "track_buffer", "match_thresh", "fuse_score", "with_reid",
        )
        result = {
            "type": {
                "bytetrack": "ByteTrack",
                "botsort": "BoT-SORT",
            }.get(str(self._settings.get("tracker_type", "bytetrack")).lower(),
                  str(self._settings.get("tracker_type", "bytetrack"))),
            "config": self.config,
        }
        result.update({key: self._settings[key] for key in keys if key in self._settings})
        result["with_reid"] = bool(self._settings.get("with_reid", False))
        return result

    def track_batch(
        self,
        model: Any,
        frames: list[Any],
        *,
        imgsz: int = 640,
        conf: float = 0.1,
        device: str | int = "0",
        classes: list[int] | None = None,
        agnostic_nms: bool = True,
        nms_iou: float = 0.7,
        max_detections: int = 300,
    ) -> list[Any]:
        """Track a chronological batch while retaining IDs across calls."""
        if not frames:
            return []
        results = model.track(
            source=frames,
            batch=len(frames),
            imgsz=imgsz,
            conf=conf,
            device=device,
            classes=classes,
            iou=nms_iou,
            max_det=max_detections,
            tracker=self.config,
            agnostic_nms=agnostic_nms,
            persist=True,
            verbose=False,
            stream=False,
        )
        return list(results)

    @staticmethod
    def ids_for(result: Any, count: int) -> list[int | None]:
        """Return box-aligned track IDs, using None when IDs are unavailable."""
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            return [None] * count
        ids = boxes.id.detach().cpu().numpy().astype(int).tolist()
        return (ids + [None] * count)[:count]
