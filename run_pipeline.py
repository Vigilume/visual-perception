"""Run YOLO26s object detection and independent YOLO Pose as JSONL."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterator

import cv2

from src.detection.detection import detect_batch, load_detector
from src.pose.yolo_pose import YoloPoseAloneEstimator
from src.tracking.tracking import ByteTrack


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".webm"}
BAG_CLASS_IDS = [24, 26, 28]
COCO17_KEYPOINT_ORDER = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def iter_frame_batches(
    source: Path,
    batch_size: int,
    capture_start_epoch: float,
    start_frame: int = 0,
    max_frames: int | None = None,
    source_fps: float | None = None,
) -> Iterator[list[tuple[int, Any, float, int, float]]]:
    """Read an image or video in order and report any frame-index gaps."""
    suffix = source.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Could not read image: {source}")
        yield [(0, image, float(capture_start_epoch), 0, 0.0)]
        return
    if suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported input type: {suffix}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    try:
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = float(source_fps or reported_fps)
        if fps <= 0:
            raise ValueError("Video FPS is unavailable; provide --source-fps.")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame < 0 or (total_frames > 0 and start_frame >= total_frames):
            raise ValueError(f"Invalid start frame {start_frame} for {total_frames}-frame video.")
        if start_frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError(f"Could not seek to video frame {start_frame}.")

        batch: list[tuple[int, Any, float, int, float]] = []
        next_expected_id = start_frame
        decoded_count = 0
        while max_frames is None or decoded_count < max_frames:
            ok, frame = capture.read()
            if not ok:
                if total_frames > 0 and next_expected_id < total_frames:
                    if max_frames is None or decoded_count < max_frames:
                        raise RuntimeError(
                            f"Video decoding stopped at frame {next_expected_id}; "
                            f"the file reports {total_frames} frames."
                        )
                break

            next_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            decoded_id = int(round(next_position)) - 1 if next_position > 0 else next_expected_id
            decoded_id = max(decoded_id, next_expected_id)
            gap_before = decoded_id - next_expected_id
            timestamp = float(capture_start_epoch + decoded_id / fps)
            batch.append((decoded_id, frame, timestamp, gap_before, fps))
            next_expected_id = decoded_id + 1
            decoded_count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
    finally:
        capture.release()


def non_person_class_ids(detector: Any) -> list[int]:
    """Return the detector's COCO class IDs other than person."""
    names = detector.names
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    return [int(class_id) for class_id, name in entries if str(name).lower() != "person"]


def build_model_info(args: argparse.Namespace, detector_weights: Path,
                     pose_weights: Path) -> dict[str, Any]:
    """Build the per-frame model metadata required by the JSONL schema."""
    return {
        "both_models_every_frame": True,
        "detector": {
            "weights": str(detector_weights),
            "input_size": [args.imgsz, args.imgsz],
            "confidence_threshold": args.det_conf,
            "max_detections": args.max_detections,
            "class_policy": "all_coco_classes_except_person",
            "bag_class_ids": BAG_CLASS_IDS,
            "bag_confidence_floor": args.det_conf,
            "iou_threshold": args.nms_iou,
            "agnostic_nms": True,
        },
        "pose": {
            "weights": str(pose_weights),
            "input_size": [args.imgsz, args.imgsz],
            "person_confidence_threshold": args.pose_conf,
            "max_detections": args.max_detections,
            "iou_threshold": args.nms_iou,
            "keypoint_format": "COCO-17",
            "keypoint_coordinate_space": "original_frame_pixels",
            "keypoint_order": COCO17_KEYPOINT_ORDER,
        },
    }


def make_frame_record(
    *,
    frame_id: int,
    camera_id: str,
    timestamp: float,
    timestamp_source: str,
    frame_size: list[int],
    model_info: dict[str, Any],
    tracker_info: dict[str, Any],
    detections: list[dict[str, Any]],
    people: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    status: str = "processed",
    gap_before: int = 0,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    """Create one schema-compliant processed-frame or skipped-frame record."""
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "status": status,
        "camera_id": camera_id,
        "frame_id": int(frame_id),
        "timestamp": float(timestamp),
        "timestamp_source": timestamp_source,
        "frame_size": frame_size,
        "frame_gap_before": gap_before > 0,
        "skipped_frames_before": int(gap_before),
        "model_info": model_info,
        "tracker_info": tracker_info,
        "models_run": {"detector": status == "processed", "pose": status == "processed"},
        "detections": detections,
        "people": people,
        "poses": poses,
    }
    if skip_reason is not None:
        record["skip_reason"] = skip_reason
    return record


def write_jsonl_record(stream: Any, record: dict[str, Any]) -> None:
    """Write exactly one compact JSON object followed by a newline."""
    stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
    stream.write("\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run detection and pose on matching frames and save per-frame JSONL."""
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if args.capture_start_epoch is None:
        raise ValueError("Provide --capture-start-epoch using the recording's capture start time.")
    if not args.camera_id.strip():
        raise ValueError("Provide a non-empty --camera-id.")

    project_dir = Path(__file__).resolve().parent
    detector_weights = Path(args.detector_weights or project_dir / "models/detection/yolo26s.pt")
    pose_weights = Path(args.pose_weights or project_dir / "models/pose/yolo26s-pose.pt")
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = load_detector(detector_weights)
    is_video = source.suffix.lower() in VIDEO_EXTENSIONS
    legacy_tracker = args.tracker
    object_tracker_config = args.object_tracker or legacy_tracker or "bytetrack.yaml"
    person_tracker_config = args.person_tracker or legacy_tracker or "bytetrack.yaml"
    object_tracker = ByteTrack(object_tracker_config) if is_video else None
    person_tracker = ByteTrack(person_tracker_config) if is_video else None
    pose_estimator = YoloPoseAloneEstimator(
        pose_weights,
        imgsz=args.imgsz,
        device=args.device,
        confidence=args.pose_conf,
        max_detections=args.max_detections,
        nms_iou=args.nms_iou,
        tracker=person_tracker,
    )
    model_info = build_model_info(args, detector_weights, pose_weights)
    timestamp_source = (
        "image_capture_epoch_argument"
        if source.suffix.lower() in IMAGE_EXTENSIONS
        else "capture_start_epoch_plus_frame_index_over_source_fps"
    )
    tracker_info = {
        "objects": object_tracker.metadata() if object_tracker else ByteTrack(object_tracker_config).metadata(),
        "people": person_tracker.metadata() if person_tracker else ByteTrack(person_tracker_config).metadata(),
    }
    tracker_info["objects"]["enabled"] = is_video
    tracker_info["people"]["enabled"] = is_video
    classes = non_person_class_ids(detector)

    output_path = output_dir / f"{source.stem}_predictions.jsonl"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame_count = 0
    skipped_count = 0
    detector_seconds = 0.0
    pose_seconds = 0.0
    started = time.perf_counter()
    completed = False
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            for batch in iter_frame_batches(
                source,
                args.batch_size,
                args.capture_start_epoch,
                start_frame=args.start_frame,
                max_frames=args.max_frames,
                source_fps=args.source_fps,
            ):
                frame_ids, frames, timestamps, gaps, frame_rates = zip(*batch)
                frames_list = list(frames)

                stage_started = time.perf_counter()
                _, detections_by_frame, _ = detect_batch(
                    detector,
                    frames_list,
                    imgsz=args.imgsz,
                    conf=args.det_conf,
                    device=args.device,
                    classes=classes,
                    tracker=object_tracker,
                    agnostic_nms=True,
                    nms_iou=args.nms_iou,
                    max_detections=args.max_detections,
                )
                detector_seconds += time.perf_counter() - stage_started

                stage_started = time.perf_counter()
                pose_outputs = pose_estimator.predict_batch(frames_list)
                pose_seconds += time.perf_counter() - stage_started

                for frame_id, frame, timestamp, gap, fps, detections, pose_output in zip(
                    frame_ids, frames, timestamps, gaps, frame_rates,
                    detections_by_frame, pose_outputs
                ):
                    height, width = frame.shape[:2]
                    frame_size = [int(width), int(height)]
                    for missing_id in range(frame_id - gap, frame_id):
                        skipped_record = make_frame_record(
                            frame_id=missing_id,
                            camera_id=args.camera_id,
                            timestamp=float(args.capture_start_epoch + missing_id / fps),
                            timestamp_source=timestamp_source,
                            frame_size=frame_size,
                            model_info={**model_info, "both_models_every_frame": False},
                            tracker_info=tracker_info,
                            detections=[], people=[], poses=[],
                            status="skipped",
                            skip_reason="decoder_frame_gap",
                        )
                        write_jsonl_record(stream, skipped_record)
                        skipped_count += 1

                    people = pose_output["people"]
                    for pose in pose_output["poses"]:
                        if len(pose["keypoints"]) != len(COCO17_KEYPOINT_ORDER):
                            raise RuntimeError("The pose model must return exactly 17 COCO keypoints per person.")
                    poses = [
                        {
                            "detection_index": int(pose["detection_index"]),
                            "track_id": pose["track_id"],
                            "keypoints": pose["keypoints"].tolist(),
                        }
                        for pose in pose_output["poses"]
                    ]
                    record = make_frame_record(
                        frame_id=frame_id,
                        camera_id=args.camera_id,
                        timestamp=timestamp,
                        timestamp_source=timestamp_source,
                        frame_size=frame_size,
                        model_info=model_info,
                        tracker_info=tracker_info,
                        detections=detections,
                        people=people,
                        poses=poses,
                        gap_before=gap,
                    )
                    write_jsonl_record(stream, record)
                    frame_count += 1
                stream.flush()
        completed = True
    finally:
        if completed:
            temporary_path.replace(output_path)
        elif temporary_path.exists():
            temporary_path.unlink()

    elapsed = time.perf_counter() - started
    inference_seconds = detector_seconds + pose_seconds
    summary = {
        "processed_frames": frame_count,
        "skipped_frames": skipped_count,
        "inference_fps": frame_count / inference_seconds if inference_seconds else 0.0,
        "end_to_end_fps": frame_count / elapsed if elapsed else 0.0,
        "wall_seconds": elapsed,
        "output_jsonl": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for running the perception pipeline."""
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="YOLO26s object detection and YOLO Pose JSONL pipeline")
    parser.add_argument("input", help="Path to an image or video")
    parser.add_argument("--camera-id", required=True, help="Stable ID of the source camera")
    parser.add_argument(
        "--capture-start-epoch", type=float, required=True,
        help="Capture time in epoch seconds for source frame zero (or the image)",
    )
    parser.add_argument("--detector-weights", default=str(project_dir / "models/detection/yolo26s.pt"))
    parser.add_argument("--pose-weights", default=str(project_dir / "models/pose/yolo26s-pose.pt"))
    parser.add_argument("--output", default=str(project_dir / "results"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.1)
    parser.add_argument("--pose-conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--tracker", default=None, help="Tracker YAML applied to both streams")
    parser.add_argument("--object-tracker", default=None, help="Tracker YAML for non-person objects")
    parser.add_argument("--person-tracker", default=None, help="Tracker YAML for people")
    parser.add_argument("--device", default="0", help="Ultralytics device, such as 0 or cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--source-fps", type=float, default=None,
                        help="Override source FPS if video metadata is missing or incorrect")
    return parser


def main() -> None:
    """Parse command-line arguments and run the pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.max_detections < 1:
        parser.error("--max-detections must be at least 1")
    if args.capture_start_epoch < 0:
        parser.error("--capture-start-epoch must be non-negative")
    for name in ("det_conf", "pose_conf", "nms_iou"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not 0.10 <= args.det_conf <= 0.15:
        parser.error("--det-conf must be between 0.10 and 0.15 for the configured object floor")
    if args.pose_conf < 0.25:
        parser.error("--pose-conf must be at least 0.25 for the configured person floor")
    run(args)


if __name__ == "__main__":
    main()
