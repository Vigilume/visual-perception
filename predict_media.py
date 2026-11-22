from pathlib import Path
import argparse

import cv2
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".webm"}


def main():
    parser = argparse.ArgumentParser(
        description="Annotate images/videos with YOLO objects and tracked YOLO Pose."
    )
    parser.add_argument("input", help="Path to an image or video")
    parser.add_argument(
        "--detector-weights",
        default="models/detection/yolo26s.pt",
        help="YOLO object detection weights",
    )
    parser.add_argument(
        "--pose-weights",
        default="models/pose/yolo26s-pose.pt",
        help="YOLO Pose weights",
    )
    parser.add_argument(
        "--tracker",
        default="configs/trackers/bytetrack_buffer_60.yaml",
        help="Tracker YAML used for person tracking in videos",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--det-conf", type=float, default=0.10, help="Object confidence threshold")
    parser.add_argument("--pose-conf", type=float, default=0.25, help="Person confidence threshold")
    parser.add_argument("--output", default="outputs", help="Folder for annotated output")
    parser.add_argument("--device", default="0", help="Inference device, such as 0 or cpu")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    detector_weights = Path(args.detector_weights)
    pose_weights = Path(args.pose_weights)
    if not detector_weights.is_file():
        raise FileNotFoundError(f"YOLO detector weights not found: {detector_weights}")
    if not pose_weights.is_file():
        raise FileNotFoundError(f"YOLO Pose weights not found: {pose_weights}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = YOLO(str(detector_weights))
    pose_model = YOLO(str(pose_weights), task="pose")

    # The project uses the detector for non-person objects and YOLO Pose for people.
    names = detector.names
    entries = names.items() if isinstance(names, dict) else enumerate(names)
    object_classes = [
        int(class_id)
        for class_id, class_name in entries
        if str(class_name).lower() != "person"
    ]

    if suffix in IMAGE_EXTENSIONS:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Could not read image: {source}")

        detection = detector.predict(
            frame, imgsz=args.imgsz, conf=args.det_conf,
            device=args.device, classes=object_classes, verbose=False,
        )[0]
        annotated = detection.plot()

        pose = pose_model.predict(
            frame, imgsz=args.imgsz, conf=args.pose_conf,
            device=args.device, verbose=False,
        )[0]
        annotated = pose.plot(img=annotated)

        output_path = output_dir / f"{source.stem}_annotated{source.suffix}"
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not save annotated image: {output_path}")

    else:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_dir / f"{source.stem}_annotated.mp4"

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {output_path}")

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                detection = detector.predict(
                    frame, imgsz=args.imgsz, conf=args.det_conf,
                    device=args.device, classes=object_classes, verbose=False,
                )[0]
                annotated = detection.plot()

                # Tracking IDs and pose keypoints are added on top of YOLO's object boxes.
                pose = pose_model.track(
                    frame,
                    imgsz=args.imgsz,
                    conf=args.pose_conf,
                    device=args.device,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                )[0]
                annotated = pose.plot(img=annotated)

                writer.write(annotated)
        finally:
            capture.release()
            writer.release()

    print(f"Annotated output saved to: {output_path}")


if __name__ == "__main__":
    main()