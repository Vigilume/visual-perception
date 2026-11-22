"""Validate JSONL records against the event-team frame schema."""

import argparse
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


def load_validator(schema_path: Path) -> Draft202012Validator:
    """Load the frame schema and prepare a JSON Schema validator."""
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_record_links(record: dict[str, Any]) -> list[str]:
    """Check that each pose links to the matching person detection index."""
    issues = []
    people = record.get("people", [])
    indices = [person.get("index") for person in people]
    if len(indices) != len(set(indices)):
        issues.append("people indices are not unique")
    people_by_index = {person.get("index"): person for person in people}
    for pose in record.get("poses", []):
        detection_index = pose.get("detection_index")
        person = people_by_index.get(detection_index)
        if person is None:
            issues.append(f"pose detection_index {detection_index!r} has no people entry")
        elif pose.get("track_id") != person.get("track_id"):
            issues.append(f"pose/person track_id mismatch at people index {detection_index}")
    return issues


def validate_jsonl(jsonl_path: Path, schema_path: Path) -> tuple[int, list[str]]:
    """Validate every JSONL line, frame order, timestamps, and pose links."""
    validator = load_validator(schema_path)
    errors: list[str] = []
    records = 0
    previous_frame_id: int | None = None
    previous_timestamp: float | None = None
    camera_id: str | None = None

    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                errors.append(f"line {line_number}: blank lines are not valid JSONL records")
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            records += 1
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: each line must contain a JSON object")
                continue

            for issue in validator.iter_errors(record):
                path = ".".join(str(part) for part in issue.absolute_path) or "record"
                errors.append(f"line {line_number}: {path}: {issue.message}")
            for issue in validate_record_links(record):
                errors.append(f"line {line_number}: {issue}")

            gap_flag = record.get("frame_gap_before")
            skipped_before = record.get("skipped_frames_before")
            if isinstance(gap_flag, bool) and isinstance(skipped_before, int):
                if gap_flag != (skipped_before > 0):
                    errors.append(f"line {line_number}: frame_gap_before does not match skipped_frames_before")

            frame_id = record.get("frame_id")
            timestamp = record.get("timestamp")
            current_camera = record.get("camera_id")
            if previous_frame_id is not None and isinstance(frame_id, int):
                if frame_id <= previous_frame_id:
                    errors.append(f"line {line_number}: frame_id is not increasing")
                elif frame_id > previous_frame_id + 1:
                    missing = frame_id - previous_frame_id - 1
                    if not record.get("frame_gap_before") or record.get("skipped_frames_before") != missing:
                        errors.append(f"line {line_number}: frame gap is not reported (expected {missing})")
            if previous_timestamp is not None and isinstance(timestamp, (int, float)):
                if timestamp < previous_timestamp:
                    errors.append(f"line {line_number}: timestamp moved backwards")
            if camera_id is not None and current_camera is not None and current_camera != camera_id:
                errors.append(f"line {line_number}: camera_id changed inside one output stream")
            if isinstance(frame_id, int):
                previous_frame_id = frame_id
            if isinstance(timestamp, (int, float)):
                previous_timestamp = float(timestamp)
            if isinstance(current_camera, str):
                camera_id = current_camera

    if records == 0:
        errors.append("file contains no frame records")
    return records, errors


def main() -> None:
    """Parse file paths, validate the JSONL output, and return a status code."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate per-frame visual perception JSONL")
    parser.add_argument("jsonl", type=Path, help="Path to the pipeline JSONL output")
    parser.add_argument("--schema", type=Path, default=root / "schemas/frame_record.schema.json")
    args = parser.parse_args()
    count, errors = validate_jsonl(args.jsonl, args.schema)
    if errors:
        print(f"Invalid: {len(errors)} issue(s) across {count} records.")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Valid: {count} JSONL frame records.")


if __name__ == "__main__":
    main()
