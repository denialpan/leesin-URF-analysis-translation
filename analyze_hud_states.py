from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from hud_training_annotator import load_regions, probe_video_info
from train_hud_state_models import extract_features


CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
OUTPUT_STATES = {"ready", "cooldown", "disabled", "recast", "missing"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect stabilized HUD states and transition timestamps."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--regions",
        type=Path,
        default=Path("hud-regions.json"),
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("training-data/hud-state-models.joblib"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=30.0,
        help="Frames analyzed per second. Use source FPS for frame-level timing.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Minimum classifier probability accepted as a state.",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=2,
        help="Consecutive matching samples required to commit a state change.",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Source timestamp in seconds at which analysis begins.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Optional number of seconds to analyze.",
    )
    parser.add_argument(
        "--exclude-region",
        action="append",
        default=[],
        help="Region to omit from analysis. May be specified more than once.",
    )
    return parser.parse_args()


def start_frame_stream(
    video: Path,
    sample_fps: float,
    crop_box: tuple[int, int, int, int],
    start: float,
    duration: float | None,
) -> subprocess.Popen[bytes]:
    x, y, width, height = crop_box
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.9f}",
        "-i",
        str(video),
    ]
    if duration is not None:
        command.extend(["-t", f"{duration:.9f}"])
    command.extend([
        "-vf",
        f"fps={sample_fps:g},format=rgb24,crop={width}:{height}:{x}:{y}",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "-",
    ])
    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )


def classify_region(
    model: object,
    image: Image.Image,
    confidence_threshold: float,
) -> tuple[str, float]:
    probabilities = model.predict_proba([extract_features(image)])[0]
    classes = model.classes_
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    label = str(classes[best_index])
    if confidence < confidence_threshold:
        return "uncertain", confidence
    return label, confidence


def read_exact(stream: object, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if args.stable_frames < 1:
        raise ValueError("--stable-frames must be at least one.")
    if args.start < 0:
        raise ValueError("--start cannot be negative.")
    if args.duration is not None and args.duration <= 0:
        raise ValueError("--duration must be greater than zero.")

    regions = load_regions(args.regions)
    bundle = joblib.load(args.models)
    models: dict[str, object] = bundle["models"]
    excluded_regions = {
        name.strip().casefold() for name in args.exclude_region if name.strip()
    }
    if excluded_regions:
        print(
            "Excluding regions from analysis: "
            + ", ".join(sorted(excluded_regions))
        )
    active_regions = {
        name: region
        for name, region in regions.items()
        if name in models and name.casefold() not in excluded_regions
    }
    unmodeled_regions = [
        name
        for name in regions
        if name not in models and name.casefold() not in excluded_regions
    ]
    if unmodeled_regions:
        print(
            "Skipping regions without trained models: "
            + ", ".join(unmodeled_regions)
        )
    if not active_regions:
        raise RuntimeError("No loaded model matches a region in the region JSON.")

    source_fps, source_frames, duration = probe_video_info(args.video)
    sample_fps = min(args.sample_fps, source_fps)

    left = min(region["x"] for region in active_regions.values())
    top = min(region["y"] for region in active_regions.values())
    right = max(
        region["x"] + region["width"] for region in active_regions.values()
    )
    bottom = max(
        region["y"] + region["height"] for region in active_regions.values()
    )
    crop_box = (left, top, right - left, bottom - top)
    frame_bytes = crop_box[2] * crop_box[3] * 3

    pending: dict[str, deque[str]] = {
        name: deque(maxlen=args.stable_frames) for name in active_regions
    }
    current: dict[str, str | None] = {name: None for name in active_regions}
    segment_start: dict[str, float] = {name: 0.0 for name in active_regions}
    segments: dict[str, list[dict[str, object]]] = {
        name: [] for name in active_regions
    }
    events: list[dict[str, object]] = []

    process = start_frame_stream(
        args.video, sample_fps, crop_box, args.start, args.duration
    )
    assert process.stdout is not None
    analyzed_index = 0
    try:
        while True:
            raw = read_exact(process.stdout, frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("ffmpeg returned an incomplete video frame.")

            timestamp = args.start + analyzed_index / sample_fps
            source_frame = min(
                source_frames - 1, round(timestamp * source_fps)
            )
            array = np.frombuffer(raw, dtype=np.uint8).reshape(
                crop_box[3], crop_box[2], 3
            )
            frame_image = Image.fromarray(array, "RGB")

            for name, region in active_regions.items():
                relative_x = region["x"] - left
                relative_y = region["y"] - top
                image = frame_image.crop(
                    (
                        relative_x,
                        relative_y,
                        relative_x + region["width"],
                        relative_y + region["height"],
                    )
                )
                label, confidence = classify_region(
                    models[name], image, args.confidence
                )
                queue = pending[name]
                queue.append(label)
                if (
                    len(queue) < args.stable_frames
                    or len(set(queue)) != 1
                    or label == "uncertain"
                    or label == current[name]
                ):
                    continue

                transition_time = max(
                    args.start,
                    timestamp - (args.stable_frames - 1) / sample_fps,
                )
                previous = current[name]
                if previous is not None:
                    segments[name].append(
                        {
                            "state": previous,
                            "start": round(segment_start[name], 6),
                            "end": round(transition_time, 6),
                        }
                    )
                current[name] = label
                segment_start[name] = transition_time
                events.append(
                    {
                        "region": name,
                        "state": label,
                        "previous_state": previous,
                        "timestamp": round(transition_time, 6),
                        "frame": min(
                            source_frames - 1,
                            round(transition_time * source_fps),
                        ),
                        "confidence": round(confidence, 6),
                    }
                )

            analyzed_index += 1
            if analyzed_index % max(1, round(sample_fps * 5)) == 0:
                processed = analyzed_index / sample_fps
                analysis_duration = (
                    min(args.duration, max(0.0, duration - args.start))
                    if args.duration is not None
                    else max(0.0, duration - args.start)
                )
                percent = (
                    min(100.0, processed / analysis_duration * 100)
                    if analysis_duration
                    else 0
                )
                print(
                    f"\rProcessed {processed:8.1f}s / {analysis_duration:8.1f}s "
                    f"({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
    finally:
        process.stdout.close()

    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    print()
    if return_code != 0:
        raise RuntimeError(stderr.strip() or "ffmpeg video decoding failed.")

    analyzed_end = min(duration, args.start + analyzed_index / sample_fps)
    for name, state in current.items():
        if state is not None:
            segments[name].append(
                {
                    "state": state,
                    "start": round(segment_start[name], 6),
                    "end": round(analyzed_end, 6),
                }
            )

    output = args.output or args.video.with_suffix(".hud-states.json")
    payload = {
        "video": str(args.video.resolve()),
        "source_fps": source_fps,
        "source_frames": source_frames,
        "duration": duration,
        "analysis_start": args.start,
        "analysis_end": analyzed_end,
        "analysis_fps": sample_fps,
        "confidence_threshold": args.confidence,
        "stable_frames": args.stable_frames,
        "reported_states": sorted(OUTPUT_STATES),
        "events": [
            event for event in events if event["state"] in OUTPUT_STATES
        ],
        "segments": {
            name: [
                segment
                for segment in region_segments
                if segment["state"] in OUTPUT_STATES
            ]
            for name, region_segments in segments.items()
        },
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(payload['events'])} state transitions to {output}")


if __name__ == "__main__":
    main()
