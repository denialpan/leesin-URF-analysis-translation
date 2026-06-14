# process only one video

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from run_video_batch_pipeline import (
    VIDEO_EXTENSIONS,
    console_text,
    process_hud,
    process_transcription,
    validate_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HUD states, counter, keystrokes, overlays, and a "
            "Chinese SRT for one video."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--style",
        choices=("auto", "old", "new"),
        default="auto",
        help=(
            "HUD icon style. Auto detects a parent folder named old or new."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a folder beside the video matching its filename.",
    )
    parser.add_argument("--sample-fps", type=float, default=60.0)
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--stable-frames", type=int, default=2)
    parser.add_argument("--recast-timeout", type=float, default=3.1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate outputs even when they already exist.",
    )
    parser.add_argument(
        "--skip-hud",
        action="store_true",
        help="Only run vocal isolation and Chinese transcription.",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Only run HUD analysis and overlay generation.",
    )
    parser.add_argument(
        "--keep-vocals",
        action="store_true",
        help="Keep the isolated vocal WAV after transcription.",
    )
    parser.add_argument(
        "--transcription-quality",
        choices=("standard", "high"),
        default="high",
        help=(
            "High uses more aggressive confidence checks and wider raw-audio "
            "context for questionable Chinese cues."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def detect_style(video: Path) -> str:
    for parent in video.parents:
        name = parent.name.casefold()
        if name == "old":
            return "old"
        if name == "new":
            return "new"
    raise ValueError(
        "Could not infer the icon style from the video path. "
        "Pass --style old or --style new."
    )


def main() -> None:
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if video.suffix.casefold() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video extension: {video.suffix}")
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if args.stable_frames < 1:
        raise ValueError("--stable-frames must be at least one.")
    if args.recast_timeout <= 0:
        raise ValueError("--recast-timeout must be greater than zero.")
    if args.skip_hud and args.skip_transcription:
        raise ValueError(
            "--skip-hud and --skip-transcription cannot both be used."
        )

    style = detect_style(video) if args.style == "auto" else args.style
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else video.parent / video.stem
    )

    print(console_text(f"Video: {video}"))
    print(f"Icon style: {style}")
    print(console_text(f"Output folder: {output_dir}"))
    print("Existing completed artifacts will be reused.")
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    if not args.skip_hud:
        process_hud(video, output_dir, style, args)
    if not args.skip_transcription:
        process_transcription(video, output_dir, args)

    missing = (
        [] if args.dry_run else validate_outputs(video, output_dir, args)
    )
    if missing:
        raise RuntimeError(
            "Expected outputs were not created: " + ", ".join(missing)
        )

    elapsed = time.perf_counter() - started
    print(f"\nSingle-video pipeline completed in {elapsed:.1f}s.")
    if not args.dry_run:
        print(console_text(f"Outputs: {output_dir}"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Run the same command again to resume.")
        raise SystemExit(130)
    except Exception as error:
        print(console_text(f"ERROR: {type(error).__name__}: {error}"))
        raise SystemExit(1)
