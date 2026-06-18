# process only one video

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from run_video_batch_pipeline import (
    VIDEO_EXTENSIONS,
    console_text,
    default_uvr_executable,
    process_hud,
    process_transcription,
    resolve_whisper_python,
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
        help=(
            "Reuse the existing Chinese transcript. With contextual "
            "translation enabled, only run the Qwen/context stage."
        ),
    )
    parser.add_argument(
        "--keep-vocals",
        action="store_true",
        help="Keep the isolated vocal WAV after transcription.",
    )
    parser.add_argument(
        "--whisper-python",
        type=Path,
        default=None,
        help=(
            "Python executable for the faster-whisper environment. Defaults "
            "to .venv-whisperx/Scripts/python.exe on Windows, "
            ".venv-whisperx/bin/python elsewhere, or the active Python if it "
            "can import faster_whisper."
        ),
    )
    parser.add_argument(
        "--uvr-executable",
        type=Path,
        default=default_uvr_executable(),
        help=(
            "audio-separator executable. Defaults to "
            ".venv-audio-separator/Scripts/audio-separator.exe on Windows "
            "and .venv-audio-separator/bin/audio-separator elsewhere."
        ),
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
        "--contextual-translation",
        choices=("off", "bundle", "api"),
        default="off",
        help=(
            "Build gameplay-aware translation jobs, or submit them to an "
            "OpenAI-compatible vision endpoint."
        ),
    )
    parser.add_argument("--context-api-url")
    parser.add_argument(
        "--context-model",
        help="Legacy alias for --context-vision-model.",
    )
    parser.add_argument(
        "--context-text-model",
        default="qwen2.5:14b",
    )
    parser.add_argument(
        "--context-vision-model",
        default="qwen2.5vl:7b",
    )
    parser.add_argument(
        "--context-glossary",
        type=Path,
        default=Path(__file__).resolve().parent / "league-terminology.json",
    )
    parser.add_argument(
        "--context-visual-verify-threshold",
        type=float,
        default=0.78,
    )
    parser.add_argument(
        "--context-api-key-env",
        default="OPENAI_API_KEY",
    )
    parser.add_argument("--context-api-context-size", type=int, default=8192)
    parser.add_argument("--context-request-timeout", type=float, default=600.0)
    parser.add_argument("--context-retries", type=int, default=2)
    parser.add_argument(
        "--context-force-results",
        action="store_true",
        help="Discard and regenerate completed contextual API results.",
    )
    parser.add_argument("--context-start-cue", type=int, default=1)
    parser.add_argument("--context-end-cue", type=int)
    parser.add_argument(
        "--context-start-frame",
        type=int,
        help="Inclusive source-video frame where contextual translation begins.",
    )
    parser.add_argument(
        "--context-end-frame",
        type=int,
        help="Inclusive source-video frame where contextual translation ends.",
    )
    parser.add_argument(
        "--context-no-audio-recovery",
        action="store_true",
        help=(
            "Do not retry empty frame ranges using multiple audio-volume "
            "profiles and focused Chinese ASR."
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
    if args.context_request_timeout <= 0:
        raise ValueError("--context-request-timeout must be greater than zero.")
    if args.context_retries < 0:
        raise ValueError("--context-retries cannot be negative.")
    if not 0 <= args.context_visual_verify_threshold <= 1:
        raise ValueError(
            "--context-visual-verify-threshold must be between zero and one."
        )
    if (
        args.contextual_translation != "off"
        and not args.context_glossary.resolve().is_file()
    ):
        raise FileNotFoundError(
            f"League glossary not found: {args.context_glossary.resolve()}"
        )
    if (args.context_start_frame is None) != (
        args.context_end_frame is None
    ):
        raise ValueError(
            "--context-start-frame and --context-end-frame must be "
            "provided together."
        )
    if args.context_start_frame is not None:
        if args.context_start_frame < 0:
            raise ValueError("--context-start-frame cannot be negative.")
        if args.context_end_frame < args.context_start_frame:
            raise ValueError(
                "--context-end-frame cannot precede --context-start-frame."
            )
    if (
        args.skip_hud
        and args.skip_transcription
        and args.contextual_translation == "off"
    ):
        raise ValueError(
            "--skip-hud and --skip-transcription require "
            "--contextual-translation bundle or api."
        )
    if args.contextual_translation == "api" and not args.context_api_url:
        raise ValueError(
            "API contextual translation requires --context-api-url."
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
    if args.contextual_translation != "off":
        command = [
            sys.executable,
            str(Path(__file__).resolve().parent / "generate_contextual_translation.py"),
            str(video),
            "--output-dir",
            str(output_dir / "contextual-translation"),
            "--provider",
            args.contextual_translation,
            "--start-cue",
            str(args.context_start_cue),
        ]
        if args.context_end_cue is not None:
            command.extend(["--end-cue", str(args.context_end_cue)])
        if args.context_start_frame is not None:
            command.extend(
                [
                    "--start-frame",
                    str(args.context_start_frame),
                    "--end-frame",
                    str(args.context_end_frame),
                ]
            )
        if args.context_no_audio_recovery:
            command.append("--no-audio-recovery")
        whisper_python = resolve_whisper_python(
            args.whisper_python,
            require_import=not args.dry_run,
        )
        command.extend(["--whisper-python", str(whisper_python)])
        if args.contextual_translation == "api":
            vision_model = (
                args.context_model or args.context_vision_model
            )
            command.extend(
                [
                    "--api-url",
                    args.context_api_url,
                    "--text-model",
                    args.context_text_model,
                    "--vision-model",
                    vision_model,
                    "--glossary",
                    str(args.context_glossary),
                    "--visual-verify-threshold",
                    str(args.context_visual_verify_threshold),
                    "--api-key-env",
                    args.context_api_key_env,
                    "--api-context-size",
                    str(args.context_api_context_size),
                    "--request-timeout",
                    str(args.context_request_timeout),
                    "--retries",
                    str(args.context_retries),
                ]
            )
        if args.force:
            command.append("--force-frames")
        if args.context_force_results:
            command.append("--force-results")
        print("\nBuild contextual gameplay translation")
        print(f"  > {subprocess.list2cmdline(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)

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
