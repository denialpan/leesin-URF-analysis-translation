# process all videos

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
WHISPER_PYTHON = (
    SCRIPT_DIR / ".venv-whisperx" / "Scripts" / "python.exe"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-generate Chinese SRT, counter, and keystroke history "
            "artifacts for old- and new-icon videos."
        )
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("downloads/old"),
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("downloads/new"),
    )
    parser.add_argument(
        "--old",
        action="store_true",
        help="Process videos from --old-dir.",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Process videos from --new-dir.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--recast-timeout",
        type=float,
        default=3.1,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate HUD-derived artifacts. Transcription and contextual "
            "translation still reuse completed outputs unless their own force "
            "flags are passed."
        ),
    )
    parser.add_argument(
        "--skip-hud",
        action="store_true",
        help="Skip HUD analysis, timelines, and overlay rendering.",
    )
    parser.add_argument(
        "--skip-transcription",
        action="store_true",
        help="Skip UVR vocal isolation and Chinese transcription.",
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
        "--force-transcription",
        action="store_true",
        help="Regenerate Chinese transcription instead of reusing it.",
    )
    parser.add_argument(
        "--keep-vocals",
        action="store_true",
        help="Keep each isolated vocal WAV after Chinese transcription.",
    )
    parser.add_argument(
        "--contextual-translation",
        choices=("off", "bundle", "api"),
        default="off",
        help=(
            "Build contextual translation jobs, or submit them to an "
            "OpenAI-compatible endpoint."
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
        default=SCRIPT_DIR / "league-terminology.json",
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
        "--stop-on-error",
        action="store_true",
        help="Stop immediately instead of continuing to the next video.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many videos, useful for testing.",
    )
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def console_text(value: object) -> str:
    text = str(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(
        encoding, errors="backslashreplace"
    ).decode(encoding)


def discover_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        print(console_text(f"Skipping missing directory: {directory}"))
        return []
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name.casefold(),
    )


def run_command(
    label: str,
    command: list[str],
    dry_run: bool,
) -> None:
    print(console_text(f"  {label}"))
    print(console_text(f"  > {subprocess.list2cmdline(command)}"))
    if dry_run:
        return
    started = time.perf_counter()
    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode} "
            f"after {elapsed:.1f}s"
        )
    print(f"  Completed in {elapsed:.1f}s")


def should_run(path: Path, force: bool) -> bool:
    return force or not path.is_file()


def hud_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "states": output_dir / "hud-states.json",
        "counter": output_dir / "counter.json",
        "keystrokes": output_dir / "keystrokes.json",
        "counter_overlay": output_dir / "counter.overlay.mov",
        "keystroke_overlay": output_dir / "keystrokes.history.overlay.mov",
    }


def process_hud(
    video: Path,
    output_dir: Path,
    style: str,
    args: argparse.Namespace,
) -> None:
    paths = hud_paths(output_dir)
    force_hud_artifacts = getattr(args, "force_hud_artifacts", args.force)
    if style == "old":
        models = SCRIPT_DIR / "training-data" / "hud-state-models.joblib"
        excluded_region = "rocketbelt"
        icon_set = "old"
    else:
        models = (
            SCRIPT_DIR
            / "training-data-new-icons"
            / "hud-state-models.joblib"
        )
        excluded_region = "prowler"
        icon_set = "new"

    required = [
        models,
        SCRIPT_DIR / "hud-regions.json",
        SCRIPT_DIR / "rules.md",
        SCRIPT_DIR / "keystrokerules.md",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing HUD dependency: " + ", ".join(str(path) for path in missing)
        )

    python = sys.executable
    states_changed = should_run(paths["states"], force_hud_artifacts)
    if states_changed:
        run_command(
            "Analyze HUD states",
            [
                python,
                str(SCRIPT_DIR / "analyze_hud_states.py"),
                str(video),
                "--regions",
                str(SCRIPT_DIR / "hud-regions.json"),
                "--models",
                str(models),
                "--sample-fps",
                f"{args.sample_fps:g}",
                "--confidence",
                f"{args.confidence:g}",
                "--stable-frames",
                str(args.stable_frames),
                "--exclude-region",
                excluded_region,
                "--output",
                str(paths["states"]),
            ],
            args.dry_run,
        )
    else:
        print("  Reusing hud-states.json")

    counter_changed = (
        states_changed or should_run(paths["counter"], force_hud_artifacts)
    )
    if counter_changed:
        run_command(
            "Generate counter timeline",
            [
                python,
                str(SCRIPT_DIR / "generate_counter_timeline.py"),
                str(paths["states"]),
                "--rules",
                str(SCRIPT_DIR / "rules.md"),
                "--output",
                str(paths["counter"]),
            ],
            args.dry_run,
        )
    else:
        print("  Reusing counter.json")

    keystrokes_changed = (
        states_changed or should_run(paths["keystrokes"], force_hud_artifacts)
    )
    if keystrokes_changed:
        run_command(
            "Generate keystroke history",
            [
                python,
                str(SCRIPT_DIR / "generate_keystroke_history.py"),
                str(paths["states"]),
                "--rules",
                str(SCRIPT_DIR / "keystrokerules.md"),
                "--recast-timeout",
                f"{args.recast_timeout:g}",
                "--output",
                str(paths["keystrokes"]),
            ],
            args.dry_run,
        )
    else:
        print("  Reusing keystrokes.json")

    if counter_changed or should_run(
        paths["counter_overlay"], force_hud_artifacts
    ):
        run_command(
            "Render counter overlay",
            [
                python,
                str(SCRIPT_DIR / "render_counter_overlay.py"),
                str(paths["counter"]),
                "--fps",
                f"{args.sample_fps:g}",
                "--output",
                str(paths["counter_overlay"]),
            ],
            args.dry_run,
        )
    else:
        print("  Reusing counter.overlay.mov")

    if keystrokes_changed or should_run(
        paths["keystroke_overlay"], force_hud_artifacts
    ):
        run_command(
            "Render keystroke overlay",
            [
                python,
                str(SCRIPT_DIR / "render_keystroke_history_overlay.py"),
                str(paths["keystrokes"]),
                "--fps",
                f"{args.sample_fps:g}",
                "--icon-set",
                icon_set,
                "--output",
                str(paths["keystroke_overlay"]),
            ],
            args.dry_run,
        )
    else:
        print("  Reusing keystrokes.history.overlay.mov")


def transcription_paths(video: Path, output_dir: Path) -> dict[str, Path]:
    base = video.stem
    return {
        "srt": output_dir / f"{base}.uvr.whisperx.chinese.short.srt",
        "debug": output_dir / f"{base}.uvr.whisperx.chinese.short.json",
        "vocals": output_dir / f"{base}.uvr.vocals.wav",
    }


def process_transcription(
    video: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    paths = transcription_paths(video, output_dir)
    required = [paths["srt"], paths["debug"]]
    if args.keep_vocals:
        required.append(paths["vocals"])
    complete = all(path.is_file() for path in required)
    force_transcription = getattr(args, "force_transcription", args.force)
    if complete and not force_transcription:
        print("  Reusing Chinese SRT and isolated vocals")
        return
    if not WHISPER_PYTHON.is_file():
        raise FileNotFoundError(
            f"WhisperX Python environment not found: {WHISPER_PYTHON}"
        )
    command = [
        str(WHISPER_PYTHON),
        str(SCRIPT_DIR / "generate_chinese_vocal_srt.py"),
        str(video),
        "--output-dir",
        str(output_dir),
    ]
    if getattr(args, "transcription_quality", "standard") == "high":
        command.extend(
            [
                "--refine-min-word-probability",
                "0.60",
                "--refine-padding-seconds",
                "2.0",
                "--refine-beam-size",
                "10",
            ]
        )
    if not args.keep_vocals:
        command.append("--discard-vocals")
    run_command(
        "Isolate vocals and transcribe Chinese",
        command,
        args.dry_run,
    )


def process_contextual_translation(
    video: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "generate_contextual_translation.py"),
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
    if args.contextual_translation == "api":
        vision_model = args.context_model or args.context_vision_model
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
    run_command(
        "Build contextual gameplay translation",
        command,
        args.dry_run,
    )


def validate_outputs(
    video: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    expected: list[Path] = []
    if not args.skip_hud:
        expected.extend(hud_paths(output_dir).values())
    if not args.skip_transcription:
        transcription = transcription_paths(video, output_dir)
        expected.extend([transcription["srt"], transcription["debug"]])
        if args.keep_vocals:
            expected.append(transcription["vocals"])
    return [str(path) for path in expected if not path.is_file()]


def main() -> None:
    args = parse_args()
    args.force_hud_artifacts = True
    if not args.old and not args.new:
        raise ValueError(
            "No video style selected. Pass --old, --new, or both."
        )
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if args.stable_frames < 1:
        raise ValueError("--stable-frames must be at least one.")
    if args.recast_timeout <= 0:
        raise ValueError("--recast-timeout must be greater than zero.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least one.")
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

    jobs: list[tuple[str, Path]] = []
    if args.old:
        jobs.extend(
            ("old", video)
            for video in discover_videos(absolute(args.old_dir))
        )
    if args.new:
        jobs.extend(
            ("new", video)
            for video in discover_videos(absolute(args.new_dir))
        )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        raise RuntimeError("No videos were found.")

    print(f"Found {len(jobs)} video(s).")
    print("Existing complete artifacts will be reused.")
    results: list[dict[str, object]] = []
    batch_started = time.perf_counter()

    for index, (style, video) in enumerate(jobs, start=1):
        output_dir = video.parent / video.stem
        print(
            console_text(
                f"\n[{index}/{len(jobs)}] {style.upper()} ICONS: {video.name}"
            )
        )
        print(console_text(f"Output folder: {output_dir}"))
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        status = "completed"
        error_text: str | None = None
        try:
            if not args.skip_hud:
                process_hud(video, output_dir, style, args)
            if not args.skip_transcription:
                process_transcription(video, output_dir, args)
            if args.contextual_translation != "off":
                process_contextual_translation(video, output_dir, args)
            missing = (
                []
                if args.dry_run
                else validate_outputs(video, output_dir, args)
            )
            if missing:
                raise RuntimeError(
                    "Expected outputs were not created: " + ", ".join(missing)
                )
        except Exception as error:
            status = "failed"
            error_text = f"{type(error).__name__}: {error}"
            print(console_text(f"FAILED: {error_text}"))
            if args.stop_on_error:
                raise
        elapsed = time.perf_counter() - started
        results.append(
            {
                "style": style,
                "video": str(video),
                "output_dir": str(output_dir),
                "status": status,
                "elapsed_seconds": round(elapsed, 3),
                "error": error_text,
            }
        )

    completed = sum(result["status"] == "completed" for result in results)
    failed = len(results) - completed
    summary = {
        "completed": completed,
        "failed": failed,
        "elapsed_seconds": round(time.perf_counter() - batch_started, 3),
        "jobs": results,
    }
    summary_path = SCRIPT_DIR / "batch-pipeline-summary.json"
    if not args.dry_run:
        summary_path.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        f"\nBatch complete: {completed} completed, {failed} failed, "
        f"{summary['elapsed_seconds']:.1f}s elapsed."
    )
    if not args.dry_run:
        print(console_text(f"Summary: {summary_path}"))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBatch interrupted. Existing completed outputs can be reused.")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
