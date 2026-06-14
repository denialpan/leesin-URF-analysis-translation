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
        help="Regenerate outputs even when the destination file exists.",
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
        "--keep-vocals",
        action="store_true",
        help="Keep each isolated vocal WAV after Chinese transcription.",
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
    states_changed = should_run(paths["states"], args.force)
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
        states_changed or should_run(paths["counter"], args.force)
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
        states_changed or should_run(paths["keystrokes"], args.force)
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

    if counter_changed or should_run(paths["counter_overlay"], args.force):
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
        paths["keystroke_overlay"], args.force
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
    if complete and not args.force:
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
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if args.stable_frames < 1:
        raise ValueError("--stable-frames must be at least one.")
    if args.recast_timeout <= 0:
        raise ValueError("--recast-timeout must be greater than zero.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least one.")

    jobs = [
        ("old", video)
        for video in discover_videos(absolute(args.old_dir))
    ]
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
