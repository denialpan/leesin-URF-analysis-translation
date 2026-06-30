from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


DEFAULT_WIDTH = 390
DEFAULT_HEIGHT = 96
DEFAULT_FPS = 60.0
DEFAULT_HISTORY_LENGTH = 6
DEFAULT_TRANSITION_FRAMES = 8
DEFAULT_HORIZONTAL_GAP = 6
DEFAULT_ICON_SIZE = 56
DEFAULT_HOLD_SECONDS = 3.0
DEFAULT_FADE_FRAMES = 12
DEFAULT_ICONS_DIR = Path(__file__).resolve().parent / "rendericons"

COMMON_ICON_FILENAMES = {
    "prowler": "prowler.png",
    "rocketbelt": "rocketbelt.png",
    "flash": "flash.png",
    "cleanse": "cleanse.png",
}
ICON_SETS = {
    "old": {
        **COMMON_ICON_FILENAMES,
        "E": "E_old.png",
        "Q1": "Q_old.png",
        "Q2": "Q_old_recast.png",
        "R": "R_old.png",
        "W": "W_old.png",
        "ward": ["ward_old.png", "ward.png"],
        "cward": ["cward_old.png", "cward.png"],
    },
    "new": {
        **COMMON_ICON_FILENAMES,
        "E": "E.png",
        "Q1": "Q.png",
        "Q2": "Q_recast.png",
        "R": "R.png",
        "W": "W.png",
        "ward": "ward.png",
        "cward": "cward.png",
    },
}


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def load_history_events(
    path: Path,
    fps: float,
    duration: float | None,
) -> tuple[list[dict[str, object]], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    source_events = data.get("events", [])
    if not source_events:
        raise ValueError("The keystroke JSON contains no events.")

    events: list[dict[str, object]] = []
    for source_event in source_events:
        frame = max(0, round(float(source_event["timestamp"]) * fps))
        key = str(source_event["key"])
        events.append(
            {
                "frame": frame,
                "key": key,
            }
        )

    if duration is None:
        source_path = Path(str(data.get("source", "")))
        if source_path.is_file():
            source_data = json.loads(source_path.read_text(encoding="utf-8"))
            duration = float(
                source_data.get("analysis_end", source_data.get("duration", 0))
            )
    if duration is None or duration <= 0:
        duration = float(source_events[-1]["timestamp"]) + 3.0

    return events, max(1, round(duration * fps))


def load_icons(
    icon_directory: Path,
    keys: set[str],
    maximum_size: int,
    icon_filenames: dict[str, str | list[str]],
) -> dict[str, Image.Image]:
    unknown_keys = sorted(keys - icon_filenames.keys())
    if unknown_keys:
        raise ValueError(
            "No render icon mapping exists for: " + ", ".join(unknown_keys)
        )

    icons: dict[str, Image.Image] = {}
    for key in sorted(keys):
        filenames = icon_filenames[key]
        candidates = filenames if isinstance(filenames, list) else [filenames]
        path = next(
            (
                icon_directory / candidate
                for candidate in candidates
                if (icon_directory / candidate).is_file()
            ),
            icon_directory / candidates[0],
        )
        if not path.is_file():
            raise FileNotFoundError(f"Render icon not found for {key}: {path}")
        with Image.open(path) as source:
            icon = source.convert("RGBA")
        scale = min(maximum_size / icon.width, maximum_size / icon.height)
        icon = icon.resize(
            (
                max(1, round(icon.width * scale)),
                max(1, round(icon.height * scale)),
            ),
            Image.Resampling.LANCZOS,
        )
        icons[key] = icon
    return icons


def render_frame(
    events: list[dict[str, object]],
    frame_number: int,
    canvas_size: tuple[int, int],
    icons: dict[str, Image.Image],
    history_length: int,
    horizontal_gap: int,
    transition_frames: int,
    hold_frames: int,
    fade_frames: int,
) -> Image.Image:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    fade_end = hold_frames + fade_frames
    active = [
        event
        for event in events
        if 0 <= frame_number - int(event["frame"]) < fade_end
    ]
    if not active:
        return canvas

    newest_frame = int(active[-1]["frame"])
    added_count = sum(int(event["frame"]) == newest_frame for event in active)
    elapsed = frame_number - newest_frame
    push_progress = smoothstep(min(1.0, elapsed / transition_frames))
    slot_width = (
        canvas_size[0] - horizontal_gap * (history_length - 1)
    ) / history_length
    slot_step = slot_width + horizontal_gap
    right_center = canvas_size[0] - slot_width / 2
    center_y = canvas_size[1] / 2

    # Keep pushed-out icons around only while they are actively exiting left.
    exiting_count = added_count if elapsed < transition_frames else 0
    visible = active[-(history_length + exiting_count):]
    for reverse_index, event in enumerate(reversed(visible)):
        target_x = right_center - reverse_index * slot_step
        if newest_frame == int(event["frame"]):
            x = target_x + added_count * slot_step * (1.0 - push_progress)
            push_opacity = push_progress
        else:
            x = target_x + added_count * slot_step * (1.0 - push_progress)
            push_opacity = 1.0

        age = frame_number - int(event["frame"])
        fade_opacity = (
            1.0
            if age < hold_frames
            else max(0.0, 1.0 - (age - hold_frames) / fade_frames)
        )
        opacity = min(push_opacity, fade_opacity)
        if opacity <= 0:
            continue

        icon = icons[str(event["key"])]
        if opacity < 0.999:
            icon = icon.copy()
            icon.putalpha(
                icon.getchannel("A").point(
                    lambda value: round(value * opacity)
                )
            )
        canvas.alpha_composite(
            icon,
            (
                round(x - icon.width / 2),
                round(center_y - icon.height / 2),
            ),
        )
    return canvas


def start_encoder(
    output: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{fps:g}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        "-vendor",
        "apl0",
        str(output),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=0x08000000 if sys.platform == "win32" else 0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a transparent six-keystroke history overlay."
    )
    parser.add_argument("keystrokes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--icons-dir", type=Path, default=DEFAULT_ICONS_DIR)
    parser.add_argument(
        "--icon-set",
        choices=sorted(ICON_SETS),
        default="old",
        help="Use old or new ability icon artwork.",
    )
    parser.add_argument("--icon-size", type=int, default=DEFAULT_ICON_SIZE)
    parser.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
    )
    parser.add_argument(
        "--transition-frames",
        type=int,
        default=DEFAULT_TRANSITION_FRAMES,
    )
    parser.add_argument(
        "--horizontal-gap",
        type=int,
        default=DEFAULT_HORIZONTAL_GAP,
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
    )
    parser.add_argument(
        "--fade-frames",
        type=int,
        default=DEFAULT_FADE_FRAMES,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Overlay dimensions must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    if args.history_length < 1:
        raise ValueError("--history-length must be at least one.")
    if args.transition_frames < 1:
        raise ValueError("--transition-frames must be at least one.")
    if args.horizontal_gap < 0:
        raise ValueError("--horizontal-gap cannot be negative.")
    if args.hold_seconds < 0:
        raise ValueError("--hold-seconds cannot be negative.")
    if args.fade_frames < 1:
        raise ValueError("--fade-frames must be at least one.")
    if args.icon_size < 1:
        raise ValueError("--icon-size must be at least one.")
    if not args.icons_dir.is_dir():
        raise FileNotFoundError(f"Render icon directory not found: {args.icons_dir}")

    events, duration_frames = load_history_events(
        args.keystrokes,
        args.fps,
        args.duration,
    )
    output = args.output or args.keystrokes.with_suffix(".history.overlay.mov")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas_size = (args.width, args.height)
    slot_width = (
        args.width - args.horizontal_gap * (args.history_length - 1)
    ) / args.history_length
    if slot_width <= 0:
        raise ValueError("The overlay is too narrow for the requested gaps.")
    all_keys = {str(event["key"]) for event in events}
    maximum_icon_size = min(
        args.icon_size,
        args.height,
        max(1, int(slot_width)),
    )
    icons = load_icons(
        args.icons_dir,
        all_keys,
        maximum_icon_size,
        ICON_SETS[args.icon_set],
    )
    hold_frames = round(args.hold_seconds * args.fps)

    encoder = start_encoder(output, args.width, args.height, args.fps)
    assert encoder.stdin is not None
    started_events: list[dict[str, object]] = []
    next_event_index = 0
    try:
        for frame_number in range(duration_frames):
            while (
                next_event_index < len(events)
                and int(events[next_event_index]["frame"]) <= frame_number
            ):
                started_events.append(events[next_event_index])
                next_event_index += 1

            fade_end = hold_frames + args.fade_frames
            started_events = [
                event
                for event in started_events
                if frame_number - int(event["frame"]) < fade_end
            ]
            frame = render_frame(
                started_events,
                frame_number,
                canvas_size,
                icons,
                args.history_length,
                args.horizontal_gap,
                args.transition_frames,
                hold_frames,
                args.fade_frames,
            )

            encoder.stdin.write(frame.tobytes())
            if frame_number and frame_number % max(1, round(args.fps * 5)) == 0:
                percent = frame_number / duration_frames * 100
                print(
                    f"\rRendered {frame_number / args.fps:8.1f}s "
                    f"({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
    except BrokenPipeError:
        pass
    finally:
        encoder.stdin.close()

    stderr = encoder.stderr.read().decode(errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    print()
    if return_code != 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(stderr.strip() or "ffmpeg failed to encode the overlay.")

    print(
        f"Rendered {len(events)} history updates over {duration_frames} frames "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
