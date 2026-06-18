from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 60.0
DEFAULT_FONT_SIZE = 128
DEFAULT_TRANSITION_FRAMES = 6
DEFAULT_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\Arialbd.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    Path("/Library/Fonts/Arial Bold.ttf"),
]


def load_events(path: Path, fps: float) -> tuple[list[dict[str, int]], int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    intervals = data.get("timeline", [])
    if not intervals:
        raise ValueError("The counter JSON contains no timeline intervals.")

    events: list[dict[str, int]] = []
    for interval in intervals:
        frame = max(0, round(float(interval["start"]) * fps))
        value = int(interval["total"])
        if events and events[-1]["frame"] == frame:
            events[-1]["value"] = value
        elif not events or events[-1]["value"] != value:
            events.append({"frame": frame, "value": value})

    duration_frames = max(1, round(float(intervals[-1]["end"]) * fps))
    return events, duration_frames


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def make_text_image(
    value: int,
    font: ImageFont.FreeTypeFont,
    canvas_size: tuple[int, int],
) -> Image.Image:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    text = str(value)
    bounds = draw.textbbox((0, 0), text, font=font, stroke_width=4)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = (canvas_size[0] - width) // 2 - bounds[0]
    y = (canvas_size[1] - height) // 2 - bounds[1]
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=4,
        stroke_fill=(0, 0, 0, 220),
    )
    return canvas


def apply_alpha(image: Image.Image, opacity: float) -> Image.Image:
    if opacity >= 0.999:
        return image
    result = image.copy()
    alpha = result.getchannel("A").point(lambda value: round(value * opacity))
    result.putalpha(alpha)
    return result


def composite_at(
    canvas: Image.Image,
    image: Image.Image,
    y_offset: float,
    opacity: float,
) -> None:
    layer = apply_alpha(image, opacity)
    canvas.alpha_composite(layer, (0, round(y_offset)))


def render_frame(
    current_value: int,
    previous_value: int | None,
    progress: float | None,
    text_images: dict[int, Image.Image],
    canvas_size: tuple[int, int],
    scroll_distance: int,
) -> Image.Image:
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    if previous_value is None or progress is None:
        composite_at(canvas, text_images[current_value], 0, 1.0)
        return canvas

    eased = smoothstep(progress)
    increasing = current_value > previous_value
    direction = -1 if increasing else 1
    outgoing_y = direction * scroll_distance * eased
    incoming_y = -direction * scroll_distance * (1.0 - eased)
    composite_at(
        canvas,
        text_images[previous_value],
        outgoing_y,
        1.0 - eased,
    )
    composite_at(canvas, text_images[current_value], incoming_y, eased)
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


def resolve_font(path: Path | None, size: int) -> ImageFont.ImageFont:
    if path is not None:
        if not path.is_file():
            raise FileNotFoundError(f"Font not found: {path}")
        return ImageFont.truetype(str(path), size)

    for candidate in DEFAULT_FONT_CANDIDATES:
        if candidate.is_file():
            print(f"Using font: {candidate}")
            return ImageFont.truetype(str(candidate), size)

    print("No TrueType font found; using Pillow's default bitmap font.")
    return ImageFont.load_default()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a transparent, scrolling ProRes 4444 counter."
    )
    parser.add_argument("counter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument(
        "--font",
        type=Path,
        help="Optional TrueType/OpenType font path. Auto-detected by default.",
    )
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument(
        "--transition-frames",
        type=int,
        default=DEFAULT_TRANSITION_FRAMES,
    )
    parser.add_argument(
        "--scroll-distance",
        type=int,
        default=100,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Overlay dimensions must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")
    if args.transition_frames < 1:
        raise ValueError("--transition-frames must be at least one.")

    events, duration_frames = load_events(args.counter, args.fps)
    output = args.output or args.counter.with_suffix(".overlay.mov")
    output.parent.mkdir(parents=True, exist_ok=True)
    font = resolve_font(args.font, args.font_size)
    values = sorted({event["value"] for event in events})
    canvas_size = (args.width, args.height)
    text_images = {
        value: make_text_image(value, font, canvas_size) for value in values
    }

    transition_lengths: list[int] = []
    for index, event in enumerate(events):
        next_frame = (
            events[index + 1]["frame"]
            if index + 1 < len(events)
            else duration_frames
        )
        transition_lengths.append(
            max(1, min(args.transition_frames, next_frame - event["frame"]))
        )

    encoder = start_encoder(output, args.width, args.height, args.fps)
    assert encoder.stdin is not None
    event_index = 0
    try:
        for frame_number in range(duration_frames):
            while (
                event_index + 1 < len(events)
                and events[event_index + 1]["frame"] <= frame_number
            ):
                event_index += 1

            current_event = events[event_index]
            previous_value = (
                events[event_index - 1]["value"] if event_index > 0 else None
            )
            elapsed = frame_number - current_event["frame"]
            transition_length = transition_lengths[event_index]
            progress = (
                elapsed / transition_length
                if previous_value is not None and elapsed <= transition_length
                else None
            )
            frame = render_frame(
                current_event["value"],
                previous_value,
                progress,
                text_images,
                canvas_size,
                args.scroll_distance,
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
        f"Rendered {duration_frames} frames at {args.fps:g} fps to {output}"
    )
    print(
        f"Import the clip above the source video and position its "
        f"{args.width}x{args.height} canvas at the bottom-right."
    )


if __name__ == "__main__":
    main()
