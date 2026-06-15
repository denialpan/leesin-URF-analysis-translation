"""
Build gameplay-aware translation jobs from a Chinese transcript.

Bundle mode extracts frames and writes a reviewable JSON file without making
network requests:

    python generate_contextual_translation.py video.mp4 --provider bundle

API mode sends each cue to an OpenAI-compatible vision chat endpoint:

    python generate_contextual_translation.py video.mp4 --provider api \
        --model vision-model --api-url https://host/v1/chat/completions
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
WHISPER_PYTHON = SCRIPT_DIR / ".venv-whisperx" / "Scripts" / "python.exe"
SYSTEM_PROMPT = """You correct and translate Chinese League of Legends gameplay commentary.

Use all supplied evidence:
- raw ASR Chinese and word confidence
- nearby Chinese cues
- gameplay frames
- HUD state changes, keystrokes, and counter state

Rules:
1. Reconstruct the most likely spoken Chinese before translating.
2. Use visual gameplay only to resolve ambiguity; do not invent unspoken details.
3. Preserve League terminology, ability keys, combos, champion names, and casual streamer tone.
4. Translate meaning naturally, not character by character.
5. Use lowercase informal English with no unnecessary punctuation.
6. If evidence is insufficient, retain uncertainty and set needs_review=true.
7. Return only one JSON object with these keys:
   corrected_chinese, english, confidence, needs_review, reasoning.
confidence must be a number from 0 to 1. reasoning must be brief.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract gameplay context around Chinese cues and optionally "
            "produce corrected Chinese and English SRT files."
        )
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--transcript",
        type=Path,
        help="Chinese transcript JSON. Auto-detected beside generated outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a contextual-translation folder in the video output.",
    )
    parser.add_argument(
        "--provider",
        choices=("bundle", "api"),
        default="bundle",
        help="Bundle is offline. API calls an OpenAI-compatible vision endpoint.",
    )
    parser.add_argument("--api-url")
    parser.add_argument("--model")
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API bearer token.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait for one model attempt.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--api-context-size",
        type=int,
        default=8192,
        help=(
            "Requested model context size. Ollama reads this through the "
            "options.num_ctx field."
        ),
    )
    parser.add_argument(
        "--frame-context-seconds",
        type=float,
        default=1.0,
        help="Seconds before and after a cue represented by context frames.",
    )
    parser.add_argument("--frame-width", type=int, default=960)
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=0.75,
        help="Model confidence below this value enters the review queue.",
    )
    parser.add_argument("--start-cue", type=int, default=1)
    parser.add_argument("--end-cue", type=int)
    parser.add_argument(
        "--start-frame",
        type=int,
        help="Inclusive source-video frame where contextual analysis begins.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Inclusive source-video frame where contextual analysis ends.",
    )
    parser.add_argument(
        "--force-frames",
        action="store_true",
        help="Re-extract context frames that already exist.",
    )
    parser.add_argument(
        "--force-results",
        action="store_true",
        help="Regenerate API results that already exist.",
    )
    parser.add_argument(
        "--no-audio-recovery",
        action="store_true",
        help=(
            "Do not retry an empty frame range using multiple audio-volume "
            "profiles and focused Chinese ASR."
        ),
    )
    return parser.parse_args()


def safe_name(path: Path) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", path.stem).strip(" .")
    return value or "video"


def output_root_for_video(video: Path) -> Path:
    candidate = video.parent / video.stem
    return candidate if candidate.is_dir() else video.parent


def find_transcript(video: Path, explicit: Path | None) -> Path:
    if explicit:
        path = explicit.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    root = output_root_for_video(video)
    candidates = sorted(
        root.glob("*.uvr.whisperx.chinese.short.json"),
        key=lambda path: (
            not path.name.startswith(video.stem),
            -path.stat().st_mtime,
        ),
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Chinese transcript JSON found in {root}"
        )
    return candidates[0]


def transcript_has_range_cues(
    transcript: dict[str, object],
    range_start: float,
    range_end: float,
) -> bool:
    return any(
        overlaps(
            range_start,
            range_end,
            float(cue["start"]),
            float(cue["end"]),
        )
        for cue in transcript.get("cues", [])
    )


def recover_range_audio(
    video: Path,
    transcript: dict[str, object],
    output_dir: Path,
    range_start: float,
    range_end: float,
) -> dict[str, object] | None:
    if not WHISPER_PYTHON.is_file():
        print(
            f"No cues found and Whisper environment is missing: "
            f"{WHISPER_PYTHON}"
        )
        return None

    recovery_path = output_dir / (
        f"audio-recovery-{range_start:.3f}-{range_end:.3f}.json"
    )
    command = [
        str(WHISPER_PYTHON),
        str(SCRIPT_DIR / "recover_chinese_frame_range.py"),
        str(video),
        "--start",
        f"{range_start:.6f}",
        "--end",
        f"{range_end:.6f}",
        "--output",
        str(recovery_path),
    ]
    print("\nNo subtitle cues found; testing multiple vocal-volume profiles")
    print(f"  {subprocess.list2cmdline(command)}")
    subprocess.run(command, cwd=SCRIPT_DIR, check=True)
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    if not recovery.get("accepted"):
        print("  no reliable Chinese speech recovered")
        return recovery

    recovered_cues = list(recovery.get("cues", []))
    recovered_words = list(recovery.get("words", []))
    profile = str(recovery.get("best_profile", "unknown"))
    for cue in recovered_cues:
        cue["audio_recovery_profile"] = profile
    for word in recovered_words:
        word["audio_recovery_profile"] = profile

    transcript["cues"] = sorted(
        [*transcript.get("cues", []), *recovered_cues],
        key=lambda cue: (float(cue["start"]), float(cue["end"])),
    )
    transcript["words"] = sorted(
        [*transcript.get("words", []), *recovered_words],
        key=lambda word: (float(word["start"]), float(word["end"])),
    )
    print(
        f"  recovered {len(recovered_cues)} cue(s) using {profile}"
    )
    return recovery


def read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def overlaps(
    start: float,
    end: float,
    item_start: float,
    item_end: float,
) -> bool:
    return item_end >= start and item_start <= end


def words_for_cue(
    words: list[dict[str, object]],
    cue: dict[str, object],
) -> list[dict[str, object]]:
    start = float(cue["start"])
    end = float(cue["end"])
    return [
        word
        for word in words
        if overlaps(
            start,
            end,
            float(word["start"]),
            float(word["end"]),
        )
    ]


def mean_word_probability(words: list[dict[str, object]]) -> float | None:
    values = [
        float(word["probability"])
        for word in words
        if word.get("probability") is not None
    ]
    return sum(values) / len(values) if values else None


def cue_frame_times(
    start: float,
    end: float,
    duration: float,
    context: float,
) -> list[float]:
    middle = (start + end) / 2
    values = [
        max(0.0, start - context),
        start + min(0.15, max(0.0, end - start) / 4),
        middle,
        max(start, end - min(0.15, max(0.0, end - start) / 4)),
        min(duration, end + context),
    ]
    result: list[float] = []
    for value in values:
        value = min(max(0.0, value), max(0.0, duration - 0.001))
        if not result or abs(value - result[-1]) >= 0.05:
            result.append(value)
    return result


def probe_duration(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return float(result.stdout.strip())


def probe_frame_rate(video: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    numerator, separator, denominator = result.stdout.strip().partition("/")
    fps = (
        float(numerator) / float(denominator)
        if separator
        else float(numerator)
    )
    if fps <= 0:
        raise ValueError(f"Invalid video frame rate: {result.stdout.strip()}")
    return fps


def extract_frame(
    video: Path,
    timestamp: float,
    output: Path,
    width: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2:flags=lanczos",
            "-q:v",
            "3",
            str(output),
        ],
        check=True,
    )


def nearby_dialogue(
    cues: list[dict[str, object]],
    index: int,
    distance: int = 2,
) -> list[dict[str, object]]:
    start = max(0, index - distance)
    end = min(len(cues), index + distance + 1)
    return [
        {
            "cue": position + 1,
            "start": float(cues[position]["start"]),
            "end": float(cues[position]["end"]),
            "text": str(cues[position]["text"]),
            "current": position == index,
        }
        for position in range(start, end)
    ]


def event_time(event: dict[str, object]) -> float:
    return float(event.get("timestamp", event.get("start", 0.0)))


def collect_gameplay_context(
    start: float,
    end: float,
    hud: dict[str, object] | None,
    counter: dict[str, object] | None,
    keystrokes: dict[str, object] | None,
) -> dict[str, object]:
    context_start = max(0.0, start - 1.0)
    context_end = end + 1.0

    hud_events = []
    if hud:
        hud_events = [
            event
            for event in hud.get("events", [])
            if context_start <= event_time(event) <= context_end
        ][:40]

    key_events = []
    if keystrokes:
        key_events = [
            event
            for event in keystrokes.get("events", [])
            if context_start <= event_time(event) <= context_end
        ][:30]

    counter_states = []
    if counter:
        counter_states = [
            item
            for item in counter.get("timeline", [])
            if overlaps(
                context_start,
                context_end,
                float(item["start"]),
                float(item["end"]),
            )
        ][:20]

    return {
        "hud_events": hud_events,
        "keystrokes": key_events,
        "counter_states": counter_states,
    }


def build_jobs(
    video: Path,
    transcript: dict[str, object],
    output_dir: Path,
    duration: float,
    frame_context: float,
    frame_width: int,
    force_frames: bool,
    start_cue: int,
    end_cue: int | None,
    range_start: float | None,
    range_end: float | None,
) -> list[dict[str, object]]:
    cues = list(transcript.get("cues", []))
    words = list(transcript.get("words", []))
    artifact_root = output_root_for_video(video)
    hud = read_optional_json(artifact_root / "hud-states.json")
    counter = read_optional_json(artifact_root / "counter.json")
    keystrokes = read_optional_json(artifact_root / "keystrokes.json")
    frames_dir = output_dir / "frames"

    jobs: list[dict[str, object]] = []
    for index, cue in enumerate(cues):
        cue_number = index + 1
        if cue_number < start_cue:
            continue
        if end_cue is not None and cue_number > end_cue:
            continue
        start = float(cue["start"])
        end = float(cue["end"])
        if range_start is not None and range_end is not None:
            if not overlaps(range_start, range_end, start, end):
                continue
        cue_words = words_for_cue(words, cue)
        times = cue_frame_times(start, end, duration, frame_context)
        frame_records = []
        for frame_index, timestamp in enumerate(times):
            frame_path = frames_dir / (
                f"cue_{cue_number:05d}_{frame_index}_{timestamp:.3f}.jpg"
            )
            if force_frames or not frame_path.is_file():
                extract_frame(video, timestamp, frame_path, frame_width)
            frame_records.append(
                {
                    "timestamp": round(timestamp, 6),
                    "path": str(frame_path.resolve()),
                }
            )

        jobs.append(
            {
                "cue": cue_number,
                "start": start,
                "end": end,
                "raw_chinese": str(cue["text"]),
                "audio_recovery_profile": cue.get(
                    "audio_recovery_profile"
                ),
                "asr_mean_word_probability": mean_word_probability(cue_words),
                "asr_words": cue_words,
                "nearby_dialogue": nearby_dialogue(cues, index),
                "gameplay": collect_gameplay_context(
                    start, end, hud, counter, keystrokes
                ),
                "frames": frame_records,
            }
        )
        print(
            f"\r  context frames {len(jobs)}/{max(1, len(cues))} "
            f"(cue {cue_number})",
            end="",
            flush=True,
        )
    print()
    return jobs


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def api_messages(job: dict[str, object]) -> list[dict[str, object]]:
    compact_job = dict(job)
    compact_job["frames"] = [
        {
            "timestamp": frame["timestamp"],
            "label": Path(str(frame["path"])).name,
        }
        for frame in job["frames"]
    ]
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                "Analyze this subtitle cue and its gameplay context.\n"
                + json.dumps(compact_job, ensure_ascii=False)
            ),
        }
    ]
    for frame in job["frames"]:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/jpeg;base64,"
                        + encode_image(Path(str(frame["path"])))
                    )
                },
            }
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def ollama_api_url(api_url: str) -> str | None:
    parsed = urllib.parse.urlparse(api_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        return None
    if parsed.path.rstrip("/") not in {
        "/v1/chat/completions",
        "/api/chat",
    }:
        return None
    return urllib.parse.urlunparse(
        (parsed.scheme or "http", parsed.netloc, "/api/chat", "", "", "")
    )


def ollama_messages(job: dict[str, object]) -> list[dict[str, object]]:
    compact_job = dict(job)
    frames = list(compact_job.pop("frames"))
    compact_job["frame_timestamps"] = [
        frame["timestamp"] for frame in frames
    ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Analyze this subtitle cue and its gameplay context.\n"
                + json.dumps(compact_job, ensure_ascii=False)
            ),
            "images": [
                encode_image(Path(str(frame["path"]))) for frame in frames
            ],
        },
    ]


def reduced_job(job: dict[str, object], image_limit: int) -> dict[str, object]:
    compact = dict(job)
    frames = list(job["frames"])
    if image_limit < len(frames):
        if image_limit == 1:
            frames = [frames[len(frames) // 2]]
        else:
            indexes = [
                round(index * (len(frames) - 1) / (image_limit - 1))
                for index in range(image_limit)
            ]
            frames = [frames[index] for index in indexes]
    compact["frames"] = frames

    gameplay = dict(job["gameplay"])
    gameplay["hud_events"] = list(gameplay.get("hud_events", []))[:16]
    gameplay["keystrokes"] = list(gameplay.get("keystrokes", []))[:12]
    gameplay["counter_states"] = list(gameplay.get("counter_states", []))[:10]
    compact["gameplay"] = gameplay
    compact["asr_words"] = list(job.get("asr_words", []))[:40]
    return compact


def extract_message_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response has no choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        if not content.strip() and message.get("reasoning"):
            finish_reason = choices[0].get("finish_reason", "unknown")
            raise ValueError(
                "The model returned only reasoning and no final JSON "
                f"(finish_reason={finish_reason}). Some qwen3-vl Ollama "
                "builds ignore think=false. Use qwen2.5vl:7b for this "
                "structured translation pipeline."
            )
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )
    raise ValueError("API response has an unsupported message format.")


def extract_ollama_message_text(payload: dict[str, object]) -> str:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama response has no message.")
    content = str(message.get("content", ""))
    if not content.strip() and message.get("thinking"):
        raise ValueError(
            "Ollama returned only thinking and no final JSON. "
            "Use qwen2.5vl:7b instead of qwen3-vl for this pipeline."
        )
    return content


def ollama_progress(
    stop: threading.Event,
    cue: int,
    attempt: int,
    frames: int,
    started: float,
    generated: list[int],
) -> None:
    width = 24
    while not stop.wait(0.5):
        elapsed = time.perf_counter() - started
        position = int(elapsed * 3) % (width * 2 - 2)
        if position >= width:
            position = width * 2 - 2 - position
        bar = ["-"] * width
        bar[position] = "#"
        print(
            f"\r  Qwen cue {cue} attempt {attempt} "
            f"[{''.join(bar)}] {elapsed:6.1f}s "
            f"{frames} frame(s), {generated[0]} chars",
            end="",
            flush=True,
        )


def read_ollama_stream(
    response,
    cue: int,
    attempt: int,
    frames: int,
) -> dict[str, object]:
    started = time.perf_counter()
    stop = threading.Event()
    generated = [0]
    progress = threading.Thread(
        target=ollama_progress,
        args=(stop, cue, attempt, frames, started, generated),
        daemon=True,
    )
    progress.start()
    content: list[str] = []
    thinking: list[str] = []
    final_payload: dict[str, object] = {}
    try:
        for raw_line in response:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line.decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            message = payload.get("message", {})
            if isinstance(message, dict):
                value = str(message.get("content", ""))
                content.append(value)
                thinking.append(str(message.get("thinking", "")))
                generated[0] += len(value)
            final_payload = payload
    finally:
        stop.set()
        progress.join(timeout=1)
        elapsed = time.perf_counter() - started
        print(
            f"\r  Qwen cue {cue} attempt {attempt} "
            f"[{'#' * 24}] {elapsed:6.1f}s "
            f"{frames} frame(s), {generated[0]} chars"
        )
    final_payload["message"] = {
        "role": "assistant",
        "content": "".join(content),
        "thinking": "".join(thinking),
    }
    return final_payload


def parse_json_object(text: str) -> dict[str, object]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return JSON: {text[:200]}")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model JSON response is not an object.")
    return value


def call_api(
    job: dict[str, object],
    api_url: str,
    model: str,
    api_key: str | None,
    timeout: float,
    retries: int,
    context_size: int,
) -> dict[str, object]:
    last_error: Exception | None = None
    image_limits = (5, 3, 1)
    native_ollama_url = ollama_api_url(api_url)
    for image_limit in image_limits:
        request_job = reduced_job(job, image_limit)
        if native_ollama_url:
            request_url = native_ollama_url
            body = {
                "model": model,
                "messages": ollama_messages(request_job),
                "stream": True,
                "think": False,
                "format": "json",
                "options": {
                    "num_ctx": context_size,
                    "temperature": 0.1,
                    "num_predict": 512,
                },
            }
        else:
            request_url = api_url
            body = {
                "model": model,
                "messages": api_messages(request_job),
                "temperature": 0.1,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
            }
        request = urllib.request.Request(
            request_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {api_key}"}
                    if api_key
                    else {}
                ),
            },
            method="POST",
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=timeout
                ) as response:
                    if native_ollama_url:
                        payload = read_ollama_stream(
                            response,
                            int(job["cue"]),
                            attempt + 1,
                            len(request_job["frames"]),
                        )
                    else:
                        payload = json.loads(response.read().decode("utf-8"))
                text = (
                    extract_ollama_message_text(payload)
                    if native_ollama_url
                    else extract_message_text(payload)
                )
                return parse_json_object(text)
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"HTTP {error.code} from {request_url}: {details}"
                )
                if "exceed_context_size_error" in details:
                    break
                if attempt >= retries:
                    break
                time.sleep(2**attempt)
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                last_error = error
                if attempt >= retries:
                    break
                time.sleep(2**attempt)
    assert last_error is not None
    raise RuntimeError(f"Context API request failed: {last_error}")


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


def write_srt(
    path: Path,
    results: list[dict[str, object]],
    field: str,
) -> None:
    blocks = []
    for result in results:
        text = str(result.get(field, "")).strip()
        if not text:
            continue
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{srt_time(float(result['start']))} --> "
            f"{srt_time(float(result['end']))}\n"
            f"{text}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def normalized_result(
    job: dict[str, object],
    response: dict[str, object],
) -> dict[str, object]:
    confidence = min(1.0, max(0.0, float(response.get("confidence", 0.0))))
    raw_chinese = str(job["raw_chinese"]).strip()
    corrected_chinese = str(
        response.get("corrected_chinese", raw_chinese)
    ).strip()
    unchanged_short_fragment = (
        corrected_chinese == raw_chinese
        and len(re.sub(r"\s+", "", raw_chinese)) <= 4
    )
    if unchanged_short_fragment:
        confidence = min(confidence, 0.60)
    return {
        "cue": int(job["cue"]),
        "start": float(job["start"]),
        "end": float(job["end"]),
        "raw_chinese": raw_chinese,
        "corrected_chinese": corrected_chinese,
        "english": str(response.get("english", "")).strip(),
        "confidence": confidence,
        "needs_review": (
            bool(response.get("needs_review", False))
            or unchanged_short_fragment
        ),
        "reasoning": str(response.get("reasoning", "")).strip(),
        "asr_mean_word_probability": job.get("asr_mean_word_probability"),
        "frames": job["frames"],
    }


def print_result_summary(
    results: list[dict[str, object]],
    failures: list[dict[str, object]],
) -> None:
    print("\nLikely contextual translation")
    if not results and not failures:
        print("  no Chinese subtitle cues overlapped the requested range")
        return
    for result in results:
        confidence = float(result.get("confidence", 0.0))
        review = " [review]" if result.get("needs_review") else ""
        print(
            f"\n  cue {result['cue']} "
            f"{float(result['start']):.3f}s-{float(result['end']):.3f}s "
            f"confidence {confidence:.2f}{review}"
        )
        print(f"  Chinese: {result.get('corrected_chinese', '')}")
        print(f"  English: {result.get('english', '')}")
    for failure in failures:
        print(
            f"\n  cue {failure['cue']} "
            f"{float(failure['start']):.3f}s-{float(failure['end']):.3f}s "
            "[failed]"
        )
        print(f"  Chinese ASR: {failure.get('raw_chinese', '')}")
        print(f"  Error: {failure.get('error', '')}")


def main() -> None:
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.start_cue < 1:
        raise ValueError("--start-cue must be at least 1.")
    if args.end_cue is not None and args.end_cue < args.start_cue:
        raise ValueError("--end-cue cannot precede --start-cue.")
    if (args.start_frame is None) != (args.end_frame is None):
        raise ValueError(
            "--start-frame and --end-frame must be provided together."
        )
    if args.start_frame is not None:
        if args.start_frame < 0:
            raise ValueError("--start-frame cannot be negative.")
        if args.end_frame < args.start_frame:
            raise ValueError("--end-frame cannot precede --start-frame.")
    if args.frame_width < 160:
        raise ValueError("--frame-width must be at least 160.")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be greater than zero.")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative.")
    if not 0 <= args.review_threshold <= 1:
        raise ValueError("--review-threshold must be between zero and one.")
    if args.provider == "api" and (not args.api_url or not args.model):
        raise ValueError("API mode requires --api-url and --model.")

    transcript_path = find_transcript(video, args.transcript)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    artifact_root = output_root_for_video(video)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else artifact_root / "contextual-translation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video)
    source_fps = probe_frame_rate(video)
    range_start = (
        args.start_frame / source_fps
        if args.start_frame is not None
        else None
    )
    range_end = (
        (args.end_frame + 1) / source_fps
        if args.end_frame is not None
        else None
    )

    print(f"Video: {video}")
    print(f"Transcript: {transcript_path}")
    print(f"Output: {output_dir}")
    if range_start is not None and range_end is not None:
        print(
            f"Frame range: {args.start_frame}-{args.end_frame} "
            f"at {source_fps:g} fps "
            f"({range_start:.3f}s-{range_end:.3f}s)"
        )
        if (
            not args.no_audio_recovery
            and not transcript_has_range_cues(
                transcript, range_start, range_end
            )
        ):
            recover_range_audio(
                video,
                transcript,
                output_dir,
                range_start,
                range_end,
            )
    jobs = build_jobs(
        video,
        transcript,
        output_dir,
        duration,
        args.frame_context_seconds,
        args.frame_width,
        args.force_frames,
        args.start_cue,
        args.end_cue,
        range_start,
        range_end,
    )

    bundle_path = output_dir / f"{safe_name(video)}.context-jobs.json"
    bundle_path.write_text(
        json.dumps(
            {
                "video": str(video),
                "transcript": str(transcript_path),
                "source_fps": source_fps,
                "requested_frames": (
                    {
                        "start": args.start_frame,
                        "end": args.end_frame,
                    }
                    if args.start_frame is not None
                    else None
                ),
                "system_prompt": SYSTEM_PROMPT,
                "jobs": jobs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(jobs)} context jobs: {bundle_path}")
    if args.provider == "bundle":
        return

    results_path = output_dir / f"{safe_name(video)}.context-results.json"
    existing: dict[int, dict[str, object]] = {}
    if results_path.is_file() and not args.force_results:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        existing = {
            int(item["cue"]): item for item in payload.get("results", [])
        }

    api_key = os.environ.get(args.api_key_env)
    results: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for position, job in enumerate(jobs, start=1):
        cue_number = int(job["cue"])
        previous = existing.get(cue_number)
        reusable = (
            previous is not None
            and str(previous.get("raw_chinese", "")).strip()
            == str(job["raw_chinese"]).strip()
            and abs(float(previous.get("start", -1)) - float(job["start"]))
            < 0.05
            and abs(float(previous.get("end", -1)) - float(job["end"]))
            < 0.05
        )
        if reusable:
            result = previous
        else:
            try:
                response = call_api(
                    job,
                    args.api_url,
                    args.model,
                    api_key,
                    args.request_timeout,
                    args.retries,
                    args.api_context_size,
                )
                result = normalized_result(job, response)
            except Exception as error:
                failures.append(
                    {
                        "cue": cue_number,
                        "start": float(job["start"]),
                        "end": float(job["end"]),
                        "raw_chinese": str(job["raw_chinese"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(
                    f"\n  cue {cue_number} failed after retries: {error}"
                )
                results_path.write_text(
                    json.dumps(
                        {
                            "video": str(video),
                            "model": args.model,
                            "results": results,
                            "failures": failures,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                continue
        results.append(result)
        results_path.write_text(
            json.dumps(
                {
                    "video": str(video),
                    "model": args.model,
                    "results": results,
                    "failures": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"\r  contextual translation {position}/{len(jobs)} "
            f"({position / max(1, len(jobs)) * 100:5.1f}%)",
            end="",
            flush=True,
        )
    print()

    review = [
        result
        for result in results
        if bool(result.get("needs_review"))
        or float(result.get("confidence", 0.0)) < args.review_threshold
    ]
    review.extend(failures)
    review_path = output_dir / f"{safe_name(video)}.review.json"
    review_path.write_text(
        json.dumps({"review": review}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_srt(
        output_dir / f"{safe_name(video)}.corrected-chinese.srt",
        results,
        "corrected_chinese",
    )
    write_srt(
        output_dir / f"{safe_name(video)}.contextual-english.srt",
        results,
        "english",
    )
    print(f"Saved contextual results: {results_path}")
    print(f"Saved review queue with {len(review)} cue(s): {review_path}")
    print_result_summary(results, failures)
    if failures:
        print(
            f"{len(failures)} cue(s) failed and can be retried by running "
            "the same command again without --force-results."
        )


if __name__ == "__main__":
    main()
