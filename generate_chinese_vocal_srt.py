from __future__ import annotations

import argparse
import json
import os
import re
import site
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
UVR_MODEL_DIR = SCRIPT_DIR / "models" / "audio-separator"
DEFAULT_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
DEFAULT_HOTWORDS = "Q W E R A D F Q1 Q2 盲僧 李青 皇子 老鼠 暗爪 摸眼 天雷破 神龙摆尾"
COMMON_ASR_HALLUCINATIONS = (
    "点赞订阅",
    "点赞",
    "订阅",
    "感谢观看",
    "谢谢观看",
    "转发打赏",
    "支持明镜",
    "字幕",
    "优优独播剧场",
)
LEAGUE_CONTEXT_CORRECTIONS = (
    ("做个推特棒", "这个局特棒"),
    ("做个推推棒", "这个局特棒"),
    ("这个推特棒", "这个局特棒"),
    ("这个局特别棒", "这个局特棒"),
    ("终于可以修了", "终于可以秀了"),
    ("能修的阵容", "能秀的阵容"),
)
DLL_DIRECTORY_HANDLES: list[object] = []


def default_uvr_executable() -> Path:
    if sys.platform == "win32":
        return (
            SCRIPT_DIR
            / ".venv-audio-separator"
            / "Scripts"
            / "audio-separator.exe"
        )
    return SCRIPT_DIR / ".venv-audio-separator" / "bin" / "audio-separator"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use UVR to isolate vocals, then transcribe Chinese speech with "
            "short Whisper cues. No English translation is performed."
        )
    )
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to a 'transcriptions' folder beside the video.",
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="Faster-Whisper compute type. Use int8_float16 if VRAM is tight.",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--uvr-model",
        default=DEFAULT_UVR_MODEL,
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
        "--vad-min-silence-ms",
        type=int,
        default=200,
        help="Minimum silence used by VAD to separate speech.",
    )
    parser.add_argument(
        "--split-pause-ms",
        type=int,
        default=300,
        help="Pause between recognized words that starts a new cue.",
    )
    parser.add_argument(
        "--max-cue-seconds",
        type=float,
        default=4.0,
        help="Maximum speech duration targeted for each cue.",
    )
    parser.add_argument(
        "--max-characters",
        type=int,
        default=24,
        help="Maximum visible characters targeted for each cue.",
    )
    parser.add_argument(
        "--hotwords",
        default=DEFAULT_HOTWORDS,
        help="Short recognition hints. This is not a translation prompt.",
    )
    parser.add_argument(
        "--keep-source-audio",
        action="store_true",
        help="Keep the pre-UVR WAV in addition to the isolated vocal stem.",
    )
    parser.add_argument(
        "--discard-vocals",
        action="store_true",
        help="Delete the isolated vocal WAV after transcription.",
    )
    parser.add_argument(
        "--no-refinement",
        action="store_true",
        help="Disable the targeted raw-audio recheck of suspicious cues.",
    )
    parser.add_argument(
        "--refine-min-word-probability",
        type=float,
        default=0.45,
        help="Recheck cues whose mean word probability falls below this value.",
    )
    parser.add_argument(
        "--refine-padding-seconds",
        type=float,
        default=1.5,
        help="Raw-audio context added before and after a suspicious cue.",
    )
    parser.add_argument(
        "--refine-beam-size",
        type=int,
        default=10,
        help="Beam size used for focused raw-audio rechecks.",
    )
    return parser.parse_args()


def choose_video() -> Path:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a Chinese gameplay video",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.mkv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        raise SystemExit("No video selected.")
    return Path(path)


def safe_name(path: Path) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', "_", path.stem).strip(" .")
    return value or "video"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"  {subprocess.list2cmdline(command)}")
    subprocess.run(command, check=True, cwd=SCRIPT_DIR, env=env)


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_audio(video: Path, output: Path, duration: float) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(output),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    last_percent = -1
    for line in process.stdout:
        key, separator, value = line.strip().partition("=")
        if not separator or key not in {"out_time_us", "out_time_ms"}:
            continue
        elapsed = int(value) / 1_000_000
        percent = min(100, round(elapsed / duration * 100))
        if percent != last_percent:
            print(
                f"\r  audio extraction {elapsed:8.1f}s / "
                f"{duration:8.1f}s ({percent:3d}%)",
                end="",
                flush=True,
            )
            last_percent = percent
    stderr = process.stderr.read() if process.stderr else ""
    return_code = process.wait()
    print()
    if return_code != 0:
        raise RuntimeError(stderr.strip() or "FFmpeg audio extraction failed.")


def isolate_vocals(
    source_audio: Path,
    output_dir: Path,
    model: str,
    executable: Path,
) -> Path:
    if not executable.is_file():
        raise FileNotFoundError(
            f"UVR CLI is missing: {executable}\n"
            "Create .venv-audio-separator and install audio-separator first."
        )
    model_path = UVR_MODEL_DIR / model
    if not model_path.is_file():
        raise FileNotFoundError(
            f"UVR model is missing: {model_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["AUDIO_SEPARATOR_MODEL_DIR"] = str(UVR_MODEL_DIR)
    run(
        [
            str(executable),
            str(source_audio),
            "--model_filename",
            model,
            "--single_stem",
            "Vocals",
            "--output_dir",
            str(output_dir),
            "--output_format",
            "WAV",
        ],
        env=env,
    )
    candidates = sorted(
        output_dir.glob("*Vocals*.wav"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError("UVR completed without producing a vocal WAV.")
    return candidates[0]


def register_nvidia_dll_directories() -> None:
    if sys.platform != "win32":
        return
    roots = [Path(sys.prefix) / "Lib" / "site-packages"]
    user_site = site.getusersitepackages()
    if isinstance(user_site, str):
        roots.append(Path(user_site))
    for root in roots:
        for relative in ("nvidia/cublas/bin", "nvidia/cudnn/bin"):
            directory = root / relative
            if not directory.is_dir():
                continue
            os.environ["PATH"] = (
                f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
            )
            try:
                DLL_DIRECTORY_HANDLES.append(
                    os.add_dll_directory(str(directory))
                )
            except OSError:
                pass


def load_whisper_model(
    model_name: str,
    device: str,
    compute_type: str,
) -> object:
    try:
        from faster_whisper import WhisperModel
    except ImportError as error:
        raise RuntimeError(
            "Run this script with .venv-whisperx\\Scripts\\python.exe."
        ) from error

    register_nvidia_dll_directories()
    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )


def collect_segment_words(
    segment: object,
    time_offset: float = 0.0,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    segment_words = getattr(segment, "words", None) or []
    if segment_words:
        for word in segment_words:
            result.append(
                {
                    "start": float(word.start) + time_offset,
                    "end": float(word.end) + time_offset,
                    "text": str(word.word),
                    "probability": getattr(word, "probability", None),
                    "segment_avg_logprob": getattr(
                        segment, "avg_logprob", None
                    ),
                }
            )
    elif str(segment.text).strip():
        result.append(
            {
                "start": float(segment.start) + time_offset,
                "end": float(segment.end) + time_offset,
                "text": str(segment.text).strip(),
                "probability": None,
                "segment_avg_logprob": getattr(
                    segment, "avg_logprob", None
                ),
            }
        )
    return result


def transcribe_words(
    audio: Path,
    model: object,
    beam_size: int,
    min_silence_ms: int,
    hotwords: str,
    duration: float,
) -> list[dict[str, object]]:
    segments, _info = model.transcribe(
        str(audio),
        language="zh",
        task="transcribe",
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": min_silence_ms},
        word_timestamps=True,
        condition_on_previous_text=False,
        hotwords=hotwords or None,
    )

    words: list[dict[str, object]] = []
    last_percent = -1
    for segment in segments:
        words.extend(collect_segment_words(segment))
        percent = min(100, round(float(segment.end) / duration * 100))
        if percent != last_percent:
            print(
                f"\r  Chinese transcription {float(segment.end):8.1f}s / "
                f"{duration:8.1f}s ({percent:3d}%)",
                end="",
                flush=True,
            )
            last_percent = percent
    print()
    return words


def visible_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def joins_without_space(previous: str, current: str) -> bool:
    if not previous or not current:
        return True
    return bool(
        re.search(r"[\u3400-\u9fff]$", previous)
        or re.match(r"^[\u3400-\u9fff，。！？、；：]", current)
    )


def join_words(words: list[dict[str, object]]) -> str:
    result = ""
    for word in words:
        value = str(word["text"]).strip()
        if not value:
            continue
        if result and not joins_without_space(result, value):
            result += " "
        result += value
    return re.sub(r"\s+", " ", result).strip()


def split_words_into_cues(
    words: list[dict[str, object]],
    pause_seconds: float,
    maximum_seconds: float,
    maximum_characters: int,
) -> list[dict[str, object]]:
    cues: list[dict[str, object]] = []
    current: list[dict[str, object]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = join_words(current)
        if text:
            cues.append(
                {
                    "start": float(current[0]["start"]),
                    "end": float(current[-1]["end"]),
                    "text": text,
                }
            )
        current = []

    for word in words:
        if current:
            gap = float(word["start"]) - float(current[-1]["end"])
            duration = float(word["end"]) - float(current[0]["start"])
            projected_text = join_words([*current, word])
            punctuation_boundary = bool(
                re.search(r"[。！？!?；;]$", str(current[-1]["text"]).strip())
            )
            if (
                gap >= pause_seconds
                or duration > maximum_seconds
                or visible_length(projected_text) > maximum_characters
                or punctuation_boundary
            ):
                flush()
        current.append(word)
    flush()
    return merge_short_continuations(cues, maximum_seconds)


def apply_league_context_corrections(
    cues: list[dict[str, object]],
) -> list[dict[str, object]]:
    corrections: list[dict[str, object]] = []
    for cue in cues:
        original = str(cue["text"])
        corrected = original
        for wrong, right in LEAGUE_CONTEXT_CORRECTIONS:
            corrected = corrected.replace(wrong, right)
        if corrected == original:
            continue
        cue["text"] = corrected
        corrections.append(
            {
                "start": float(cue["start"]),
                "end": float(cue["end"]),
                "original": original,
                "corrected": corrected,
            }
        )
    return corrections


def cue_mean_probability(
    cue: dict[str, object],
    words: list[dict[str, object]],
) -> float | None:
    probabilities = [
        float(word["probability"])
        for word in words
        if word.get("probability") is not None
        and float(word["end"]) > float(cue["start"])
        and float(word["start"]) < float(cue["end"])
    ]
    if not probabilities:
        return None
    return sum(probabilities) / len(probabilities)


def suspicious_cue(
    cue: dict[str, object],
    words: list[dict[str, object]],
    minimum_probability: float,
) -> bool:
    text = str(cue["text"]).replace(" ", "")
    if any(phrase in text for phrase in COMMON_ASR_HALLUCINATIONS):
        return True
    duration = float(cue["end"]) - float(cue["start"])
    if visible_length(text) <= 2 and duration >= 1.2:
        return True
    mean_probability = cue_mean_probability(cue, words)
    return (
        mean_probability is not None
        and mean_probability < minimum_probability
    )


def merged_refinement_windows(
    cues: list[dict[str, object]],
    words: list[dict[str, object]],
    duration: float,
    padding: float,
    minimum_probability: float,
) -> list[tuple[float, float]]:
    windows = [
        (
            max(0.0, float(cue["start"]) - padding),
            min(duration, float(cue["end"]) + padding),
        )
        for cue in cues
        if suspicious_cue(cue, words, minimum_probability)
    ]
    merged: list[list[float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1] + 0.25:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def extract_audio_window(
    source_audio: Path,
    output: Path,
    start: float,
    end: float,
) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source_audio),
            "-t",
            f"{end - start:.3f}",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )


def transcribe_focused_window(
    model: object,
    audio: Path,
    time_offset: float,
    beam_size: int,
) -> list[dict[str, object]]:
    segments, _info = model.transcribe(
        str(audio),
        language="zh",
        task="transcribe",
        beam_size=beam_size,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    words: list[dict[str, object]] = []
    for segment in segments:
        words.extend(collect_segment_words(segment, time_offset))
    return words


def refine_suspicious_words(
    source_audio: Path,
    temporary_dir: Path,
    model: object,
    words: list[dict[str, object]],
    cues: list[dict[str, object]],
    duration: float,
    padding: float,
    minimum_probability: float,
    beam_size: int,
) -> tuple[list[dict[str, object]], int]:
    windows = merged_refinement_windows(
        cues,
        words,
        duration,
        padding,
        minimum_probability,
    )
    if not windows:
        print("  no suspicious cues found")
        return words, 0

    refined = list(words)
    accepted = 0
    for index, (start, end) in enumerate(windows, start=1):
        clip = temporary_dir / f"refine_{index:04d}.wav"
        extract_audio_window(source_audio, clip, start, end)
        candidate = transcribe_focused_window(
            model,
            clip,
            start,
            beam_size,
        )
        candidate_text = join_words(candidate)
        candidate_probabilities = [
            float(word["probability"])
            for word in candidate
            if word.get("probability") is not None
        ]
        candidate_probability = (
            sum(candidate_probabilities) / len(candidate_probabilities)
            if candidate_probabilities
            else 0.0
        )
        if not candidate_text or any(
            phrase in candidate_text
            for phrase in COMMON_ASR_HALLUCINATIONS
        ) or candidate_probability < minimum_probability:
            print(
                f"  refinement {index}/{len(windows)} rejected "
                f"({start:.2f}s-{end:.2f}s, "
                f"confidence {candidate_probability:.2f})"
            )
            continue

        refined = [
            word
            for word in refined
            if float(word["end"]) <= start or float(word["start"]) >= end
        ]
        refined.extend(candidate)
        refined.sort(key=lambda word: (float(word["start"]), float(word["end"])))
        accepted += 1
        print(
            f"  refinement {index}/{len(windows)} accepted "
            f"({start:.2f}s-{end:.2f}s, "
            f"confidence {candidate_probability:.2f}): {candidate_text}"
        )
    return refined, accepted


def merge_short_continuations(
    cues: list[dict[str, object]],
    maximum_seconds: float,
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for cue in cues:
        text = str(cue["text"])
        if merged:
            previous = merged[-1]
            previous_text = str(previous["text"])
            gap = float(cue["start"]) - float(previous["end"])
            duration = float(cue["end"]) - float(cue["start"])
            is_single_chinese_character = bool(
                re.fullmatch(r"[\u3400-\u9fff]", text)
            )
            if (
                is_single_chinese_character
                and duration <= 0.4
                and gap <= 2.0
                and visible_length(previous_text) <= 8
                and re.search(r"[\u3400-\u9fff]$", previous_text)
            ):
                previous["text"] = previous_text + text
                previous["end"] = min(
                    float(cue["end"]),
                    float(previous["start"]) + maximum_seconds,
                )
                continue
        merged.append(cue)
    return merged


def srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
        f"{milliseconds:03d}"
    )


def write_srt(path: Path, cues: list[dict[str, object]]) -> None:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        blocks.append(
            f"{index}\n"
            f"{srt_time(float(cue['start']))} --> "
            f"{srt_time(float(cue['end']))}\n"
            f"{cue['text']}"
        )
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    video = (args.video or choose_video()).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.max_cue_seconds <= 0:
        raise ValueError("--max-cue-seconds must be greater than zero.")
    if args.max_characters <= 0:
        raise ValueError("--max-characters must be greater than zero.")
    if not 0 <= args.refine_min_word_probability <= 1:
        raise ValueError(
            "--refine-min-word-probability must be between zero and one."
        )
    if args.refine_padding_seconds < 0:
        raise ValueError("--refine-padding-seconds cannot be negative.")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else video.parent / "transcriptions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    base = safe_name(video)
    vocal_path = output_dir / f"{base}.uvr.vocals.wav"
    srt_path = output_dir / f"{base}.uvr.whisperx.chinese.short.srt"
    debug_path = srt_path.with_suffix(".json")
    duration = probe_duration(video)
    uvr_executable = args.uvr_executable.resolve()

    print(f"Video: {video}")
    print(f"Output: {output_dir}")
    print(f"Duration: {duration:.1f}s")

    with tempfile.TemporaryDirectory(
        prefix="chinese_vocal_srt_",
        dir=output_dir,
    ) as temporary:
        temporary_dir = Path(temporary)
        source_audio = (
            output_dir / f"{base}.source.wav"
            if args.keep_source_audio
            else temporary_dir / "source.wav"
        )

        print("\n[1/5] Extracting audio")
        extract_audio(video, source_audio, duration)

        print("\n[2/5] Isolating vocals with UVR")
        generated_vocal = isolate_vocals(
            source_audio,
            temporary_dir / "uvr",
            args.uvr_model,
            uvr_executable,
        )
        if vocal_path.exists():
            vocal_path.unlink()
        generated_vocal.replace(vocal_path)

        print("\n[3/5] Transcribing Chinese vocals")
        model = load_whisper_model(
            args.model,
            args.device,
            args.compute_type,
        )
        words = transcribe_words(
            vocal_path,
            model,
            args.beam_size,
            args.vad_min_silence_ms,
            args.hotwords,
            duration,
        )

        preliminary_cues = split_words_into_cues(
            words,
            args.split_pause_ms / 1000,
            args.max_cue_seconds,
            args.max_characters,
        )
        print("\n[4/5] Refining suspicious cues from original audio")
        refinement_count = 0
        if args.no_refinement:
            print("  refinement disabled")
        else:
            words, refinement_count = refine_suspicious_words(
                source_audio,
                temporary_dir,
                model,
                words,
                preliminary_cues,
                duration,
                args.refine_padding_seconds,
                args.refine_min_word_probability,
                args.refine_beam_size,
            )

    print("\n[5/5] Building short Chinese cues")
    cues = split_words_into_cues(
        words,
        args.split_pause_ms / 1000,
        args.max_cue_seconds,
        args.max_characters,
    )
    context_corrections = apply_league_context_corrections(cues)
    if context_corrections:
        print(
            f"  applied {len(context_corrections)} conservative "
            "League-context correction(s)"
        )
    write_srt(srt_path, cues)
    debug_path.write_text(
        json.dumps(
            {
                "video": str(video),
                "vocal_audio": None if args.discard_vocals else str(vocal_path),
                "model": args.model,
                "device": args.device,
                "compute_type": args.compute_type,
                "uvr_model": args.uvr_model,
                "raw_audio_refinements": refinement_count,
                "league_context_corrections": context_corrections,
                "words": words,
                "cues": cues,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.discard_vocals:
        vocal_path.unlink(missing_ok=True)

    print(f"Saved {len(cues)} Chinese cues:")
    print(f"  {srt_path}")
    if args.discard_vocals:
        print("Discarded the isolated vocal WAV after transcription.")
    else:
        print(f"Saved isolated vocals:")
        print(f"  {vocal_path}")
    print(f"Saved recognition details:")
    print(f"  {debug_path}")


if __name__ == "__main__":
    main()
