"""Recover quiet Chinese speech from a selected time range at multiple gains."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from generate_chinese_vocal_srt import (
    COMMON_ASR_HALLUCINATIONS,
    apply_league_context_corrections,
    join_words,
    load_whisper_model,
    split_words_into_cues,
    transcribe_focused_window,
)


PROFILES = (
    ("raw", "highpass=f=80,lowpass=f=8000"),
    ("boost_8db", "highpass=f=80,lowpass=f=8000,volume=8dB,alimiter"),
    ("boost_16db", "highpass=f=80,lowpass=f=8000,volume=16dB,alimiter"),
    (
        "dynamic_normalized",
        "highpass=f=80,lowpass=f=8000,"
        "dynaudnorm=f=150:g=21:p=0.95:m=12",
    ),
)


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(errors="backslashreplace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--beam-size", type=int, default=10)
    return parser.parse_args()


def extract_profile(
    video: Path,
    output: Path,
    start: float,
    end: float,
    audio_filter: str,
) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(video),
            "-t",
            f"{end - start:.6f}",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            audio_filter,
            "-c:a",
            "pcm_s16le",
            str(output),
        ],
        check=True,
    )


def mean_probability(words: list[dict[str, object]]) -> float:
    values = [
        float(word["probability"])
        for word in words
        if word.get("probability") is not None
    ]
    return sum(values) / len(values) if values else 0.0


def compact_text(text: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:]+", "", text)


def candidate_score(
    candidate: dict[str, object],
    agreement_counts: dict[str, int],
) -> float:
    text = str(candidate["text"])
    compact = compact_text(text)
    if not compact:
        return -100.0
    if any(phrase in text for phrase in COMMON_ASR_HALLUCINATIONS):
        return -50.0
    confidence = float(candidate["mean_probability"])
    agreement = agreement_counts.get(compact, 1)
    length_bonus = min(len(compact), 16) * 0.01
    return confidence + (agreement - 1) * 0.18 + length_bonus


def main() -> None:
    configure_console()
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if args.end <= args.start:
        raise ValueError("--end must be greater than --start.")

    model = load_whisper_model(
        args.model,
        args.device,
        args.compute_type,
    )
    candidates: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix="quiet_speech_recovery_",
        dir=args.output.resolve().parent,
    ) as temporary:
        temporary_dir = Path(temporary)
        for profile, audio_filter in PROFILES:
            audio = temporary_dir / f"{profile}.wav"
            extract_profile(
                video,
                audio,
                args.start,
                args.end,
                audio_filter,
            )
            words = transcribe_focused_window(
                model,
                audio,
                args.start,
                args.beam_size,
            )
            cues = split_words_into_cues(words, 0.3, 4.0, 24)
            apply_league_context_corrections(cues)
            text = join_words(words)
            candidates.append(
                {
                    "profile": profile,
                    "text": text,
                    "mean_probability": mean_probability(words),
                    "words": words,
                    "cues": cues,
                }
            )
            print(
                f"  {profile}: {text or '[no speech]'} "
                f"(confidence {mean_probability(words):.2f})"
            )

    agreement_counts: dict[str, int] = {}
    for candidate in candidates:
        compact = compact_text(str(candidate["text"]))
        if compact:
            agreement_counts[compact] = agreement_counts.get(compact, 0) + 1
    for candidate in candidates:
        candidate["score"] = candidate_score(candidate, agreement_counts)

    best = max(candidates, key=lambda candidate: float(candidate["score"]))
    accepted = (
        bool(compact_text(str(best["text"])))
        and float(best["score"]) >= 0.45
    )
    payload = {
        "video": str(video),
        "start": args.start,
        "end": args.end,
        "accepted": accepted,
        "best_profile": best["profile"] if accepted else None,
        "words": best["words"] if accepted else [],
        "cues": best["cues"] if accepted else [],
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected: {best['profile']} "
        f"({best['text'] or 'no usable speech'})"
    )


if __name__ == "__main__":
    main()
