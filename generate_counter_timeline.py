from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REGION_PATTERN = re.compile(r'^\s*["\']?([^"\':]+)["\']?\s*:\s*$')
RULE_PATTERN = re.compile(
    r'^\s*-\s*if\s+["\']([^"\']+)["\']\s*,?\s*value\s*=\s*(-?\d+)\s*$',
    re.IGNORECASE,
)
IF_ELSE_PATTERN = re.compile(
    r'^\s*-\s*if\s+["\']([^"\']+)["\']\s*,\s*(-?\d+)'
    r'\s*,\s*else\s+(-?\d+)\s*$',
    re.IGNORECASE,
)


def parse_rules(path: Path) -> dict[str, dict[str, int]]:
    rules: dict[str, dict[str, int]] = {}
    defaults: dict[str, int] = {}
    current_region: str | None = None

    for line_number, original_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue

        region_match = REGION_PATTERN.match(original_line)
        if region_match and not line.startswith("-"):
            current_region = region_match.group(1).strip().casefold()
            rules.setdefault(current_region, {})
            continue

        rule_match = RULE_PATTERN.match(original_line)
        if rule_match and current_region is not None:
            state = rule_match.group(1).strip().casefold()
            rules[current_region][state] = int(rule_match.group(2))
            continue

        if_else_match = IF_ELSE_PATTERN.match(original_line)
        if if_else_match and current_region is not None:
            state = if_else_match.group(1).strip().casefold()
            rules[current_region][state] = int(if_else_match.group(2))
            defaults[current_region] = int(if_else_match.group(3))
            continue

        raise ValueError(
            f"Unsupported rule at {path}:{line_number}: {original_line}"
        )

    if not rules:
        raise ValueError(f"No counter rules were found in {path}.")
    return {
        region: {
            **state_rules,
            "__default__": defaults.get(region, 0),
        }
        for region, state_rules in rules.items()
    }


def state_at(
    segments: list[dict[str, object]],
    timestamp: float,
) -> str | None:
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if start <= timestamp < end:
            return str(segment["state"]).casefold()
    return None


def build_timeline(
    state_data: dict[str, object],
    rules: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    source_segments = state_data.get("segments", {})
    if not isinstance(source_segments, dict):
        raise ValueError("HUD-state JSON has no valid 'segments' object.")

    segments_by_region = {
        str(region).casefold(): region_segments
        for region, region_segments in source_segments.items()
        if isinstance(region_segments, list)
    }
    analysis_start = float(state_data.get("analysis_start", 0.0))
    analysis_end = float(
        state_data.get("analysis_end", state_data.get("duration", 0.0))
    )
    boundaries = {analysis_start, analysis_end}

    for region in rules:
        for segment in segments_by_region.get(region, []):
            boundaries.add(max(analysis_start, float(segment["start"])))
            boundaries.add(min(analysis_end, float(segment["end"])))

    ordered_boundaries = sorted(
        boundary
        for boundary in boundaries
        if analysis_start <= boundary <= analysis_end
    )
    raw_timeline: list[dict[str, object]] = []

    for start, end in zip(ordered_boundaries, ordered_boundaries[1:]):
        if end <= start:
            continue
        sample_time = start + (end - start) / 2
        states: dict[str, str | None] = {}
        contributions: dict[str, int] = {}

        for region, state_rules in rules.items():
            state = state_at(segments_by_region.get(region, []), sample_time)
            states[region] = state
            contributions[region] = state_rules.get(
                state or "", state_rules.get("__default__", 0)
            )

        raw_timeline.append(
            {
                "start": round(start, 6),
                "end": round(end, 6),
                "total": sum(contributions.values()),
                "contributions": contributions,
                "states": states,
            }
        )

    timeline: list[dict[str, object]] = []
    for interval in raw_timeline:
        if (
            timeline
            and timeline[-1]["total"] == interval["total"]
            and timeline[-1]["contributions"] == interval["contributions"]
        ):
            timeline[-1]["end"] = interval["end"]
        else:
            timeline.append(interval)
    return timeline


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def write_srt(path: Path, timeline: list[dict[str, object]]) -> None:
    blocks = []
    for index, interval in enumerate(timeline, start=1):
        blocks.append(
            f"{index}\n"
            f"{srt_timestamp(float(interval['start']))} --> "
            f"{srt_timestamp(float(interval['end']))}\n"
            f"{interval['total']}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply rules.md to a HUD-state timeline."
    )
    parser.add_argument("states", type=Path)
    parser.add_argument("--rules", type=Path, default=Path("rules.md"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--srt", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.states.is_file():
        raise FileNotFoundError(f"HUD-state JSON not found: {args.states}")
    if args.states.suffix.casefold() != ".json":
        raise ValueError(
            "generate_counter_timeline.py expects a .hud-states.json file, "
            f"not a video: {args.states}\n"
            "Run analyze_hud_states.py first."
        )

    rules = parse_rules(args.rules)
    try:
        state_data = json.loads(args.states.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"HUD-state input is not valid UTF-8 JSON: {args.states}"
        ) from error
    timeline = build_timeline(state_data, rules)

    output_path = args.output or args.states.with_name(
        f"{args.states.stem.removesuffix('.hud-states')}.counter.json"
    )
    srt_path = args.srt or output_path.with_suffix(".srt")
    output = {
        "source": str(args.states.resolve()),
        "rules": rules,
        "timeline": timeline,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_srt(srt_path, timeline)
    print(f"Saved {len(timeline)} counter intervals to {output_path}")
    print(f"Saved counter subtitles to {srt_path}")


if __name__ == "__main__":
    main()
