from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


REGION_PATTERN = re.compile(r'^\s*["\']([^"\']+)["\']\s*:\s*$')
TRANSITION_PATTERN = re.compile(
    r'^\s*-\s*["\']([^"\']+)["\']\s*:\s*'
    r'["\']([^"\']+)["\']\s*->\s*(.+?)\s*$',
    re.IGNORECASE,
)
EXACT_STATE_PATTERN = re.compile(r'^["\']([^"\']+)["\']$')
ANY_STATE_PATTERN = re.compile(
    r'^any\s*state(?:\s+besides\s+(.+))?$',
    re.IGNORECASE,
)
QUOTED_STATE_PATTERN = re.compile(r'["\']([^"\']+)["\']')
RECAST_VALIDATION_SECONDS = 3.1
RECAST_NATURAL_EXPIRY_SECONDS = 3.0
ANY_STATE = "*"
EXCLUDED_DESTINATION_STATES = {"disabled", "missing"}


def console_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="backslashreplace").decode(encoding)


def parse_rules(path: Path) -> dict[str, list[dict[str, object]]]:
    rules: dict[str, list[dict[str, object]]] = {}
    current_region: str | None = None

    for line_number, original_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = original_line.strip()
        if not line or line.startswith("#"):
            continue

        region_match = REGION_PATTERN.match(original_line)
        if region_match:
            current_region = region_match.group(1).casefold()
            rules.setdefault(current_region, [])
            continue

        transition_match = TRANSITION_PATTERN.match(original_line)
        if transition_match and current_region is not None:
            target_expression = transition_match.group(3).strip()
            exact_match = EXACT_STATE_PATTERN.fullmatch(target_expression)
            any_match = ANY_STATE_PATTERN.fullmatch(target_expression)
            if exact_match:
                target_state = exact_match.group(1).casefold()
                excluded_states: list[str] = []
            elif any_match:
                target_state = ANY_STATE
                excluded_states = [
                    state.casefold()
                    for state in QUOTED_STATE_PATTERN.findall(
                        any_match.group(1) or ""
                    )
                ]
            else:
                raise ValueError(
                    f"Unsupported transition target at "
                    f"{path}:{line_number}: {target_expression}"
                )
            rules[current_region].append(
                {
                    "key": transition_match.group(1),
                    "from": transition_match.group(2).casefold(),
                    "to": target_state,
                    "exclude": excluded_states,
                }
            )
            continue

        # Introductory prose before the first region is documentation.
        if current_region is None:
            continue
        raise ValueError(
            f"Unsupported keystroke rule at {path}:{line_number}: {original_line}"
        )

    if not rules:
        raise ValueError(f"No keystroke rules were found in {path}.")
    return rules


def previous_region_events(
    events: list[dict[str, object]],
) -> dict[int, dict[str, object] | None]:
    grouped: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for index, event in enumerate(events):
        grouped[str(event["region"]).casefold()].append((index, event))

    previous: dict[int, dict[str, object] | None] = {}
    for region_events in grouped.values():
        for position, (index, _event) in enumerate(region_events):
            previous[index] = (
                region_events[position - 1][1]
                if position > 0
                else None
            )
    return previous


def recast_episode_start(
    events: list[dict[str, object]],
    event_index: int,
) -> dict[str, object] | None:
    region = str(events[event_index].get("region", "")).casefold()
    candidate: dict[str, object] | None = None
    for previous_index in range(event_index - 1, -1, -1):
        previous = events[previous_index]
        if str(previous.get("region", "")).casefold() != region:
            continue
        state = str(previous.get("state", "")).casefold()
        if state == "recast":
            candidate = previous
            continue
        if state in EXCLUDED_DESTINATION_STATES and candidate is not None:
            continue
        break
    return candidate


def infer_keystrokes(
    state_data: dict[str, object],
    rules: dict[str, list[dict[str, object]]],
    recast_timeout: float = RECAST_VALIDATION_SECONDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_events = state_data.get("events", [])
    if not isinstance(source_events, list):
        raise ValueError("HUD-state JSON has no valid 'events' array.")

    transition_lookup = {
        (region, str(rule["from"]), str(rule["to"])): rule
        for region, region_rules in rules.items()
        for rule in region_rules
    }
    preceding = previous_region_events(source_events)
    keystrokes: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for index, event in enumerate(source_events):
        region = str(event.get("region", "")).casefold()
        previous = event.get("previous_state")
        if previous is None:
            continue
        previous_state = str(previous).casefold()
        state = str(event.get("state", "")).casefold()
        if state in EXCLUDED_DESTINATION_STATES:
            continue
        matched_rule = transition_lookup.get((region, previous_state, state))
        if matched_rule is None:
            matched_rule = transition_lookup.get(
                (region, previous_state, ANY_STATE)
            )
        if matched_rule is None:
            continue
        if state in matched_rule.get("exclude", []):
            continue
        key = str(matched_rule["key"])

        timestamp = float(event["timestamp"])
        validation: dict[str, object] | None = None
        if previous_state == "recast":
            recast_event = recast_episode_start(source_events, index)
            elapsed = (
                timestamp - float(recast_event["timestamp"])
                if recast_event is not None
                and str(recast_event.get("state", "")).casefold() == "recast"
                else None
            )
            if elapsed is None or elapsed > recast_timeout:
                rejected.append(
                    {
                        "region": str(event["region"]),
                        "key": key,
                        "timestamp": timestamp,
                        "frame": int(event["frame"]),
                        "reason": "recast_did_not_change_within_timeout",
                        "recast_duration_seconds": elapsed,
                    }
                )
                continue
            if state == "ready" and elapsed >= RECAST_NATURAL_EXPIRY_SECONDS:
                rejected.append(
                    {
                        "region": str(event["region"]),
                        "key": key,
                        "timestamp": timestamp,
                        "frame": int(event["frame"]),
                        "reason": "recast_expired_naturally",
                        "recast_duration_seconds": round(elapsed, 6),
                    }
                )
                continue
            validation = {
                "recast_start_timestamp": float(recast_event["timestamp"]),
                "recast_duration_seconds": round(elapsed, 6),
            }

        inferred = {
            "timestamp": timestamp,
            "frame": int(event["frame"]),
            "region": str(event["region"]),
            "key": key,
            "transition": {
                "from": previous_state,
                "to": state,
            },
            "confidence": float(event.get("confidence", 0.0)),
        }
        if validation is not None:
            inferred["recast_validation"] = validation
        keystrokes.append(inferred)

    keystrokes.sort(key=lambda entry: (float(entry["timestamp"]), str(entry["key"])))
    rejected.sort(key=lambda entry: float(entry["timestamp"]))
    return keystrokes, rejected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer keystrokes from HUD-state transitions."
    )
    parser.add_argument("states", type=Path)
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("keystrokerules.md"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recast-timeout",
        type=float,
        default=RECAST_VALIDATION_SECONDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recast_timeout <= 0:
        raise ValueError("--recast-timeout must be greater than zero.")

    rules = parse_rules(args.rules)
    state_data = json.loads(args.states.read_text(encoding="utf-8"))
    keystrokes, rejected = infer_keystrokes(
        state_data, rules, args.recast_timeout
    )
    output = args.output or args.states.with_name(
        f"{args.states.stem.removesuffix('.hud-states')}.keystrokes.json"
    )
    payload = {
        "source": str(args.states.resolve()),
        "recast_timeout_seconds": args.recast_timeout,
        "rules": rules,
        "events": keystrokes,
        "rejected_candidates": rejected,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(console_text(f"Saved {len(keystrokes)} inferred keystrokes to {output}"))
    print(f"Rejected {len(rejected)} unconfirmed recast candidates.")


if __name__ == "__main__":
    main()
